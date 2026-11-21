import argparse
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import boltz.main as main

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["stock", "geometric", "euler", "heun", "dpm2m"], required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--schedule", choices=["karras", "geometric"], default="geometric")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--history-weight", type=float, default=1.0)
parser.add_argument("--tail-sigma", type=float, default=0.0)
parser.add_argument("--grid-power", type=float, default=1.0)
parser.add_argument("--dpm-form", choices=["midpoint", "taylor"], default="midpoint")
parser.add_argument("--dpm-study", action="store_true")
parser.add_argument("--ligand-history-weight", type=float, default=1.0)
parser.add_argument("--input", type=Path, default=Path("data/smoke.yaml"))
parser.add_argument("--single-pass", action="store_true")
args = parser.parse_args()
name = f"{args.method}_{args.schedule}_{args.steps}_s{args.seed}"
if args.method == "dpm2m":
    name += f"_w{args.history_weight}_tail{args.tail_sigma}_p{args.grid_power}_{args.dpm_form}"
    name += f"_lig{args.ligand_history_weight}"
if args.dpm_study:
    name += "_study"
out_dir = Path("results/training_free") / name
if args.input.parent.name == "stability":
    out_dir = Path("results/stability") / args.input.stem / name
out_dir.mkdir(parents=True, exist_ok=True)
report = {"method": args.method, "schedule": args.schedule, "steps": args.steps,
          "seed": args.seed, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
          "stages": [], "execution": "stock eager; no approximate kernels"}
report["input"] = str(args.input)
report["dpm_parameters"] = {"history_weight": args.history_weight, "tail_sigma": args.tail_sigma,
                            "grid_power": args.grid_power, "form": args.dpm_form}
report["dpm_parameters"]["ligand_history_weight"] = args.ligand_history_weight
original_predict = main.Trainer.predict


def geometric_schedule(diffusion, steps, power=1.0):
    u = torch.linspace(0, 1, steps, device=diffusion.device)
    # power < 1 spends more nodes at low noise; endpoints remain fixed.
    log_sigma = math.log(diffusion.sigma_max * diffusion.sigma_data) + u.pow(power) * math.log(
        diffusion.sigma_min / diffusion.sigma_max)
    return torch.cat((log_sigma.exp(), torch.zeros(1, device=diffusion.device)))


@functools.wraps(original_predict)
def predict(trainer, model, *a, **kw):
    diffusion = model.structure_module
    original_sample = diffusion.sample
    original_schedule = diffusion.sample_schedule
    original_denoise = diffusion.preconditioned_network_forward
    stage = {"stage": "affinity" if model.affinity_prediction else "structure",
             "original_step_scale": diffusion.step_scale,
             "original_gamma_0": diffusion.gamma_0,
             "original_alignment_reverse_diff": diffusion.alignment_reverse_diff,
             "denoiser_calls": 0, "sample_equivalent_nfe": 0}
    stage["teacher_steering"] = dict(model.steering_args)
    if args.method in ("euler", "heun", "dpm2m"):
        model.steering_args = dict(model.steering_args)
        for key in ("fk_steering", "physical_guidance_update", "contact_guidance_update"):
            model.steering_args[key] = False
    stage["active_steering"] = dict(model.steering_args)
    if args.method == "geometric":
        diffusion.sample_schedule = lambda num_sampling_steps=None: geometric_schedule(
            diffusion, num_sampling_steps or diffusion.num_sampling_steps)

    def denoise(coords, sigma, network_condition_kwargs):
        stage["denoiser_calls"] += 1
        stage["sample_equivalent_nfe"] += coords.shape[0]
        return original_denoise(coords, sigma, network_condition_kwargs)

    diffusion.preconditioned_network_forward = denoise

    @torch.no_grad()
    def ode_sample(atom_mask, num_sampling_steps=None, multiplicity=1,
                   max_parallel_samples=None, steering_args=None, solver=None,
                   history_weight=None, tail_sigma=None, grid_power=None,
                   dpm_form=None, ligand_history_weight=None, **conditions):
        assert not any(steering_args[key] for key in
                       ("fk_steering", "physical_guidance_update", "contact_guidance_update"))
        solver = solver or args.method
        weight = args.history_weight if history_weight is None else history_weight
        tail = args.tail_sigma if tail_sigma is None else tail_sigma
        power = args.grid_power if grid_power is None else grid_power
        form = dpm_form or args.dpm_form
        steps = num_sampling_steps or diffusion.num_sampling_steps
        sigmas = (geometric_schedule(diffusion, steps, power) if args.schedule == "geometric"
                  else original_schedule(steps)).tolist()
        x = sigmas[0] * torch.randn((atom_mask.shape[0] * multiplicity,
                                    atom_mask.shape[1], 3), device=diffusion.device)
        # Fixed coordinate frame, no churn/alignment, unit-scale EDM ODE.
        previous_denoised = None
        previous_h = None
        max_parallel = max_parallel_samples or multiplicity
        chunks = torch.arange(x.shape[0], device=x.device).chunk(multiplicity % max_parallel + 1)
        token_ids = conditions["feats"]["atom_to_token"][0].argmax(-1)
        ligand_atoms = conditions["feats"]["mol_type"][0][token_ids] == 3
        history_weights = torch.ones((1, x.shape[1], 1), device=x.device)
        history_weights[:, ligand_atoms] = (args.ligand_history_weight if ligand_history_weight is None
                                           else ligand_history_weight)

        def evaluate(coords, sigma):
            output = torch.empty_like(coords)
            for ids in chunks:
                output[ids] = denoise(coords[ids], float(sigma), dict(
                    multiplicity=ids.numel(), **conditions))
            return output

        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            clean = evaluate(x, sigma)
            if solver == "dpm2m":
                if next_sigma == 0:
                    x = clean
                else:
                    h = math.log(sigma / next_sigma)
                    phi = math.expm1(-h)
                    first_order = (next_sigma / sigma) * x - phi * clean
                    if previous_denoised is not None and sigma > tail:
                        derivative = (h / previous_h) * (clean - previous_denoised)
                        coefficient = -0.5 * phi if form == "midpoint" else phi / h + 1
                        x = first_order + weight * coefficient * history_weights * derivative
                    else:
                        x = first_order
                    previous_h = h
                previous_denoised = clean
            else:
                drift = (x - clean) / sigma
                proposed = x + (next_sigma - sigma) * drift
                if solver == "heun" and next_sigma > 0:
                    next_clean = evaluate(proposed, next_sigma)
                    next_drift = (proposed - next_clean) / next_sigma
                    x = x + (next_sigma - sigma) * (drift + next_drift) * 0.5
                else:
                    x = proposed
        return {"sample_atom_coords": x, "diff_token_repr": None}

    chosen_sample = original_sample if args.method in ("stock", "geometric") else ode_sample

    @functools.wraps(original_sample)
    def measured_sample(*x, **y):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = chosen_sample(*x, **y)
        torch.cuda.synchronize()
        stage["sample_seconds"] = time.perf_counter() - start
        stage["denoiser_calls_first"] = stage["denoiser_calls"]
        stage["sample_equivalent_nfe_first"] = stage["sample_equivalent_nfe"]
        after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        # Same trajectory after kernel warmup; downstream consumes the first result.
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        torch.cuda.synchronize()
        start = time.perf_counter()
        warm = value if args.single_pass else chosen_sample(*x, **y)
        torch.cuda.synchronize()
        stage["warm_sample_seconds"] = None if args.single_pass else time.perf_counter() - start
        stage["repeat_max_abs_error"] = (warm["sample_atom_coords"] - value["sample_atom_coords"]).abs().max().item()
        torch.set_rng_state(after_cpu)
        torch.cuda.set_rng_state(after_cuda)
        coords = value["sample_atom_coords"]
        assert torch.isfinite(coords).all()
        feats = y["feats"]
        atom_tokens = feats["atom_to_token"][0].argmax(-1)
        ligand = feats["mol_type"][0][atom_tokens] == 3
        atom_mask = y["atom_mask"][0].bool()
        np.savez(out_dir / (stage["stage"] + "_coords.npz"),
                 coords=coords.float().cpu().numpy(), mask=atom_mask.cpu().numpy(),
                 ligand=ligand.cpu().numpy())
        if args.dpm_study:
            variants = [
                ("heun256", dict(solver="heun", num_sampling_steps=256)),
                ("heun512", dict(solver="heun", num_sampling_steps=512)),
            ]
            for steps in (16, 32):
                variants.extend([
                    (f"midpoint{steps}", dict(num_sampling_steps=steps)),
                    (f"taylor{steps}", dict(num_sampling_steps=steps, dpm_form="taylor")),
                    (f"damped{steps}", dict(num_sampling_steps=steps, history_weight=0.5)),
                    (f"tail{steps}", dict(num_sampling_steps=steps, tail_sigma=1.0)),
                    (f"low_noise_grid{steps}", dict(num_sampling_steps=steps, grid_power=0.75)),
                    (f"ligand_damped{steps}", dict(num_sampling_steps=steps, ligand_history_weight=0.5)),
                ])
            trajectories = {}
            stage["dpm_study"] = []
            for label, changes in variants:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                parameters = dict(y)
                parameters.update(solver="dpm2m", history_weight=1.0, tail_sigma=0.0,
                                  grid_power=1.0, dpm_form="midpoint", ligand_history_weight=1.0)
                parameters.update(changes)
                before_calls = stage["denoiser_calls"]
                torch.cuda.synchronize()
                start = time.perf_counter()
                trajectory = ode_sample(*x, **parameters)["sample_atom_coords"]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert torch.isfinite(trajectory).all()
                trajectories[label] = trajectory
                np.save(out_dir / f"{stage['stage']}_{label}.npy", trajectory.cpu().numpy())
                stage["dpm_study"].append({"variant": label, "seconds": elapsed,
                                           "calls": stage["denoiser_calls"] - before_calls})
            reference = trajectories["heun512"]
            for row in stage["dpm_study"]:
                trajectory = trajectories[row["variant"]]
                delta = trajectory - reference
                row["same_frame_rms_angstrom"] = delta[:, atom_mask].square().mean().sqrt().item()
                row["ligand_same_frame_rms_angstrom"] = delta[:, ligand & atom_mask].square().mean().sqrt().item()
                ligand_pair_delta = torch.cdist(trajectory[:, ligand & atom_mask], trajectory[:, ligand & atom_mask]) - torch.cdist(
                    reference[:, ligand & atom_mask], reference[:, ligand & atom_mask])
                row["ligand_pair_distance_rms_angstrom"] = ligand_pair_delta.square().mean().sqrt().item()
            torch.set_rng_state(after_cpu)
            torch.cuda.set_rng_state(after_cuda)
        return value

    diffusion.sample = measured_sample
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        value = original_predict(trainer, model, *a, **kw)
    finally:
        diffusion.sample = original_sample
        diffusion.sample_schedule = original_schedule
        diffusion.preconditioned_network_forward = original_denoise
    torch.cuda.synchronize()
    stage["trainer_seconds_including_warm_repeat"] = time.perf_counter() - start
    stage["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["stages"].append(stage)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(stage), flush=True)
    return value


main.Trainer.predict = predict
start = time.perf_counter()
main.cli.main(args=["predict", str(args.input), "--model", "boltz2", "--seed", str(args.seed),
    "--recycling_steps", "3", "--sampling_steps", str(args.steps), "--diffusion_samples", "1",
    "--sampling_steps_affinity", str(args.steps), "--diffusion_samples_affinity", "5",
    "--num_workers", "2", "--preprocessing-threads", "2", "--out_dir", str(out_dir),
    "--override"], standalone_mode=False)
report["cli_seconds_including_warm_repeat"] = time.perf_counter() - start
prediction = out_dir / f"boltz_results_{args.input.stem}/predictions/{args.input.stem}"
assert (prediction / f"{args.input.stem}_model_0.cif").is_file()
report["scores"] = json.loads((prediction / f"affinity_{args.input.stem}.json").read_text())
assert all(math.isfinite(v) for v in report["scores"].values())
assert 0 <= report["scores"]["affinity_probability_binary"] <= 1
(out_dir / "summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
