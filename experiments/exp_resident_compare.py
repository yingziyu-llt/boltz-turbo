import argparse
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import boltz.main as main

REPO_ROOT = Path(__file__).resolve().parent.parent

# 同一蛋白的芳香族氨基酸系列（与 exp_baseline.py 一致，无活性标签，仅用于稳态性能与一致性公平对比）。
BATCH6_LIGANDS = {
    "tyrosine": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "dopa": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
    "fluorophenylalanine": "N[C@@H](Cc1ccc(F)cc1)C(=O)O",
    "methylphenylalanine": "N[C@@H](Cc1ccc(C)cc1)C(=O)O",
    "tryptophan": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
}

PRESET_CONFIGS = {
    "stock_200": dict(
        structure_method="stock", structure_schedule="karras", structure_steps=200,
        structure_recycling=3, structure_samples=1,
        affinity_method="stock", affinity_schedule="karras", affinity_steps=200,
        affinity_recycling=5, affinity_samples=5,
    ),
    "stock_100": dict(
        structure_method="stock", structure_schedule="karras", structure_steps=100,
        structure_recycling=3, structure_samples=1,
        affinity_method="stock", affinity_schedule="karras", affinity_steps=100,
        affinity_recycling=5, affinity_samples=5,
    ),
    "stock_50": dict(
        structure_method="stock", structure_schedule="karras", structure_steps=50,
        structure_recycling=3, structure_samples=1,
        affinity_method="stock", affinity_schedule="karras", affinity_steps=50,
        affinity_recycling=5, affinity_samples=5,
    ),
    "geometric_50": dict(
        structure_method="geometric", structure_schedule="geometric", structure_steps=50,
        structure_recycling=3, structure_samples=1,
        affinity_method="geometric", affinity_schedule="geometric", affinity_steps=50,
        affinity_recycling=5, affinity_samples=5,
    ),
    "dpm2m_32": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=32,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=32,
        affinity_recycling=5, affinity_samples=5,
    ),
    "dpm2m_16": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
    ),
    "dpm2m_16_affkern": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
        affinity_use_kernels=True,
    ),
    "dpm2m_16_affkern_tc05": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
        affinity_use_kernels=True,
        affinity_teacache=dict(msa_cache_after=1, pf_adaptive_thresh=0.05),
    ),
    "dpm2m_16_affkern_tc03": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
        affinity_use_kernels=True,
        affinity_teacache=dict(msa_cache_after=1, pf_adaptive_thresh=0.03),
    ),
    "dpm2m_16_affkern_ops": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
        affinity_use_kernels=True,
        ops_opt=True,
    ),
    # 两阶段解耦对照配置
    "s_dpm16_a_stock200": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="stock", affinity_schedule="karras", affinity_steps=200,
        affinity_recycling=5, affinity_samples=5,
    ),
    "s_stock200_a_dpm16": dict(
        structure_method="stock", structure_schedule="karras", structure_steps=200,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=5,
    ),
    "dpm2m_16_affrec3": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=3, affinity_samples=5,
    ),
    "dpm2m_16_affrec1": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=1, affinity_samples=5,
    ),
    "dpm2m_16_affs1": dict(
        structure_method="dpm2m", structure_schedule="geometric", structure_steps=16,
        structure_recycling=3, structure_samples=1,
        affinity_method="dpm2m", affinity_schedule="geometric", affinity_steps=16,
        affinity_recycling=5, affinity_samples=1,
    ),
}

SUITES = {
    "fair": ["stock_200", "stock_100", "geometric_50", "dpm2m_32", "dpm2m_16"],
    "decoupled": [
        "stock_200", "s_dpm16_a_stock200", "s_stock200_a_dpm16",
        "dpm2m_16", "dpm2m_16_affrec3", "dpm2m_16_affrec1", "dpm2m_16_affs1",
    ],
    "quick": ["stock_200", "dpm2m_32", "dpm2m_16"],
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batch6() -> Path:
    from rdkit import Chem

    base = yaml.safe_load((REPO_ROOT / "data/smoke.yaml").read_text())
    batch_dir = REPO_ROOT / "data/batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for name, smiles in BATCH6_LIGANDS.items():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, name
        payload = json.loads(json.dumps(base))
        payload["sequences"][1]["ligand"]["smiles"] = smiles
        (batch_dir / f"{name}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    return batch_dir


def resolve_inputs(input_path: Path):
    if input_path.is_dir():
        yamls = sorted(p for p in input_path.iterdir() if p.suffix in (".yaml", ".yml"))
        assert yamls, f"No YAML files found in {input_path}"
        return input_path, yamls, input_path.name
    assert input_path.is_file(), f"Input not found: {input_path}"
    return input_path, [input_path], input_path.stem


def geometric_schedule(diffusion, steps: int, power: float = 1.0):
    u = torch.linspace(0, 1, steps, device=diffusion.device)
    log_sigma = math.log(diffusion.sigma_max * diffusion.sigma_data) + u.pow(power) * math.log(
        diffusion.sigma_min / diffusion.sigma_max
    )
    return torch.cat((log_sigma.exp(), torch.zeros(1, device=diffusion.device)))


def make_ode_sampler(diffusion, original_schedule, denoise_fn, solver: str, schedule_name: str,
                     dpm_form: str = "midpoint", history_weight: float = 1.0,
                     tail_sigma: float = 0.0, grid_power: float = 1.0,
                     ligand_history_weight: float = 1.0):
    @torch.no_grad()
    def ode_sample(atom_mask, num_sampling_steps=None, multiplicity=1,
                   max_parallel_samples=None, steering_args=None, **conditions):
        if steering_args is not None:
            assert not any(
                steering_args.get(key, False)
                for key in ("fk_steering", "physical_guidance_update", "contact_guidance_update")
            )
        steps = num_sampling_steps or diffusion.num_sampling_steps
        sigmas = (
            geometric_schedule(diffusion, steps, grid_power)
            if schedule_name == "geometric"
            else original_schedule(steps)
        ).tolist()
        x = sigmas[0] * torch.randn(
            (atom_mask.shape[0] * multiplicity, atom_mask.shape[1], 3),
            device=diffusion.device,
        )
        previous_denoised = None
        previous_h = None
        max_parallel = max_parallel_samples or multiplicity
        chunks = torch.arange(x.shape[0], device=x.device).chunk(multiplicity % max_parallel + 1)
        token_ids = conditions["feats"]["atom_to_token"][0].argmax(-1)
        ligand_atoms = conditions["feats"]["mol_type"][0][token_ids] == 3
        history_weights = torch.ones((1, x.shape[1], 1), device=x.device)
        history_weights[:, ligand_atoms] = ligand_history_weight

        def evaluate(coords, sigma):
            output = torch.empty_like(coords)
            for ids in chunks:
                output[ids] = denoise_fn(
                    coords[ids], float(sigma), dict(multiplicity=ids.numel(), **conditions)
                )
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
                    if previous_denoised is not None and sigma > tail_sigma:
                        derivative = (h / previous_h) * (clean - previous_denoised)
                        coefficient = -0.5 * phi if dpm_form == "midpoint" else phi / h + 1
                        x = first_order + history_weight * coefficient * history_weights * derivative
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

    return ode_sample


def collect_outputs(out_dir: Path, results_stem: str, ligand_names: list[str]):
    scores = {}
    cif_paths = {}
    for name in ligand_names:
        prediction = out_dir / f"boltz_results_{results_stem}" / "predictions" / name
        cif = prediction / f"{name}_model_0.cif"
        aff = prediction / f"affinity_{name}.json"
        assert cif.is_file(), f"Missing structure output: {cif}"
        assert aff.is_file(), f"Missing affinity output: {aff}"
        value = json.loads(aff.read_text())
        for key in ("affinity_pred_value", "affinity_probability_binary"):
            assert math.isfinite(value[key]), (name, key, value[key])
        assert 0.0 <= value["affinity_probability_binary"] <= 1.0, (name, value)
        scores[name] = value
        cif_paths[name] = str(cif)
    return scores, cif_paths


def compare_cifs(cif_a: str, cif_b: str) -> dict:
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    xyz_a = np.array([a.coord for a in parser.get_structure("a", cif_a).get_atoms()], dtype=float)
    xyz_b = np.array([a.coord for a in parser.get_structure("b", cif_b).get_atoms()], dtype=float)
    assert xyz_a.shape == xyz_b.shape, (xyz_a.shape, xyz_b.shape)
    diff = np.abs(xyz_a - xyz_b)
    return {
        "atoms": int(xyz_a.shape[0]),
        "max_abs_coord_error_A": float(diff.max()),
        "rms_coord_error_A": float(np.sqrt(np.mean((xyz_a - xyz_b) ** 2))),
    }


class ResidentBoltzRunner:
    """Keeps Boltz-2 structure & affinity models resident in memory across runs."""

    def __init__(self, offload_inactive_to_cpu: bool = False, reseed_per_stage: bool = True,
                 num_workers: int = 2, preprocessing_threads: int = 2):
        self.offload_inactive_to_cpu = offload_inactive_to_cpu
        self.reseed_per_stage = reseed_per_stage
        self.num_workers = num_workers
        self.preprocessing_threads = preprocessing_threads
        self.cache = {}
        self.profile = None
        self.cold_loads = []
        self._active_run = None
        self._orig_process = main.process_inputs
        self._orig_load = main.Boltz2.load_from_checkpoint
        self._orig_predict = main.Trainer.predict
        self._install_hooks()

    def _install_hooks(self):
        runner = self

        @functools.wraps(runner._orig_process)
        def process_inputs(*a, **kw):
            start = time.perf_counter()
            result = runner._orig_process(*a, **kw)
            if runner._active_run is not None:
                runner._active_run["preprocessing_seconds"] += time.perf_counter() - start
            return result

        def load_from_checkpoint(*a, **kw):
            ckpt_key = str(Path(a[0]).resolve())
            run = runner._active_run
            bypass = run is not None and run.get("bypass_cache", False)
            if (not bypass) and (ckpt_key in runner.cache):
                start = time.perf_counter()
                entry = runner.cache[ckpt_key]
                model = entry["model"]
                model._trainer = None
                if "predict_args" in kw and kw["predict_args"] is not None:
                    model.predict_args = dict(kw["predict_args"])
                else:
                    model.predict_args = dict(entry["default_predict_args"])
                if "steering_args" in kw and kw["steering_args"] is not None:
                    model.steering_args = dict(kw["steering_args"])
                else:
                    model.steering_args = dict(entry["default_steering_args"])
                if runner.offload_inactive_to_cpu and torch.cuda.is_available():
                    model.cuda()
                model.eval()
                elapsed = time.perf_counter() - start
                if run is not None:
                    run["checkpoint_loads"].append({
                        "checkpoint": Path(ckpt_key).name,
                        "cache_hit": True,
                        "seconds": elapsed,
                    })
                return model

            start = time.perf_counter()
            model = runner._orig_load(*a, **kw)
            model.eval()
            elapsed = time.perf_counter() - start
            if not bypass:
                runner.cache[ckpt_key] = {
                    "model": model,
                    "default_predict_args": dict(getattr(model, "predict_args", {})),
                    "default_steering_args": dict(getattr(model, "steering_args", {})),
                    "cold_load_seconds": elapsed,
                }
                runner.cold_loads.append({
                    "checkpoint": Path(ckpt_key).name,
                    "seconds": elapsed,
                })
            if run is not None:
                run["checkpoint_loads"].append({
                    "checkpoint": Path(ckpt_key).name,
                    "cache_hit": False,
                    "seconds": elapsed,
                })
            return model

        @functools.wraps(runner._orig_predict)
        def predict(trainer, model, *a, **kw):
            run = runner._active_run
            if run is None:
                return runner._orig_predict(trainer, model, *a, **kw)

            is_affinity = bool(getattr(model, "affinity_prediction", False))
            stage_name = "affinity" if is_affinity else "structure"
            cfg = run["config"]
            prefix = "affinity_" if is_affinity else "structure_"
            method = cfg[prefix + "method"]
            schedule = cfg[prefix + "schedule"]
            steps = int(cfg[prefix + "steps"])
            recycling = int(cfg[prefix + "recycling"])
            samples = int(cfg[prefix + "samples"])

            diffusion = model.structure_module
            orig_forward = model.forward
            orig_sample = diffusion.sample
            profile = runner.profile
            orig_schedule = diffusion.sample_schedule
            orig_denoise = diffusion.preconditioned_network_forward
            orig_predict_args = dict(getattr(model, "predict_args", {}))
            orig_steering_args = dict(getattr(model, "steering_args", {}))
            orig_use_kernels = bool(getattr(model, "use_kernels", False))
            if cfg.get(prefix + "use_kernels") is not None:
                model.use_kernels = bool(cfg[prefix + "use_kernels"])

            restore_ops = None
            if cfg.get("ops_opt", False):
                from boltz_ops_opt import install_ops_opt
                restore_ops = install_ops_opt(model)

            restore_trunk = None
            trunk_stats = None
            if is_affinity and cfg.get("affinity_teacache"):
                from boltz_trunk_opt import install_affinity_teacache
                restore_trunk, trunk_stats = install_affinity_teacache(model, **cfg["affinity_teacache"])
            forward_fn = model.forward

            # 支持独立覆盖两阶段 recycling / steps / samples（包括原本硬编码的 affinity recycling=5）。
            model.predict_args = dict(orig_predict_args)
            model.predict_args["recycling_steps"] = recycling
            model.predict_args["sampling_steps"] = steps
            model.predict_args["diffusion_samples"] = samples

            if method in ("euler", "heun", "dpm2m"):
                model.steering_args = dict(orig_steering_args)
                for key in ("fk_steering", "physical_guidance_update", "contact_guidance_update"):
                    if key in model.steering_args:
                        model.steering_args[key] = False

            stage_info = {
                "stage": stage_name,
                "method": method,
                "schedule": schedule,
                "steps": steps,
                "recycling_steps": recycling,
                "diffusion_samples": samples,
                "use_kernels": bool(getattr(model, "use_kernels", False)),
                "ops_opt": bool(cfg.get("ops_opt", False)),
                "affinity_teacache": cfg.get("affinity_teacache"),
                "denoiser_calls": 0,
                "sample_equivalent_nfe": 0,
                "model_forward_seconds_per_item": [],
                "diffusion_sample_seconds_per_item": [],
            }

            if method == "geometric":
                diffusion.sample_schedule = lambda num_sampling_steps=None: geometric_schedule(
                    diffusion, num_sampling_steps or diffusion.num_sampling_steps
                )

            def denoise(coords, sigma, network_condition_kwargs):
                stage_info["denoiser_calls"] += 1
                stage_info["sample_equivalent_nfe"] += int(coords.shape[0])
                if profile is not None:
                    return profile.measure("diffusion_sample.denoiser", orig_denoise,
                                           coords, sigma, network_condition_kwargs)
                return orig_denoise(coords, sigma, network_condition_kwargs)

            diffusion.preconditioned_network_forward = denoise

            if method in ("euler", "heun", "dpm2m"):
                chosen_sample = make_ode_sampler(
                    diffusion, orig_schedule, denoise,
                    solver=method,
                    schedule_name=schedule,
                    dpm_form=cfg.get("dpm_form", "midpoint"),
                    history_weight=float(cfg.get("history_weight", 1.0)),
                    tail_sigma=float(cfg.get("tail_sigma", 0.0)),
                    grid_power=float(cfg.get("grid_power", 1.0)),
                    ligand_history_weight=float(cfg.get("ligand_history_weight", 1.0)),
                )
            else:
                chosen_sample = orig_sample

            @functools.wraps(orig_sample)
            def timed_sample(*x, **y):
                if profile is not None:
                    return profile.measure("diffusion_sample", chosen_sample, *x, **y)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = chosen_sample(*x, **y)
                torch.cuda.synchronize()
                stage_info["diffusion_sample_seconds_per_item"].append(time.perf_counter() - t0)
                assert torch.isfinite(out["sample_atom_coords"]).all()
                return out

            @functools.wraps(orig_forward)
            def timed_forward(*x, **y):
                if profile is not None:
                    out, elapsed, sample_elapsed = profile.forward(forward_fn, *x, **y)
                    stage_info["model_forward_seconds_per_item"].append(elapsed)
                    stage_info["diffusion_sample_seconds_per_item"].append(sample_elapsed)
                    assert torch.isfinite(out["sample_atom_coords"]).all()
                    return out
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = forward_fn(*x, **y)
                torch.cuda.synchronize()
                stage_info["model_forward_seconds_per_item"].append(time.perf_counter() - t0)
                return out

            diffusion.sample = timed_sample
            model.forward = timed_forward

            if runner.reseed_per_stage:
                seed_all(int(run["seed"]))

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                if profile is None:
                    result = runner._orig_predict(trainer, model, *a, **kw)
                else:
                    with profile.stage(model, stage_name):
                        result = runner._orig_predict(trainer, model, *a, **kw)
            finally:
                if restore_trunk is not None:
                    restore_trunk()
                if restore_ops is not None:
                    restore_ops()
                model.forward = orig_forward
                diffusion.sample = orig_sample
                diffusion.sample_schedule = orig_schedule
                diffusion.preconditioned_network_forward = orig_denoise
                model.predict_args = orig_predict_args
                model.steering_args = orig_steering_args
                model.use_kernels = orig_use_kernels
            torch.cuda.synchronize()
            stage_info["trainer_wall_seconds"] = time.perf_counter() - t0
            stage_info["model_forward_seconds"] = float(sum(stage_info["model_forward_seconds_per_item"]))
            stage_info["diffusion_sample_seconds"] = float(sum(stage_info["diffusion_sample_seconds_per_item"]))
            stage_info["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            stage_info["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if trunk_stats is not None:
                if trunk_stats["msa_recycles"]:
                    stage_info["mean_msa_recycles"] = float(np.mean(trunk_stats["msa_recycles"]))
                if trunk_stats["pf_recycles"]:
                    stage_info["mean_pf_recycles"] = float(np.mean(trunk_stats["pf_recycles"]))
            run["stages"][stage_name] = stage_info

            if runner.offload_inactive_to_cpu and not run.get("bypass_cache", False):
                model.cpu()
                torch.cuda.empty_cache()
            return result

        main.process_inputs = process_inputs
        main.Boltz2.load_from_checkpoint = staticmethod(load_from_checkpoint)
        main.Trainer.predict = predict

    def run_once(self, label: str, config: dict, input_path: Path, out_dir: Path,
                 seed: int, repeat: int, bypass_cache: bool = False, tag: str = "benchmark") -> dict:
        target_path, yamls, results_stem = resolve_inputs(input_path)
        ligand_names = [p.stem for p in yamls]
        out_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if self.offload_inactive_to_cpu:
                torch.cuda.empty_cache()

        self._active_run = {
            "label": label,
            "tag": tag,
            "repeat": repeat,
            "seed": seed,
            "config": dict(config),
            "bypass_cache": bypass_cache,
            "preprocessing_seconds": 0.0,
            "checkpoint_loads": [],
            "stages": {},
        }

        cli_args = [
            "predict", str(target_path),
            "--model", "boltz2",
            "--seed", str(seed),
            "--recycling_steps", str(config["structure_recycling"]),
            "--sampling_steps", str(config["structure_steps"]),
            "--diffusion_samples", str(config["structure_samples"]),
            "--sampling_steps_affinity", str(config["affinity_steps"]),
            "--diffusion_samples_affinity", str(config["affinity_samples"]),
            "--num_workers", str(self.num_workers),
            "--preprocessing-threads", str(self.preprocessing_threads),
            "--out_dir", str(out_dir),
            "--override",
        ]

        t0 = time.perf_counter()
        try:
            main.cli.main(args=cli_args, standalone_mode=False)
        finally:
            run_data = self._active_run
            self._active_run = None
        wall_seconds = time.perf_counter() - t0

        scores, cif_paths = collect_outputs(out_dir, results_stem, ligand_names)
        load_seconds = float(sum(item["seconds"] for item in run_data["checkpoint_loads"]))
        all_cache_hits = bool(run_data["checkpoint_loads"]) and all(
            item["cache_hit"] for item in run_data["checkpoint_loads"]
        )
        steady_seconds = wall_seconds - load_seconds
        n_mols = len(ligand_names)
        struct_diff = run_data["stages"]["structure"]["diffusion_sample_seconds"]
        aff_diff = run_data["stages"]["affinity"]["diffusion_sample_seconds"]

        record = {
            "label": label,
            "tag": tag,
            "repeat": repeat,
            "seed": seed,
            "input": str(target_path),
            "results_stem": results_stem,
            "out_dir": str(out_dir),
            "molecules": n_mols,
            "ligand_names": ligand_names,
            "config": config,
            "all_cache_hits": all_cache_hits,
            "checkpoint_loads": run_data["checkpoint_loads"],
            "checkpoint_load_seconds": load_seconds,
            "preprocessing_seconds": run_data["preprocessing_seconds"],
            "wall_seconds": wall_seconds,
            "steady_state_wall_seconds": steady_seconds,
            "gpu_seconds_per_molecule": steady_seconds / n_mols,
            "molecules_per_second": n_mols / steady_seconds,
            "total_diffusion_seconds": struct_diff + aff_diff,
            "diffusion_share": (struct_diff + aff_diff) / steady_seconds,
            "stages": run_data["stages"],
            "scores": scores,
            "cif_paths": cif_paths,
        }
        return record


def main_cli():
    parser = argparse.ArgumentParser(
        description="Persistent in-process Boltz-2 loader for fair steady-state multi-method comparison."
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="YAML file or directory of YAML files. Default: auto-generate 6-ligand data/batch.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/resident_compare",
                        help="Output directory for JSONL logs, summaries, and predictions.")
    parser.add_argument("--suite", choices=list(SUITES.keys()), default="fair",
                        help="Preset comparison suite (ignored if --configs is set).")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated preset config names (e.g. stock_200,dpm2m_32,dpm2m_16).")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Number of interleaved repeats per config (alternating forward/reverse order).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preprocessing-threads", type=int, default=2)
    parser.add_argument("--offload-inactive-to-cpu", action="store_true",
                        help="Move inactive checkpoint to CPU between stages to reduce VRAM.")
    parser.add_argument("--no-verify-cache", action="store_true",
                        help="Skip cold-vs-cached exactness verification (still runs 1 warmup molecule).")
    parser.add_argument("--no-reseed-per-stage", action="store_true",
                        help="Disable deterministic RNG reset at the start of each stage.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    from importlib.metadata import version

    assert version("boltz") == "2.2.1", f"Expected boltz==2.2.1, got {version('boltz')}"

    input_path = prepare_batch6() if args.input is None else args.input
    _, yamls, _ = resolve_inputs(input_path)
    config_names = (
        [c.strip() for c in args.configs.split(",") if c.strip()]
        if args.configs
        else list(SUITES[args.suite])
    )
    for name in config_names:
        assert name in PRESET_CONFIGS, f"Unknown config: {name}. Available: {sorted(PRESET_CONFIGS)}"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    runner = ResidentBoltzRunner(
        offload_inactive_to_cpu=args.offload_inactive_to_cpu,
        reseed_per_stage=not args.no_reseed_per_stage,
        num_workers=args.num_workers,
        preprocessing_threads=args.preprocessing_threads,
    )

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "boltz": version("boltz"),
        "input": str(input_path),
        "molecules": len(yamls),
        "ligand_names": [p.stem for p in yamls],
        "suite": args.suite if not args.configs else "custom",
        "configs": config_names,
        "repeats": args.repeats,
        "seed": args.seed,
        "offload_inactive_to_cpu": args.offload_inactive_to_cpu,
        "reseed_per_stage": not args.no_reseed_per_stage,
    }

    # 阶段 0：验证持久化缓存零漂移 + 完成冷加载与 CUDA/JIT 预热（不计入稳态对比）
    warmup_yaml = yamls[0]
    verification = {}
    if not args.no_verify_cache:
        cold_rec = runner.run_once(
            label="verify_cold_stock_200",
            config=PRESET_CONFIGS["stock_200"],
            input_path=warmup_yaml,
            out_dir=out_dir / "_verify_cold",
            seed=args.seed,
            repeat=0,
            bypass_cache=False,
            tag="verify_cold",
        )
        warm_rec = runner.run_once(
            label="verify_warm_stock_200",
            config=PRESET_CONFIGS["stock_200"],
            input_path=warmup_yaml,
            out_dir=out_dir / "_verify_warm",
            seed=args.seed,
            repeat=0,
            bypass_cache=False,
            tag="verify_warm",
        )
        lig = warmup_yaml.stem
        cif_diff = compare_cifs(cold_rec["cif_paths"][lig], warm_rec["cif_paths"][lig])
        score_diffs = {
            k: abs(cold_rec["scores"][lig][k] - warm_rec["scores"][lig][k])
            for k in cold_rec["scores"][lig]
        }
        verification = {
            "ligand": lig,
            "cold_wall_seconds": cold_rec["wall_seconds"],
            "cold_checkpoint_load_seconds": cold_rec["checkpoint_load_seconds"],
            "warm_wall_seconds": warm_rec["wall_seconds"],
            "warm_checkpoint_load_seconds": warm_rec["checkpoint_load_seconds"],
            "warm_all_cache_hits": warm_rec["all_cache_hits"],
            "cif_diff": cif_diff,
            "score_abs_diffs": score_diffs,
            "max_abs_score_diff": max(score_diffs.values()),
            "exact_match": cif_diff["max_abs_coord_error_A"] == 0.0 and max(score_diffs.values()) == 0.0,
        }
        print("[VERIFY_CACHE]", json.dumps(verification), flush=True)
    else:
        warm_rec = runner.run_once(
            label="warmup_stock_200",
            config=PRESET_CONFIGS["stock_200"],
            input_path=warmup_yaml,
            out_dir=out_dir / "_warmup",
            seed=args.seed,
            repeat=0,
            bypass_cache=False,
            tag="warmup",
        )
        verification = {
            "skipped_exact_diff": True,
            "cold_checkpoint_load_seconds": warm_rec["checkpoint_load_seconds"],
            "warmup_wall_seconds": warm_rec["wall_seconds"],
        }
        print("[WARMUP]", json.dumps(verification), flush=True)
        if any(PRESET_CONFIGS[name].get("affinity_use_kernels") for name in config_names):
            affkern_warm = runner.run_once(
                label="warmup_affkern",
                config=PRESET_CONFIGS["dpm2m_16_affkern"],
                input_path=warmup_yaml,
                out_dir=out_dir / "_warmup_affkern",
                seed=args.seed,
                repeat=0,
                bypass_cache=False,
                tag="warmup",
            )
            print("[WARMUP_AFFKERN]", json.dumps({"wall_seconds": affkern_warm["wall_seconds"]}), flush=True)
        if any(PRESET_CONFIGS[name].get("ops_opt") for name in config_names):
            ops_warm = runner.run_once(
                label="warmup_ops",
                config=PRESET_CONFIGS["dpm2m_16_affkern_ops"],
                input_path=warmup_yaml,
                out_dir=out_dir / "_warmup_ops",
                seed=args.seed,
                repeat=0,
                bypass_cache=False,
                tag="warmup",
            )
            print("[WARMUP_OPS]", json.dumps({"wall_seconds": ops_warm["wall_seconds"]}), flush=True)

    # 阶段 1：交错顺序（偶数轮正序、奇数轮倒序）稳态公平对比
    records = []
    for repeat in range(args.repeats):
        order = config_names if repeat % 2 == 0 else list(reversed(config_names))
        for name in order:
            run_out = out_dir / f"{name}_r{repeat}"
            rec = runner.run_once(
                label=name,
                config=PRESET_CONFIGS[name],
                input_path=input_path,
                out_dir=run_out,
                seed=args.seed,
                repeat=repeat,
                bypass_cache=False,
                tag="benchmark",
            )
            assert rec["all_cache_hits"], f"Expected cache hit during steady-state run: {name}_r{repeat}"
            records.append(rec)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"[BENCH] config={name} repeat={repeat} "
                f"steady={rec['steady_state_wall_seconds']:.2f}s "
                f"({rec['gpu_seconds_per_molecule']:.2f}s/mol, {rec['molecules_per_second']:.3f} mol/s) "
                f"diff={rec['total_diffusion_seconds']:.2f}s ({100*rec['diffusion_share']:.1f}%)",
                flush=True,
            )

    summary = {
        "metadata": metadata,
        "cold_loads": runner.cold_loads,
        "total_cold_load_seconds": float(sum(x["seconds"] for x in runner.cold_loads)),
        "cache_verification": verification,
        "runs": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] Wrote", out_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main_cli()
