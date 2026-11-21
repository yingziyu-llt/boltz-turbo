import argparse
import functools
import json
import math
import time
from pathlib import Path

import torch
import boltz.main as main

parser = argparse.ArgumentParser()
parser.add_argument("--graph", action="store_true")
args = parser.parse_args()
mode = "graph" if args.graph else "eager"
out_dir = Path("results/resident") / mode
out_dir.mkdir(parents=True, exist_ok=True)
report = {"mode": mode, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "stages": []}
original_predict = main.Trainer.predict


@functools.wraps(original_predict)
def predict(trainer, model, *a, **kw):
    stage = {"stage": "affinity" if model.affinity_prediction else "structure"}
    diffusion = model.structure_module
    original_denoise = diffusion.preconditioned_network_forward
    original_sample = diffusion.sample
    graph = None
    calls = 0

    def denoise(coords, sigma, network_condition_kwargs):
        nonlocal graph, calls
        calls += 1
        if calls < 3:
            return original_denoise(coords, sigma, network_condition_kwargs)
        if graph is None:
            static_coords = coords.clone()
            static_sigma = torch.full((coords.shape[0],), sigma, device=coords.device)
            stage["coordinate_shape"] = list(coords.shape)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    original_denoise(static_coords, static_sigma, network_condition_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            reference = original_denoise(static_coords, static_sigma, network_condition_kwargs).clone()
            graph = torch.cuda.CUDAGraph()
            start = time.perf_counter()
            with torch.cuda.graph(graph):
                static_output = original_denoise(static_coords, static_sigma, network_condition_kwargs)
            torch.cuda.synchronize()
            stage["capture_seconds"] = time.perf_counter() - start
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(static_output, reference, rtol=1e-5, atol=1e-5)
            stage["fixed_input_max_abs_error"] = (static_output - reference).abs().max().item()
            # 改变坐标与时间，验证重放确实消费更新后的输入。
            static_coords.add_(0.01)
            static_sigma.mul_(0.9)
            changed = original_denoise(static_coords, static_sigma, network_condition_kwargs).clone()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(static_output, changed, rtol=1e-5, atol=1e-5)
            stage["changed_input_max_abs_error"] = (static_output - changed).abs().max().item()
            stage["microbenchmark"] = {}
            for label in ("eager", "graph"):
                times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(20):
                        if label == "eager":
                            original_denoise(static_coords, static_sigma, network_condition_kwargs)
                        else:
                            graph.replay()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) / 20)
                stage["microbenchmark"][label + "_seconds_per_call"] = times
            denoise.buffers = (static_coords, static_sigma, static_output)
            # 条件张量在本次 sample 内固定；下一阶段使用独立捕获。
            denoise.conditions = network_condition_kwargs
        static_coords, static_sigma, static_output = denoise.buffers
        static_coords.copy_(coords)
        static_sigma.fill_(sigma)
        graph.replay()
        # sampler 消费该结果后才发起下一步，不保留跨重放输出。
        return static_output

    @functools.wraps(original_sample)
    def sample(*x, **y):
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = original_sample(*x, **y)
        torch.cuda.synchronize()
        stage["sample_wall_seconds"] = time.perf_counter() - start
        after_cpu = torch.get_rng_state()
        after_cuda = torch.cuda.get_rng_state()
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        torch.cuda.synchronize()
        start = time.perf_counter()
        original_sample(*x, **y)
        torch.cuda.synchronize()
        stage["warm_sample_wall_seconds"] = time.perf_counter() - start
        torch.set_rng_state(after_cpu)
        torch.cuda.set_rng_state(after_cuda)
        return value

    diffusion.sample = sample
    if args.graph:
        diffusion.preconditioned_network_forward = denoise
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = original_predict(trainer, model, *a, **kw)
    torch.cuda.synchronize()
    stage["trainer_wall_seconds"] = time.perf_counter() - start
    stage["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    stage["denoiser_calls"] = calls if args.graph else None
    report["stages"].append(stage)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(stage), flush=True)
    diffusion.sample = original_sample
    diffusion.preconditioned_network_forward = original_denoise
    return value


main.Trainer.predict = predict
start = time.perf_counter()
main.cli.main(args=["predict", "data/smoke.yaml", "--model", "boltz2", "--seed", "0",
    "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
    "--sampling_steps_affinity", "200", "--diffusion_samples_affinity", "5",
    "--num_workers", "2", "--preprocessing-threads", "2", "--out_dir", str(out_dir),
    "--override"], standalone_mode=False)
report["cli_seconds"] = time.perf_counter() - start
prediction = out_dir / "boltz_results_smoke/predictions/smoke"
assert (prediction / "smoke_model_0.cif").is_file()
scores = json.loads((prediction / "affinity_smoke.json").read_text())
assert all(math.isfinite(v) for v in scores.values())
assert 0 <= scores["affinity_probability_binary"] <= 1
report["scores"] = scores
if args.graph:
    baseline = json.loads(Path("results/resident/eager/summary.json").read_text())
    report["score_abs_errors"] = {key: abs(value - baseline["scores"][key]) for key, value in scores.items()}
    # 完整采样后的结构与最终 affinity 同时检查。
    import numpy as np
    from Bio.PDB import MMCIFParser
    cif_parser = MMCIFParser(QUIET=True)
    before = cif_parser.get_structure("eager", "results/resident/eager/boltz_results_smoke/predictions/smoke/smoke_model_0.cif")
    after = cif_parser.get_structure("graph", str(prediction / "smoke_model_0.cif"))
    xyz_before = np.array([atom.coord for atom in before.get_atoms()])
    xyz_after = np.array([atom.coord for atom in after.get_atoms()])
    assert xyz_before.shape == xyz_after.shape
    report["structure_coordinate_max_abs_error_angstrom"] = float(np.abs(xyz_before - xyz_after).max())
    report["structure_coordinate_rms_error_angstrom"] = float(np.sqrt(np.mean((xyz_before - xyz_after)**2)))
(out_dir / "summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
