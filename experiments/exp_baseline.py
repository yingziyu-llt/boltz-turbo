import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
os.chdir(root)
result_dir = root / "results" / ("profile" if args.profile else "throughput")
result_dir.mkdir(parents=True, exist_ok=True)

# 同一蛋白的芳香族氨基酸系列，仅用于性能测量，不赋予活性标签。
ligands = {
    "tyrosine": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "dopa": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
    "fluorophenylalanine": "N[C@@H](Cc1ccc(F)cc1)C(=O)O",
    "methylphenylalanine": "N[C@@H](Cc1ccc(C)cc1)C(=O)O",
    "tryptophan": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
}
import yaml
from rdkit import Chem

base = yaml.safe_load(Path("data/smoke.yaml").read_text())
inputs = root / "data" / "batch"
inputs.mkdir(parents=True, exist_ok=True)
for name, smiles in ligands.items():
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, name
    base["sequences"][1]["ligand"]["smiles"] = smiles
    (inputs / f"{name}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))


def command(input_path, out_dir):
    return [
        str(input_path), "--model", "boltz2", "--seed", "0",
        "--recycling_steps", "3", "--sampling_steps", "200",
        "--diffusion_samples", "1", "--sampling_steps_affinity", "200",
        "--diffusion_samples_affinity", "5", "--num_workers", "2",
        "--preprocessing-threads", "2", "--out_dir", str(out_dir), "--override",
    ]


def outputs(out_dir, names):
    scores = {}
    for name in names:
        prediction = out_dir / f"boltz_results_{inputs.name}" / "predictions" / name
        if args.profile:
            prediction = out_dir / "boltz_results_tyrosine" / "predictions" / name
        assert (prediction / f"{name}_model_0.cif").is_file(), prediction
        value = json.loads((prediction / f"affinity_{name}.json").read_text())
        assert all(math.isfinite(value[k]) for k in ("affinity_pred_value", "affinity_probability_binary"))
        assert 0 <= value["affinity_probability_binary"] <= 1
        scores[name] = value
    return scores


if not args.profile:
    rows = []
    # 两次独立 CLI：第二次仅复用磁盘/JIT缓存，不宣称模型跨进程常驻。
    for repetition in range(2):
        out_dir = result_dir / f"run_{repetition}"
        cmd = [sys.executable, "-m", "boltz.main", "predict", *command(inputs, out_dir)]
        start = time.perf_counter()
        with (result_dir / f"run_{repetition}.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start
        row = {"repetition": repetition, "molecules": len(ligands), "wall_seconds": elapsed,
               "molecules_per_second": len(ligands) / elapsed,
               "gpu_seconds_per_molecule": elapsed / len(ligands),
               "scores": outputs(out_dir, ligands)}
        rows.append(row)
        (result_dir / "summary.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
else:
    import functools
    from importlib.metadata import version
    import torch
    import boltz.main as main

    assert version("boltz") == "2.2.1"
    report = {"environment": {"python": sys.version, "torch": torch.__version__,
              "boltz": version("boltz"), "gpu": torch.cuda.get_device_name(0)},
              "preprocessing_seconds": [], "checkpoint_load_seconds": [], "stages": []}
    original_process = main.process_inputs

    @functools.wraps(original_process)
    def process_inputs(*a, **kw):
        start = time.perf_counter()
        result = original_process(*a, **kw)
        report["preprocessing_seconds"].append(time.perf_counter() - start)
        return result

    main.process_inputs = process_inputs
    original_load = main.Boltz2.load_from_checkpoint

    def load_from_checkpoint(*a, **kw):
        start = time.perf_counter()
        result = original_load(*a, **kw)
        report["checkpoint_load_seconds"].append({"checkpoint": str(a[0]),
            "seconds": time.perf_counter() - start})
        return result

    main.Boltz2.load_from_checkpoint = staticmethod(load_from_checkpoint)
    original_predict = main.Trainer.predict

    @functools.wraps(original_predict)
    def predict(trainer, model, *a, **kw):
        stage = "affinity" if model.affinity_prediction else "structure"
        events = []
        restores = []
        denoiser_calls = 0

        def wrap(obj, attr, label):
            original = getattr(obj, attr)

            @functools.wraps(original)
            def measured(*x, **y):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter()
                with torch.profiler.record_function(label):
                    value = original(*x, **y)
                end.record()
                events.append((label, begin, end, time.perf_counter() - start))
                return value

            setattr(obj, attr, measured)
            restores.append((obj, attr, original))

        wrap(model, "forward", "model_forward")
        for name in ("input_embedder", "template_module", "msa_module", "pairformer_module",
                     "diffusion_conditioning", "confidence_module", "affinity_module",
                     "affinity_module1", "affinity_module2"):
            if hasattr(model, name):
                wrap(getattr(model, name), "forward", name)
        wrap(model.structure_module, "sample", "diffusion_sample")
        original_denoise = model.structure_module.preconditioned_network_forward

        @functools.wraps(original_denoise)
        def denoise(*x, **y):
            nonlocal denoiser_calls
            denoiser_calls += 1
            # 只记录已过前两次调用后的一个真实 denoiser，避免全轨迹 trace 的扰动与体积。
            if denoiser_calls != 3:
                return original_denoise(*x, **y)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                value = original_denoise(*x, **y)
            averages = prof.key_averages()
            (result_dir / f"{stage}_denoiser_ops.txt").write_text(
                averages.table(sort_by="self_cuda_time_total", row_limit=40))
            ops = [{"name": event.key, "calls": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "self_device_us": event.self_device_time_total} for event in averages]
            (result_dir / f"{stage}_denoiser_ops.json").write_text(json.dumps(ops, indent=2))
            return value

        model.structure_module.preconditioned_network_forward = denoise
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original_predict(trainer, model, *a, **kw)
        torch.cuda.synchronize()
        stage_row = {"stage": stage, "trainer_wall_seconds": time.perf_counter() - start,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                     "denoiser_calls": denoiser_calls, "modules": {}}
        for label, begin, end, cpu in events:
            row = stage_row["modules"].setdefault(label, {"calls": 0, "gpu_timeline_seconds": 0,
                                                          "host_call_seconds": 0})
            row["calls"] += 1
            row["gpu_timeline_seconds"] += begin.elapsed_time(end) / 1000
            row["host_call_seconds"] += cpu
        report["stages"].append(stage_row)
        (result_dir / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(stage_row), flush=True)
        for obj, attr, original in restores:
            setattr(obj, attr, original)
        model.structure_module.preconditioned_network_forward = original_denoise
        return result

    main.Trainer.predict = predict
    out_dir = result_dir / "prediction"
    start = time.perf_counter()
    main.cli.main(args=["predict", *command(inputs / "tyrosine.yaml", out_dir)], standalone_mode=False)
    report["instrumented_cli_seconds"] = time.perf_counter() - start
    report["scores"] = outputs(out_dir, ["tyrosine"])
    report["timing_note"] = "Instrumented GPU timeline includes host gaps and one profiled denoiser per stage; nested model_forward is inclusive, not additive. Throughput is measured separately."
    (result_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
