import argparse
import functools
import json
import re
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

import torch

from exp_resident_compare import PRESET_CONFIGS, ResidentBoltzRunner, main

CONFIGS = {
    name: PRESET_CONFIGS[name] for name in ("dpm2m_16", "dpm2m_16_affrec1")
}
CONFIGS["dpm2m_16_affrec1_affs1"] = dict(CONFIGS["dpm2m_16_affrec1"], affinity_samples=1)
STAGES = ("structure", "affinity")
SCOPES = (
    "input_embedder", "input_embedder.affinity", "template_module", "msa_module",
    "pairformer_module.first", "pairformer_module.recycle", "distogram_module",
    "diffusion_conditioning", "diffusion_sample.sampler_overhead", "diffusion_sample.denoiser",
    "confidence_module.overhead", "confidence_module.sample",
    *(f"{mod}.{part}" for mod in ("affinity_module1", "affinity_module2")
      for part in ("conditioning", "trunk", "head")),
    "forward.other",
)
PREFIX = "hsp90::"


def activation_dtypes(value):
    if isinstance(value, torch.Tensor):
        return {str(value.dtype)}
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (tuple, list)):
        return set()
    return set().union(*(activation_dtypes(v) for v in value))


def stats(values):
    return {"mean": mean(values), "median": median(values)}


def cuda_activity(trace_path):
    # Kineto emits a header and timestamp row per Chrome event. Stream the trace
    # instead of materializing hundreds of MB of events for each ligand.
    header = re.compile(r'"ph": "X", "cat": "([^"]+)", "name": "([^"]*)"')
    times = re.compile(r'"ts": ([\d.]+), "dur": ([\d.]+)')
    intervals = []
    window = None
    with trace_path.open() as trace:
        for line in trace:
            match = header.search(line)
            if match is None:
                continue
            category, name = match.groups()
            root = category == "user_annotation" and name == PREFIX + "forward"
            if root or category in {"kernel", "gpu_memcpy", "gpu_memset"}:
                start, duration = map(float, times.search(next(trace)).groups())
                if root:
                    assert window is None
                    window = start, duration
                else:
                    intervals.append((start, start + duration))
    start, duration = window
    end = start + duration
    active_us = 0.0
    last = start
    for left, right in sorted(intervals):
        right = min(right, end)
        active_us += max(0, right - max(left, last))
        last = max(last, right)
    assert 0 < active_us <= duration + 0.01
    return dict(forward_trace_ms=duration / 1e3, cuda_active_ms=active_us / 1e3,
                cuda_inactive_ms=(duration - active_us) / 1e3, cuda_active_fraction=active_us / duration)


def operator_tables(prof):
    totals = defaultdict(lambda: [0, 0, 0])
    events = prof.events()
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU or event.name.startswith(PREFIX):
            continue
        # Scope rows include their descendants; operator rows use self time only.
        parent = event.cpu_parent
        scopes = set()
        while parent is not None:
            if parent.name.startswith(PREFIX):
                scopes.add(parent.name[len(PREFIX):].split(".")[0])
            parent = parent.cpu_parent
        for scope in scopes:
            counts = totals[scope, event.name]
            counts[0] += 1
            counts[1] += event.self_device_time_total
            counts[2] += event.self_cpu_time_total

    tables = {}
    for scope in (
        "pairformer_module", "msa_module", "template_module", "confidence_module",
        "diffusion_conditioning", "diffusion_sample", "denoiser", "affinity_module1", "affinity_module2",
    ):
        rows = [
            dict(operator=op, calls=n, self_cuda_us=gpu, self_cpu_us=cpu)
            for (group, op), (n, gpu, cpu) in totals.items() if group == scope
        ]
        tables[scope] = {
            key: sorted((r for r in rows if r[key] > 0), key=lambda r: r[key], reverse=True)[:20]
            for key in ("self_cuda_us", "self_cpu_us")
        }
    return tables


class HSP90Profile:
    def __init__(self, trace_dir=None):
        self.trace_dir = trace_dir
        self.items = defaultdict(dict)
        self.flags = {}
        self.features = {}
        self.tables = {}
        self.msa_parses = []

    def parse_a3m(self, parser, path, *args, **kwargs):
        start = time.perf_counter()
        result = parser(path, *args, **kwargs)
        self.msa_parses.append(dict(path=str(path), seconds=time.perf_counter() - start))
        return result

    @contextmanager
    def stage(self, model, stage):
        self.stage_name = stage
        self.flags[stage] = {key: bool(getattr(model, key)) for key in (
            "use_templates", "use_kernels", "run_trunk_and_structure", "skip_run_structure",
        )}
        with ExitStack() as stack:
            def wrap(module, scope):
                # Boltz inference calls _orig_mod directly for compiled trunk modules.
                module = getattr(module, "_orig_mod", module)
                original = module.forward

                @functools.wraps(original)
                def forward(*args, **kwargs):
                    name = scope
                    if name == "pairformer_module":
                        name += ".recycle" if self.pairformer_calls else ".first"
                        self.pairformer_calls += 1
                    elif name == "input_embedder" and kwargs.get("affinity", False):
                        name += ".affinity"
                    elif name == "confidence_module":
                        name += ".overhead" if (kwargs.get("run_sequentially", False)
                                               and kwargs["multiplicity"] > 1) else ".sample"
                    out = self.measure(name, original, *args, **kwargs)
                    if self.trace_dir is None and name not in self.dtypes:
                        self.dtypes[name] = sorted(activation_dtypes(out))
                    return out

                stack.enter_context(patch.object(module, "forward", forward))

            for name in ("input_embedder", "template_module", "msa_module", "pairformer_module",
                         "distogram_module", "diffusion_conditioning", "confidence_module",
                         "affinity_module1", "affinity_module2"):
                module = getattr(model, name, None)
                if module is not None:
                    wrap(module, name)
                    if name in ("affinity_module1", "affinity_module2"):
                        inner = getattr(module, "_orig_mod", module)
                        wrap(inner.pairformer_stack, name + ".trunk")
                        wrap(inner.affinity_heads, name + ".head")
            yield

    def measure(self, name, fn, *args, **kwargs):
        if self.trace_dir is not None:
            label = "denoiser" if name == "diffusion_sample.denoiser" else name
            with torch.profiler.record_function(PREFIX + label):
                return fn(*args, **kwargs)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        index = len(self.events)
        self.events.append((name, start, stop, self.parent))
        previous = self.parent
        self.parent = index
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            stop.record()
            self.parent = previous

    def forward(self, fn, feats, *args, **kwargs):
        stage = self.stage_name
        ligand = feats["record"][0].id
        self.pairformer_calls = 0
        if self.trace_dir is not None:
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(PREFIX + "forward"):
                    start = time.perf_counter()
                    out = fn(feats, *args, **kwargs)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
            trace_path = self.trace_dir / f"{stage}_trace.json"
            prof.export_chrome_trace(str(trace_path))
            self.tables[stage] = dict(operators=operator_tables(prof), **cuda_activity(trace_path))
            return out, wall, 0.0  # Pass B records never enter the timing summary.

        self.dtypes = {}
        self.events = []
        self.parent = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.measure("forward", fn, feats, *args, **kwargs)
        self.events[0][2].synchronize()
        wall = time.perf_counter() - start
        inclusive = [a.elapsed_time(b) for _, a, b, _ in self.events]
        exclusive = inclusive.copy()
        for i, (_, _, _, parent) in enumerate(self.events):
            if parent is not None:
                exclusive[parent] -= inclusive[i]

        aliases = {
            "forward": "forward.other",
            "diffusion_sample": "diffusion_sample.sampler_overhead",
            "affinity_module1": "affinity_module1.conditioning",
            "affinity_module2": "affinity_module2.conditioning",
        }
        rows = {name: dict(calls=0, exclusive_ms=0.0, inclusive_ms=0.0) for name in SCOPES}
        sample_ms = 0.0
        for (name, _, _, _), total, own in zip(self.events, inclusive, exclusive):
            if name == "diffusion_sample":
                sample_ms += total
            row = rows[aliases.get(name, name)]
            row["calls"] += 1
            row["inclusive_ms"] += total
            row["exclusive_ms"] += own
        assert all(r["exclusive_ms"] >= -0.05 for r in rows.values()), rows
        assert abs(sum(r["exclusive_ms"] for r in rows.values()) - inclusive[0]) < 1e-6
        denoiser = rows["diffusion_sample.denoiser"]
        assert denoiser["calls"] == kwargs["num_sampling_steps"]
        assert rows["pairformer_module.recycle"]["calls"] == kwargs["recycling_steps"]
        assert rows["confidence_module.sample"]["calls"] == kwargs["diffusion_samples"]
        self.items[stage][ligand] = dict(
            forward_wall_ms=wall * 1e3, forward_cuda_span_ms=inclusive[0], scopes=rows,
            activation_dtypes=self.dtypes, denoiser_ms_per_call=denoiser["exclusive_ms"] / denoiser["calls"],
        )
        self.features[stage, ligand] = {name: feats[name] for name in (
            "token_pad_mask", "atom_pad_mask", "msa_mask", "template_mask",
        )}
        return out, wall, sample_ms / 1e3

    def summarize(self, run):
        for (stage, ligand), masks in self.features.items():
            token = masks["token_pad_mask"].bool()
            self.items[stage][ligand]["features"] = dict(
                n_tokens=int(token.sum()), n_atoms=int(masks["atom_pad_mask"].sum()),
                n_msa=int((masks["msa_mask"].bool() & token[:, None, :]).any(-1).sum()),
                n_templates=int(masks["template_mask"].bool().any(-1).sum()),
                template_slots=masks["template_mask"].shape[1],
            )
        self.features.clear()
        assert run["all_cache_hits"]
        assert len(self.msa_parses) == run["molecules"], self.msa_parses
        report = {}
        for stage in STAGES:
            items = self.items[stage]
            assert set(items) == set(run["ligand_names"]), (stage, list(items))
            cuda_ms = sum(item["forward_cuda_span_ms"] for item in items.values())
            trainer = run["stages"][stage]
            report[stage] = dict(
                flags=self.flags[stage], items=items,
                forward_wall_ms=stats([i["forward_wall_ms"] for i in items.values()]),
                forward_cuda_span_ms=stats([i["forward_cuda_span_ms"] for i in items.values()]),
                denoiser_ms_per_call=stats([i["denoiser_ms_per_call"] for i in items.values()]),
                trainer_outside_forward_seconds=trainer["trainer_wall_seconds"] - trainer["model_forward_seconds"],
                scopes={name: dict(
                    calls=stats([i["scopes"][name]["calls"] for i in items.values()]),
                    exclusive_ms=stats([i["scopes"][name]["exclusive_ms"] for i in items.values()]),
                    share_of_forward_span=sum(i["scopes"][name]["exclusive_ms"] for i in items.values()) / cuda_ms,
                ) for name in SCOPES},
            )
        msa_seconds = sum(item["seconds"] for item in self.msa_parses)
        return dict(run=run, stages=report, preprocessing=dict(
            msa_parses=self.msa_parses, msa_parse_seconds=msa_seconds,
            other_seconds=run["preprocessing_seconds"] - msa_seconds,
        ), cli_outside_trainer_preprocessing_load_seconds=(
            run["steady_state_wall_seconds"] - run["preprocessing_seconds"]
            - sum(s["trainer_wall_seconds"] for s in run["stages"].values())
        ))


def run_profiles():
    parser = argparse.ArgumentParser(description="Two-pass profiling of resident Boltz-2 on HSP90")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    assert version("boltz") == "2.2.1"
    ligands = sorted(args.input.glob("hsp90_*.yaml"), key=lambda p: int(p.stem.split("_")[1]))[:8]
    assert len(ligands) == 8
    batch = args.out_dir / "inputs"
    batch.mkdir()
    for ligand in ligands:
        (batch / ligand.name).symlink_to(ligand.resolve())

    # Serialize preprocessing so A3M timings are in-process and additive.
    runner = ResidentBoltzRunner(preprocessing_threads=1)
    for label, config in CONFIGS.items():
        runner.run_once(label, config, ligands[0], args.out_dir / "warmup" / label, seed=0, repeat=0)

    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "boltz": version("boltz"),
              "cuequivariance": version("cuequivariance"), "triton": version("triton"), "pass_a": {}, "pass_b": {}}
    parse_a3m = main.parse_a3m
    for label, config in CONFIGS.items():
        profile = runner.profile = HSP90Profile()
        with patch.object(main, "parse_a3m", functools.partial(profile.parse_a3m, parse_a3m)):
            run = runner.run_once(label, config, batch, args.out_dir / "pass_a" / label, seed=0, repeat=0)
        report["pass_a"][label] = profile.summarize(run)
        (args.out_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[PASS A] {label}: {run['gpu_seconds_per_molecule']:.3f} s/mol", flush=True)

    for label, config in CONFIGS.items():
        trace_dir = args.out_dir / "pass_b" / label
        trace_dir.mkdir(parents=True)
        profile = runner.profile = HSP90Profile(trace_dir)
        runner.run_once(label, config, ligands[0], trace_dir, seed=0, repeat=0)
        assert set(profile.tables) == set(STAGES)
        report["pass_b"][label] = profile.tables
        (args.out_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[PASS B] {label}: " + ", ".join(
            f"{stage} active={row['cuda_active_fraction']:.1%}" for stage, row in profile.tables.items()
        ), flush=True)


if __name__ == "__main__":
    run_profiles()
