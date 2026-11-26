#!/usr/bin/env python3
"""Benchmark Direction 1: Structure 0-Drift + Affinity-Only TeaCache & Cross-Stage Reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_resident_compare import kabsch_align, load_cif_atoms
from boltz import main
from experiments.exp_resident_compare import PRESET_CONFIGS, ResidentBoltzRunner


def _rel_l1(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().abs().sum().item()
    if denom <= 1e-12:
        return 0.0
    return float(diff.float().abs().sum().item() / denom)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    na = af.norm()
    nb = bf.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(af, bf) / (na * nb))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


MODES = (
    "baseline",
    "aff_msa_r1_adapt03",
    "aff_msa_r1_adapt05",
    "xstage_direct",
    "xstage_polish1_msacache",
    "xstage_polish1",
    "xstage_polish2",
)


class AffinityReuseController:
    """Applies Structure-0-Drift Affinity-Only TeaCache and Cross-Stage Reuse."""

    def __init__(self):
        self.mode = "baseline"
        self.struct_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.cross_stage_diag: dict[str, dict] = {}
        self.reset_stats()

    def reset_stats(self):
        self.msa_recycles = {"structure": [], "affinity": []}
        self.pf_recycles = {"structure": [], "affinity": []}
        self.fallback_count = 0

    def wrap_boltz2_forward(self, orig_forward):
        ctrl = self

        def wrapped_forward(model, feats, *args, **kwargs):
            mode = ctrl.mode
            stage = "affinity" if getattr(model, "affinity_prediction", False) else "structure"
            lig = feats["record"][0].id
            n_rec = int(kwargs.get("recycling_steps", 0))

            msa_mod = getattr(model.msa_module, "_orig_mod", model.msa_module)
            pf_mod = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
            orig_msa_fwd = msa_mod.forward
            orig_pf_fwd = pf_mod.forward

            # Structure stage: 100% unmodified; only record read-only snapshots.
            if stage == "structure":
                state = {"msa_r": 0, "pf_r": 0, "msa_delta_r1": None, "msa_delta_r3": None}

                def struct_msa_hook(z, emb, f, *a, **kw):
                    r = state["msa_r"]
                    state["msa_r"] += 1
                    z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                    if r == 1:
                        state["msa_delta_r1"] = (z_out - z).detach().cpu()
                    elif r == n_rec:
                        state["msa_delta_r3"] = (z_out - z).detach().cpu()
                    return z_out

                def struct_pf_hook(s, z, *a, **kw):
                    r = state["pf_r"]
                    state["pf_r"] += 1
                    s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                    if r == n_rec:
                        ctrl.struct_cache[lig] = {
                            "s_r3": s_out.detach().cpu(),
                            "z_r3": z_out.detach().cpu(),
                            "msa_delta_r1": state["msa_delta_r1"],
                            "msa_delta_r3": state["msa_delta_r3"],
                        }
                    return s_out, z_out

                msa_mod.forward = struct_msa_hook
                pf_mod.forward = struct_pf_hook
                try:
                    out = orig_forward(model, feats, *args, **kwargs)
                finally:
                    msa_mod.forward = orig_msa_fwd
                    pf_mod.forward = orig_pf_fwd
                ctrl.msa_recycles["structure"].append(n_rec + 1)
                ctrl.pf_recycles["structure"].append(n_rec + 1)
                return out

            # Affinity stage
            if mode == "baseline":
                cached = ctrl.struct_cache.get(lig)
                diag = ctrl.cross_stage_diag.setdefault(lig, {})
                state = {"msa_r": 0, "pf_r": 0}

                def base_msa_hook(z, emb, f, *a, **kw):
                    r = state["msa_r"]
                    state["msa_r"] += 1
                    z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                    if cached is not None and r == 3 and cached.get("msa_delta_r3") is not None:
                        if z_out.shape == cached["msa_delta_r3"].shape:
                            diag["cos_msa_delta_r3"] = _cos_sim(
                                (z_out - z).detach(),
                                cached["msa_delta_r3"].to(z_out.device),
                            )
                    return z_out

                def base_pf_hook(s, z, *a, **kw):
                    r = state["pf_r"]
                    state["pf_r"] += 1
                    s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                    if cached is not None:
                        if r == 3:
                            sm_s = bool(s_out.shape == cached["s_r3"].shape)
                            sm_z = bool(z_out.shape == cached["z_r3"].shape)
                            diag["shape_match_s"] = sm_s
                            diag["shape_match_z"] = sm_z
                            if sm_s and sm_z:
                                s_ref = cached["s_r3"].to(s_out.device)
                                z_ref = cached["z_r3"].to(z_out.device)
                                diag["rel_l1_s_r3"] = _rel_l1(s_out - s_ref, s_ref)
                                diag["rel_l1_z_r3"] = _rel_l1(z_out - z_ref, z_ref)
                                diag["cos_z_r3"] = _cos_sim(z_out, z_ref)
                        elif r == n_rec and diag.get("shape_match_z"):
                            z_ref = cached["z_r3"].to(z_out.device)
                            diag["rel_l1_z_r5_vs_s3"] = _rel_l1(z_out - z_ref, z_ref)
                            diag["cos_z_r5_vs_s3"] = _cos_sim(z_out, z_ref)
                    return s_out, z_out

                msa_mod.forward = base_msa_hook
                pf_mod.forward = base_pf_hook
                try:
                    out = orig_forward(model, feats, *args, **kwargs)
                finally:
                    msa_mod.forward = orig_msa_fwd
                    pf_mod.forward = orig_pf_fwd
                ctrl.msa_recycles["affinity"].append(n_rec + 1)
                ctrl.pf_recycles["affinity"].append(n_rec + 1)
                return out

            if mode in ("aff_msa_r1_adapt03", "aff_msa_r1_adapt05"):
                thresh = 0.03 if mode == "aff_msa_r1_adapt03" else 0.05
                tok_valid = feats["token_pad_mask"][0].bool()
                mol_type = feats["mol_type"][0]
                prot_idx = torch.where((mol_type == 0) & tok_valid)[0]
                lig_idx = torch.where((mol_type == 3) & tok_valid)[0]
                state = {
                    "msa_r": 0,
                    "pf_r": 0,
                    "msa_exec": 0,
                    "pf_exec": 0,
                    "msa_delta": None,
                    "prev_z": None,
                    "pf_converged": False,
                    "pf_conv_s": None,
                    "pf_conv_z": None,
                }

                def adapt_msa_hook(z, emb, f, *a, **kw):
                    r = state["msa_r"]
                    state["msa_r"] += 1
                    if state["pf_converged"]:
                        return z
                    if r >= 2 and state["msa_delta"] is not None:
                        return z + state["msa_delta"]
                    state["msa_exec"] += 1
                    z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                    if r == 1:
                        state["msa_delta"] = (z_out - z).detach()
                    return z_out

                def adapt_pf_hook(s, z, *a, **kw):
                    r = state["pf_r"]
                    state["pf_r"] += 1
                    if state["pf_converged"]:
                        return state["pf_conv_s"], state["pf_conv_z"]
                    state["pf_exec"] += 1
                    s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                    if r >= 1 and state["prev_z"] is not None:
                        z_pl_cur = z_out[:, prot_idx][:, :, lig_idx]
                        z_pl_prev = state["prev_z"][:, prot_idx][:, :, lig_idx]
                        if _rel_l1(z_pl_cur - z_pl_prev, z_pl_prev) < thresh:
                            state["pf_converged"] = True
                            state["pf_conv_s"] = s_out.detach()
                            state["pf_conv_z"] = z_out.detach()
                    state["prev_z"] = z_out.detach()
                    return s_out, z_out

                msa_mod.forward = adapt_msa_hook
                pf_mod.forward = adapt_pf_hook
                try:
                    out = orig_forward(model, feats, *args, **kwargs)
                finally:
                    msa_mod.forward = orig_msa_fwd
                    pf_mod.forward = orig_pf_fwd
                ctrl.msa_recycles["affinity"].append(state["msa_exec"])
                ctrl.pf_recycles["affinity"].append(state["pf_exec"])
                return out

            # Cross-stage reuse modes: xstage_direct, xstage_polish1_msacache, xstage_polish1, xstage_polish2
            cached = ctrl.struct_cache.get(lig)
            state = {"msa_r": 0, "pf_r": 0, "msa_exec": 0, "pf_exec": 0, "fallback": False}

            def _check_fallback(s_like, z_like) -> bool:
                if state["fallback"]:
                    return True
                if cached is None:
                    state["fallback"] = True
                    ctrl.fallback_count += 1
                    return True
                if (s_like is not None and s_like.shape != cached["s_r3"].shape) or (
                    z_like is not None and z_like.shape != cached["z_r3"].shape
                ):
                    state["fallback"] = True
                    ctrl.fallback_count += 1
                    return True
                return False

            def xstage_msa_hook(z, emb, f, *a, **kw):
                r = state["msa_r"]
                state["msa_r"] += 1
                if _check_fallback(None, z):
                    state["msa_exec"] += 1
                    return orig_msa_fwd(z, emb, f, *a, **kw)

                if mode == "xstage_direct":
                    return z
                if mode == "xstage_polish1_msacache":
                    if r < n_rec:
                        return z
                    if cached.get("msa_delta_r3") is not None and cached["msa_delta_r3"].shape == z.shape:
                        return z + cached["msa_delta_r3"].to(device=z.device, dtype=z.dtype)
                    state["msa_exec"] += 1
                    return orig_msa_fwd(z, emb, f, *a, **kw)
                if mode == "xstage_polish1":
                    if r < n_rec:
                        return z
                    state["msa_exec"] += 1
                    return orig_msa_fwd(z, emb, f, *a, **kw)
                if mode == "xstage_polish2":
                    if r < n_rec - 1:
                        return z
                    state["msa_exec"] += 1
                    return orig_msa_fwd(z, emb, f, *a, **kw)
                raise ValueError(f"Unknown mode: {mode}")

            def xstage_pf_hook(s, z, *a, **kw):
                r = state["pf_r"]
                state["pf_r"] += 1
                if _check_fallback(s, z):
                    state["pf_exec"] += 1
                    return orig_pf_fwd(s, z, *a, **kw)

                if mode == "xstage_direct":
                    if r < n_rec:
                        return s, z
                    return (
                        cached["s_r3"].to(device=s.device, dtype=s.dtype),
                        cached["z_r3"].to(device=z.device, dtype=z.dtype),
                    )
                if mode in ("xstage_polish1_msacache", "xstage_polish1"):
                    if r < n_rec - 1:
                        return s, z
                    if r == n_rec - 1:
                        return (
                            cached["s_r3"].to(device=s.device, dtype=s.dtype),
                            cached["z_r3"].to(device=z.device, dtype=z.dtype),
                        )
                    state["pf_exec"] += 1
                    return orig_pf_fwd(s, z, *a, **kw)
                if mode == "xstage_polish2":
                    if r < n_rec - 2:
                        return s, z
                    if r == n_rec - 2:
                        return (
                            cached["s_r3"].to(device=s.device, dtype=s.dtype),
                            cached["z_r3"].to(device=z.device, dtype=z.dtype),
                        )
                    state["pf_exec"] += 1
                    return orig_pf_fwd(s, z, *a, **kw)
                raise ValueError(f"Unknown mode: {mode}")

            msa_mod.forward = xstage_msa_hook
            pf_mod.forward = xstage_pf_hook
            try:
                out = orig_forward(model, feats, *args, **kwargs)
            finally:
                msa_mod.forward = orig_msa_fwd
                pf_mod.forward = orig_pf_fwd

            ctrl.msa_recycles["affinity"].append(state["msa_exec"])
            ctrl.pf_recycles["affinity"].append(state["pf_exec"])
            return out

        return wrapped_forward


def evaluate_mode(label: str, rec: dict, ref_rec: dict, ctrl: AffinityReuseController) -> dict:
    ligs = sorted(ref_rec["scores"])
    n = rec["molecules"]
    out = {
        "label": label,
        "molecules": n,
        "steady_s_mol": rec["gpu_seconds_per_molecule"],
        "speedup_vs_baseline": ref_rec["gpu_seconds_per_molecule"] / rec["gpu_seconds_per_molecule"],
        "struct_fwd_s_mol": rec["stages"]["structure"]["model_forward_seconds"] / n,
        "aff_fwd_s_mol": rec["stages"]["affinity"]["model_forward_seconds"] / n,
        "peak_allocated_gib": max(
            rec["stages"]["structure"]["peak_allocated_gib"],
            rec["stages"]["affinity"]["peak_allocated_gib"],
        ),
        "all_cache_hits": rec["all_cache_hits"],
        "fallback_count": ctrl.fallback_count,
        "mean_msa_recycles_struct": float(np.mean(ctrl.msa_recycles["structure"])) if ctrl.msa_recycles["structure"] else None,
        "mean_msa_recycles_aff": float(np.mean(ctrl.msa_recycles["affinity"])) if ctrl.msa_recycles["affinity"] else None,
        "mean_pf_recycles_struct": float(np.mean(ctrl.pf_recycles["structure"])) if ctrl.pf_recycles["structure"] else None,
        "mean_pf_recycles_aff": float(np.mean(ctrl.pf_recycles["affinity"])) if ctrl.pf_recycles["affinity"] else None,
    }
    for m in ("affinity_pred_value", "affinity_probability_binary"):
        va = np.array([ref_rec["scores"][k][m] for k in ligs], dtype=float)
        vb = np.array([rec["scores"][k][m] for k in ligs], dtype=float)
        if np.isnan(vb).any():
            raise ValueError(f"NaN detected in {label} {m}")
        diff = np.abs(va - vb)
        out[m] = {
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "pearson_r": float(np.corrcoef(va, vb)[0, 1]),
            "spearman_rho": _spearman(va, vb),
        }
    ca_rmsds, lig_rmsds = [], []
    for k in ligs:
        c_ref, keys_ref, ca_ref, lg_ref = load_cif_atoms(ref_rec["cif_paths"][k])
        c_new, keys_new, ca_new, lg_new = load_cif_atoms(rec["cif_paths"][k])
        if keys_ref == keys_new:
            aligned = kabsch_align(c_new, c_ref, ca_ref)
            ca_rmsds.append(float(np.sqrt(np.mean(np.sum((aligned[ca_ref] - c_ref[ca_ref]) ** 2, axis=-1)))))
            lig_rmsds.append(float(np.sqrt(np.mean(np.sum((aligned[lg_ref] - c_ref[lg_ref]) ** 2, axis=-1)))))
    out["median_ca_rmsd_A"] = float(np.median(ca_rmsds)) if ca_rmsds else None
    out["max_ca_rmsd_A"] = float(np.max(ca_rmsds)) if ca_rmsds else None
    out["median_ligand_rmsd_A"] = float(np.median(lig_rmsds)) if lig_rmsds else None
    out["max_ligand_rmsd_A"] = float(np.max(lig_rmsds)) if lig_rmsds else None
    return out


def aggregate_cross_stage_diag(diag_by_lig: dict[str, dict]) -> dict:
    if not diag_by_lig:
        return {}
    keys = (
        "shape_match_s",
        "shape_match_z",
        "rel_l1_s_r3",
        "rel_l1_z_r3",
        "cos_z_r3",
        "rel_l1_z_r5_vs_s3",
        "cos_z_r5_vs_s3",
        "cos_msa_delta_r3",
    )
    summary = {"n_ligands": len(diag_by_lig)}
    for k in keys:
        vals = [d[k] for d in diag_by_lig.values() if k in d]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            summary[k] = all(vals)
        else:
            summary[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    return summary


def main_cli():
    parser = argparse.ArgumentParser(description="Benchmark Direction 1: Structure 0-Drift + Affinity Reuse")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-ligands", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    yamls = sorted(args.input.glob("hsp90_*.yaml"), key=lambda p: int(p.stem.split("_")[1]))[: args.n_ligands]
    assert len(yamls) == args.n_ligands, f"Expected {args.n_ligands} yamls, got {len(yamls)}"

    batch_dir = args.out_dir / f"inputs{args.n_ligands}"
    batch_dir.mkdir(exist_ok=True)
    for y in yamls:
        dst = batch_dir / y.name
        if not dst.exists():
            dst.symlink_to(y.resolve())

    ctrl = AffinityReuseController()
    orig_boltz2_fwd = main.Boltz2.forward
    main.Boltz2.forward = ctrl.wrap_boltz2_forward(orig_boltz2_fwd)

    runner = ResidentBoltzRunner(num_workers=2, preprocessing_threads=2)
    cfg = PRESET_CONFIGS["dpm2m_16_affkern"]

    print("=== Warmup (1 ligand) ===", flush=True)
    ctrl.mode = "baseline"
    runner.run_once("warmup", cfg, yamls[0], args.out_dir / "_warmup", seed=0, repeat=0, bypass_cache=False, tag="warmup")

    results = {}
    raw_recs = {}
    for idx, mode in enumerate(MODES):
        print(f"=== [{idx + 1}/{len(MODES)}] Running mode: {mode} ===", flush=True)
        ctrl.mode = mode
        ctrl.reset_stats()
        if mode == "baseline":
            ctrl.cross_stage_diag.clear()
        rec = runner.run_once(
            label=mode,
            config=cfg,
            input_path=batch_dir,
            out_dir=args.out_dir / mode,
            seed=0,
            repeat=0,
            bypass_cache=False,
            tag="benchmark",
        )
        assert rec["molecules"] == args.n_ligands, f"{mode}: got {rec['molecules']} molecules"
        assert rec["all_cache_hits"], f"{mode}: all_cache_hits is False"
        raw_recs[mode] = rec
        ev = evaluate_mode(mode, rec, raw_recs["baseline"], ctrl)
        results[mode] = ev
        print(
            f"  {mode:24s}: {ev['steady_s_mol']:.3f} s/mol ({ev['speedup_vs_baseline']:.2f}x) | "
            f"struct_fwd={ev['struct_fwd_s_mol']:.3f}s aff_fwd={ev['aff_fwd_s_mol']:.3f}s | "
            f"val_rho={ev['affinity_pred_value']['spearman_rho']:.4f} "
            f"prob_rho={ev['affinity_probability_binary']['spearman_rho']:.4f} | "
            f"med_ca={ev['median_ca_rmsd_A']:.3f}A med_lig={ev['median_ligand_rmsd_A']:.3f}A "
            f"fb={ev['fallback_count']}",
            flush=True,
        )

    cross_stage_summary = aggregate_cross_stage_diag(ctrl.cross_stage_diag)
    print(f"=== Cross-Stage Diag: {json.dumps(cross_stage_summary)} ===", flush=True)

    out_json = args.out_dir / "affinity_reuse.json"
    out_json.write_text(
        json.dumps(
            {
                "config": "dpm2m_16_affkern",
                "n_ligands": args.n_ligands,
                "cross_stage_diag": cross_stage_summary,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved summary to {out_json}", flush=True)


if __name__ == "__main__":
    main_cli()
