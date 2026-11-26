#!/usr/bin/env python3
"""Benchmark Trunk-TeaCache algorithmic optimizations on HSP90 (dpm2m_16_affkern)."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


MODES = (
    "baseline",
    "msa_tc_r0",
    "msa_tc_r1",
    "pf_tc_skip_odd",
    "pf_adaptive_exit",
    "trunk_tc_full",
    "trunk_tc_adaptive",
)


class TrunkTeaCacheController:
    """Applies Trunk-TeaCache policies to Boltz2.forward during inference."""

    def __init__(self, adaptive_thresh: float = 0.03):
        self.mode = "baseline"
        self.adaptive_thresh = adaptive_thresh
        self.reset_stats()

    def reset_stats(self):
        self.msa_recycles = {"structure": [], "affinity": []}
        self.pf_recycles = {"structure": [], "affinity": []}

    def wrap_boltz2_forward(self, orig_forward):
        ctrl = self

        def wrapped_forward(model, feats, *args, **kwargs):
            mode = ctrl.mode
            stage = "affinity" if getattr(model, "affinity_prediction", False) else "structure"

            if mode == "baseline":
                n_rec = int(kwargs.get("recycling_steps", 0)) + 1
                ctrl.msa_recycles[stage].append(n_rec)
                ctrl.pf_recycles[stage].append(n_rec)
                return orig_forward(model, feats, *args, **kwargs)

            tok_valid = feats["token_pad_mask"][0].bool()
            mol_type = feats["mol_type"][0]
            prot_idx = torch.where((mol_type == 0) & tok_valid)[0]
            lig_idx = torch.where((mol_type == 3) & tok_valid)[0]

            msa_mod = getattr(model.msa_module, "_orig_mod", model.msa_module)
            pf_mod = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
            orig_msa_fwd = msa_mod.forward
            orig_pf_fwd = pf_mod.forward

            use_msa_r0 = mode == "msa_tc_r0"
            use_msa_r1 = mode in ("msa_tc_r1", "trunk_tc_full", "trunk_tc_adaptive")
            use_pf_skip_odd = mode in ("pf_tc_skip_odd", "trunk_tc_full")
            use_pf_adaptive = mode in ("pf_adaptive_exit", "trunk_tc_adaptive")

            state = {
                "msa_r": 0,
                "pf_r": 0,
                "msa_exec": 0,
                "pf_exec": 0,
                "msa_delta": None,
                "odd_deltas": {},
                "prev_z": None,
                "pf_converged": False,
                "pf_conv_s": None,
                "pf_conv_z": None,
            }

            def msa_hook(z, emb, f, *a, **kw):
                r = state["msa_r"]
                state["msa_r"] += 1

                if use_pf_adaptive and state["pf_converged"]:
                    return z

                if use_msa_r0 and r >= 1 and state["msa_delta"] is not None:
                    return z + state["msa_delta"]
                if use_msa_r1 and r >= 2 and state["msa_delta"] is not None:
                    return z + state["msa_delta"]

                state["msa_exec"] += 1
                z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                if (use_msa_r0 and r == 0) or (use_msa_r1 and r == 1):
                    state["msa_delta"] = (z_out - z).detach()
                return z_out

            def pf_hook(s, z, *a, **kw):
                r = state["pf_r"]
                state["pf_r"] += 1

                if use_pf_adaptive and state["pf_converged"]:
                    return state["pf_conv_s"], state["pf_conv_z"]

                state["pf_exec"] += 1
                s_out, z_out = orig_pf_fwd(s, z, *a, **kw)

                if use_pf_adaptive:
                    if r >= 1 and state["prev_z"] is not None:
                        z_pl_cur = z_out[:, prot_idx][:, :, lig_idx]
                        z_pl_prev = state["prev_z"][:, prot_idx][:, :, lig_idx]
                        delta_pl = _rel_l1(z_pl_cur - z_pl_prev, z_pl_prev)
                        if delta_pl < ctrl.adaptive_thresh:
                            state["pf_converged"] = True
                            state["pf_conv_s"] = s_out.detach()
                            state["pf_conv_z"] = z_out.detach()
                    state["prev_z"] = z_out.detach()

                return s_out, z_out

            layer_patches = []
            if use_pf_skip_odd:
                for l_idx, layer in enumerate(pf_mod.layers):
                    if l_idx % 2 == 1:
                        orig_l_fwd = layer.forward

                        def make_layer_hook(idx, fn):
                            def l_hook(s_in, z_in, *a, **kw):
                                r = state["pf_r"] - 1
                                if r >= 2 and idx in state["odd_deltas"]:
                                    ds_l, dz_l = state["odd_deltas"][idx]
                                    return s_in + ds_l, z_in + dz_l
                                s_o, z_o = fn(s_in, z_in, *a, **kw)
                                if r == 1:
                                    state["odd_deltas"][idx] = (
                                        (s_o - s_in).detach(),
                                        (z_o - z_in).detach(),
                                    )
                                return s_o, z_o

                            return l_hook

                        layer.forward = make_layer_hook(l_idx, orig_l_fwd)
                        layer_patches.append((layer, orig_l_fwd))

            msa_mod.forward = msa_hook
            pf_mod.forward = pf_hook
            try:
                out = orig_forward(model, feats, *args, **kwargs)
            finally:
                msa_mod.forward = orig_msa_fwd
                pf_mod.forward = orig_pf_fwd
                for layer, orig_l_fwd in layer_patches:
                    layer.forward = orig_l_fwd

            ctrl.msa_recycles[stage].append(state["msa_exec"])
            ctrl.pf_recycles[stage].append(state["pf_exec"])
            return out

        return wrapped_forward


def evaluate_mode(label: str, rec: dict, ref_rec: dict, ctrl_stats: dict) -> dict:
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
        "mean_msa_recycles_struct": float(np.mean(ctrl_stats["msa"]["structure"])) if ctrl_stats["msa"]["structure"] else None,
        "mean_msa_recycles_aff": float(np.mean(ctrl_stats["msa"]["affinity"])) if ctrl_stats["msa"]["affinity"] else None,
        "mean_pf_recycles_struct": float(np.mean(ctrl_stats["pf"]["structure"])) if ctrl_stats["pf"]["structure"] else None,
        "mean_pf_recycles_aff": float(np.mean(ctrl_stats["pf"]["affinity"])) if ctrl_stats["pf"]["affinity"] else None,
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


def main_cli():
    parser = argparse.ArgumentParser(description="Benchmark Trunk-TeaCache on HSP90")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-ligands", type=int, default=12)
    parser.add_argument("--adaptive-thresh", type=float, default=0.03)
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

    ctrl = TrunkTeaCacheController(adaptive_thresh=args.adaptive_thresh)
    orig_boltz2_fwd = main.Boltz2.forward
    main.Boltz2.forward = ctrl.wrap_boltz2_forward(orig_boltz2_fwd)

    runner = ResidentBoltzRunner(num_workers=2, preprocessing_threads=2)
    cfg = PRESET_CONFIGS["dpm2m_16_affkern"]

    print("=== Warmup (1 ligand) ===", flush=True)
    ctrl.mode = "baseline"
    runner.run_once("warmup", cfg, yamls[0], args.out_dir / "_warmup", seed=0, repeat=0, tag="warmup")

    results = {}
    raw_recs = {}
    for idx, mode in enumerate(MODES):
        print(f"=== [{idx + 1}/{len(MODES)}] Running mode: {mode} ===", flush=True)
        ctrl.mode = mode
        ctrl.reset_stats()
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
        stats_snap = {
            "msa": {k: list(v) for k, v in ctrl.msa_recycles.items()},
            "pf": {k: list(v) for k, v in ctrl.pf_recycles.items()},
        }
        ev = evaluate_mode(mode, rec, raw_recs["baseline"], stats_snap)
        results[mode] = ev
        print(
            f"  {mode:20s}: {ev['steady_s_mol']:.3f} s/mol ({ev['speedup_vs_baseline']:.2f}x) | "
            f"struct_fwd={ev['struct_fwd_s_mol']:.3f}s aff_fwd={ev['aff_fwd_s_mol']:.3f}s | "
            f"val_rho={ev['affinity_pred_value']['spearman_rho']:.4f} "
            f"prob_rho={ev['affinity_probability_binary']['spearman_rho']:.4f} | "
            f"med_ca={ev['median_ca_rmsd_A']:.3f}A med_lig={ev['median_ligand_rmsd_A']:.3f}A",
            flush=True,
        )

    out_json = args.out_dir / "trunk_teacache.json"
    import json

    out_json.write_text(
        json.dumps(
            {
                "config": "dpm2m_16_affkern",
                "n_ligands": args.n_ligands,
                "adaptive_thresh": args.adaptive_thresh,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved summary to {out_json}", flush=True)


if __name__ == "__main__":
    main_cli()
