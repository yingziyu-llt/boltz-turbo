#!/usr/bin/env python3
"""Benchmark Cross-Ligand (Inter-System) Protein Caching on HSP90 (dpm2m_16_affkern)."""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def _extract_prot_keys(feats: dict, prot_idx: torch.Tensor) -> list[tuple[int, int]]:
    if "asym_id" in feats and "residue_index" in feats:
        asym = feats["asym_id"][0]
        res = feats["residue_index"][0]
        return [(int(asym[i].item()), int(res[i].item())) for i in prot_idx]
    return [(0, int(idx)) for idx in range(len(prot_idx))]


def _map_prot_indices(
    cur_keys: list[tuple[int, int]],
    ref_key_to_pos: dict[tuple[int, int], int],
    prot_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_pos, ref_pos = [], []
    for i, k in enumerate(cur_keys):
        if k in ref_key_to_pos:
            cur_pos.append(i)
            ref_pos.append(ref_key_to_pos[k])
    device = prot_idx.device
    if not cur_pos:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    cur_t = prot_idx[torch.tensor(cur_pos, dtype=torch.long, device=device)]
    ref_t = torch.tensor(ref_pos, dtype=torch.long, device=device)
    return cur_t, ref_t


def _slice_pf_arg(val, n_tok: int, active_idx: torch.Tensor):
    if isinstance(val, torch.Tensor):
        if val.dim() == 2 and val.shape[1] == n_tok:
            return val[:, active_idx]
        if val.dim() == 3 and val.shape[1] == n_tok and val.shape[2] == n_tok:
            return val[:, active_idx][:, :, active_idx]
    return val


MODES = (
    "baseline",
    "xlig_msa_pp",
    "xlig_msa_aff_adapt05",
    "xlig_pocket128",
    "xlig_pocket96",
    "xlig_warm_r2",
)


class CrossLigandCacheController:
    """Reuses target-protein Trunk representations from ligand 0 across ligands 1..N-1."""

    def __init__(self, ref_ligand: str):
        self.ref_ligand = ref_ligand
        self.mode = "baseline"
        self.target_cache: dict = {}
        self.token_diag: dict[str, list[dict]] = {"structure": [], "affinity": []}
        self.reset_stats()

    def reset_stats(self):
        self.msa_recycles = {"structure": [], "affinity": []}
        self.pf_recycles = {"structure": [], "affinity": []}
        self.per_lig_fwd_s = {"structure": {}, "affinity": {}}

    def wrap_boltz2_forward(self, orig_forward):
        ctrl = self

        def wrapped_forward(model, feats, *args, **kwargs):
            mode = ctrl.mode
            stage = "affinity" if getattr(model, "affinity_prediction", False) else "structure"
            lig = feats["record"][0].id
            n_rec = int(kwargs.get("recycling_steps", 0))

            tok_valid = feats["token_pad_mask"][0].bool()
            mol_type = feats["mol_type"][0]
            prot_idx = torch.where((mol_type == 0) & tok_valid)[0]
            lig_idx = torch.where((mol_type == 3) & tok_valid)[0]
            cur_keys = _extract_prot_keys(feats, prot_idx)

            msa_mod = getattr(model.msa_module, "_orig_mod", model.msa_module)
            pf_mod = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
            orig_msa_fwd = msa_mod.forward
            orig_pf_fwd = pf_mod.forward

            is_ref = lig == ctrl.ref_ligand
            tc = ctrl.target_cache

            if mode == "baseline" and not is_ref and tc:
                cur_m, _ = _map_prot_indices(cur_keys, tc["key_to_pos"], prot_idx)
                ctrl.token_diag[stage].append(
                    {
                        "lig": lig,
                        "n_prot": int(prot_idx.numel()),
                        "n_lig": int(lig_idx.numel()),
                        "n_total": int(tok_valid.sum().item()),
                        "matched_prot_ratio": float(cur_m.numel() / max(1, prot_idx.numel())),
                    }
                )

            # Ligand 0 (or baseline): run 100% unmodified; populate target_cache during structure stage.
            if is_ref or mode == "baseline":
                state = {"msa_r": 0, "pf_r": 0, "msa_delta_pp": {}, "pf_s_p": {}, "pf_z_pp": {}}

                def ref_msa_hook(z, emb, f, *a, **kw):
                    r = state["msa_r"]
                    state["msa_r"] += 1
                    z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                    if is_ref and stage == "structure":
                        state["msa_delta_pp"][r] = (z_out - z)[:, prot_idx][:, :, prot_idx].detach().cpu()
                    return z_out

                def ref_pf_hook(s, z, *a, **kw):
                    r = state["pf_r"]
                    state["pf_r"] += 1
                    s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                    if is_ref and stage == "structure":
                        state["pf_s_p"][r] = s_out[:, prot_idx].detach().cpu()
                        state["pf_z_pp"][r] = z_out[:, prot_idx][:, :, prot_idx].detach().cpu()
                        if r == n_rec:
                            z_pl = z_out[0, prot_idx][:, lig_idx].float()
                            score_i = z_pl.norm(dim=-1).max(dim=-1).values
                            rank_order = torch.argsort(score_i, descending=True).cpu().tolist()
                            ctrl.target_cache = {
                                "prot_keys": cur_keys,
                                "key_to_pos": {k: i for i, k in enumerate(cur_keys)},
                                "msa_delta_pp": state["msa_delta_pp"],
                                "pf_s_p": state["pf_s_p"],
                                "pf_z_pp": state["pf_z_pp"],
                                "pocket_rank_order": rank_order,
                            }
                    return s_out, z_out

                msa_mod.forward = ref_msa_hook
                pf_mod.forward = ref_pf_hook
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    out = orig_forward(model, feats, *args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    msa_mod.forward = orig_msa_fwd
                    pf_mod.forward = orig_pf_fwd
                ctrl.per_lig_fwd_s[stage][lig] = dt
                ctrl.msa_recycles[stage].append(n_rec + 1)
                ctrl.pf_recycles[stage].append(n_rec + 1)
                return out

            # Ligands 1..N-1 with cross-ligand caching
            cur_prot, ref_pos = _map_prot_indices(cur_keys, tc["key_to_pos"], prot_idx)
            ref_pos_cpu = ref_pos.cpu()
            use_aff_adapt = (
                stage == "affinity"
                and mode in ("xlig_msa_aff_adapt05", "xlig_pocket128", "xlig_pocket96", "xlig_warm_r2")
            )
            pocket_k = 128 if mode == "xlig_pocket128" else (96 if mode == "xlig_pocket96" else None)
            use_pocket = stage == "structure" and pocket_k is not None
            use_warm_r2 = stage == "structure" and mode == "xlig_warm_r2"

            active_idx = None
            if use_pocket:
                ref_pos_set = set(tc["pocket_rank_order"][: min(pocket_k, len(tc["pocket_rank_order"]))])
                selected_cur = [i for i, k in enumerate(cur_keys) if tc["key_to_pos"].get(k, -1) in ref_pos_set]
                active_prot = prot_idx[torch.tensor(selected_cur, dtype=torch.long, device=prot_idx.device)]
                active_idx = torch.cat([active_prot, lig_idx])

            state = {
                "msa_r": 0,
                "pf_r": 0,
                "msa_exec": 0,
                "pf_exec": 0,
                "prev_z": None,
                "pf_converged": False,
                "pf_conv_s": None,
                "pf_conv_z": None,
            }

            def xlig_msa_hook(z, emb, f, *a, **kw):
                r = state["msa_r"]
                state["msa_r"] += 1
                if use_aff_adapt and state["pf_converged"]:
                    return z
                r_key = min(r, max(tc["msa_delta_pp"].keys()))
                delta_ref = tc["msa_delta_pp"][r_key][:, ref_pos_cpu][:, :, ref_pos_cpu].to(device=z.device, dtype=z.dtype)
                z_new = z.clone()
                z_new[:, cur_prot.unsqueeze(1), cur_prot.unsqueeze(0)] = (
                    z_new[:, cur_prot.unsqueeze(1), cur_prot.unsqueeze(0)] + delta_ref
                )
                return z_new

            def xlig_pf_hook(s, z, *a, **kw):
                r = state["pf_r"]
                state["pf_r"] += 1
                if use_aff_adapt and state["pf_converged"]:
                    return state["pf_conv_s"], state["pf_conv_z"]

                if use_warm_r2:
                    if r == 0:
                        return s, z
                    if r == 1:
                        s_new = s.clone()
                        z_new = z.clone()
                        s_new[:, cur_prot] = tc["pf_s_p"][1][:, ref_pos_cpu].to(device=s.device, dtype=s.dtype)
                        z_new[:, cur_prot.unsqueeze(1), cur_prot.unsqueeze(0)] = (
                            tc["pf_z_pp"][1][:, ref_pos_cpu][:, :, ref_pos_cpu].to(device=z.device, dtype=z.dtype)
                        )
                        return s_new, z_new

                if use_pocket and active_idx is not None:
                    state["pf_exec"] += 1
                    n_tok = s.shape[1]
                    s_sub = s[:, active_idx]
                    z_sub = z[:, active_idx][:, :, active_idx]
                    a_sub = tuple(_slice_pf_arg(v, n_tok, active_idx) for v in a)
                    kw_sub = {k: _slice_pf_arg(v, n_tok, active_idx) for k, v in kw.items()}
                    s_sub_out, z_sub_out = orig_pf_fwd(s_sub, z_sub, *a_sub, **kw_sub)

                    r_key = min(r, max(tc["pf_s_p"].keys()))
                    s_out = s.to(dtype=s_sub_out.dtype).clone()
                    z_out = z.to(dtype=z_sub_out.dtype).clone()
                    s_out[:, cur_prot] = tc["pf_s_p"][r_key][:, ref_pos_cpu].to(device=s.device, dtype=s_sub_out.dtype)
                    z_out[:, cur_prot.unsqueeze(1), cur_prot.unsqueeze(0)] = (
                        tc["pf_z_pp"][r_key][:, ref_pos_cpu][:, :, ref_pos_cpu].to(device=z.device, dtype=z_sub_out.dtype)
                    )
                    s_out[:, active_idx] = s_sub_out
                    z_out[:, active_idx.unsqueeze(1), active_idx.unsqueeze(0)] = z_sub_out
                    return s_out, z_out

                state["pf_exec"] += 1
                s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                if use_aff_adapt:
                    if r >= 1 and state["prev_z"] is not None:
                        z_pl_cur = z_out[:, prot_idx][:, :, lig_idx]
                        z_pl_prev = state["prev_z"][:, prot_idx][:, :, lig_idx]
                        if _rel_l1(z_pl_cur - z_pl_prev, z_pl_prev) < 0.05:
                            state["pf_converged"] = True
                            state["pf_conv_s"] = s_out.detach()
                            state["pf_conv_z"] = z_out.detach()
                    state["prev_z"] = z_out.detach()
                return s_out, z_out

            msa_mod.forward = xlig_msa_hook
            pf_mod.forward = xlig_pf_hook
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                out = orig_forward(model, feats, *args, **kwargs)
            finally:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                msa_mod.forward = orig_msa_fwd
                pf_mod.forward = orig_pf_fwd

            ctrl.per_lig_fwd_s[stage][lig] = dt
            ctrl.msa_recycles[stage].append(state["msa_exec"])
            ctrl.pf_recycles[stage].append(state["pf_exec"])
            return out

        return wrapped_forward


def evaluate_mode(
    label: str,
    rec: dict,
    ref_rec: dict,
    eval_ligs: list[str],
    ctrl: CrossLigandCacheController,
) -> dict:
    n = rec["molecules"]
    s_fwd_1_11 = float(np.mean([ctrl.per_lig_fwd_s["structure"][k] for k in eval_ligs]))
    a_fwd_1_11 = float(np.mean([ctrl.per_lig_fwd_s["affinity"][k] for k in eval_ligs]))
    tot_fwd_1_11 = s_fwd_1_11 + a_fwd_1_11

    out = {
        "label": label,
        "molecules": n,
        "steady_s_mol": rec["gpu_seconds_per_molecule"],
        "speedup_vs_baseline": ref_rec["gpu_seconds_per_molecule"] / rec["gpu_seconds_per_molecule"],
        "struct_fwd_s_mol": rec["stages"]["structure"]["model_forward_seconds"] / n,
        "aff_fwd_s_mol": rec["stages"]["affinity"]["model_forward_seconds"] / n,
        "ligs_1_to_11_struct_fwd_s_mol": s_fwd_1_11,
        "ligs_1_to_11_aff_fwd_s_mol": a_fwd_1_11,
        "ligs_1_to_11_total_fwd_s_mol": tot_fwd_1_11,
        "peak_allocated_gib": max(
            rec["stages"]["structure"]["peak_allocated_gib"],
            rec["stages"]["affinity"]["peak_allocated_gib"],
        ),
        "all_cache_hits": rec["all_cache_hits"],
        "mean_msa_recycles_struct": float(np.mean(ctrl.msa_recycles["structure"])),
        "mean_msa_recycles_aff": float(np.mean(ctrl.msa_recycles["affinity"])),
        "mean_pf_recycles_struct": float(np.mean(ctrl.pf_recycles["structure"])),
        "mean_pf_recycles_aff": float(np.mean(ctrl.pf_recycles["affinity"])),
    }
    for m in ("affinity_pred_value", "affinity_probability_binary"):
        va = np.array([ref_rec["scores"][k][m] for k in eval_ligs], dtype=float)
        vb = np.array([rec["scores"][k][m] for k in eval_ligs], dtype=float)
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
    for k in eval_ligs:
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
    parser = argparse.ArgumentParser(description="Benchmark Cross-Ligand Protein Caching on HSP90")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-ligands", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    yamls = sorted(args.input.glob("hsp90_*.yaml"), key=lambda p: int(p.stem.split("_")[1]))[: args.n_ligands]
    assert len(yamls) == args.n_ligands, f"Expected {args.n_ligands} yamls, got {len(yamls)}"
    ref_lig = yamls[0].stem
    eval_ligs = [y.stem for y in yamls[1:]]

    batch_dir = args.out_dir / f"inputs{args.n_ligands}"
    batch_dir.mkdir(exist_ok=True)
    for y in yamls:
        dst = batch_dir / y.name
        if not dst.exists():
            dst.symlink_to(y.resolve())

    ctrl = CrossLigandCacheController(ref_ligand=ref_lig)
    orig_boltz2_fwd = main.Boltz2.forward
    main.Boltz2.forward = ctrl.wrap_boltz2_forward(orig_boltz2_fwd)

    runner = ResidentBoltzRunner(num_workers=2, preprocessing_threads=2)
    cfg = PRESET_CONFIGS["dpm2m_16_affkern"]

    print(f"=== Warmup (ref ligand: {ref_lig}) ===", flush=True)
    ctrl.mode = "baseline"
    runner.run_once("warmup", cfg, yamls[0], args.out_dir / "_warmup", seed=0, repeat=0, bypass_cache=False, tag="warmup")

    results = {}
    raw_recs = {}
    base_fwd_1_11 = None
    for idx, mode in enumerate(MODES):
        print(f"=== [{idx + 1}/{len(MODES)}] Running mode: {mode} ===", flush=True)
        ctrl.mode = mode
        ctrl.reset_stats()
        if mode == "baseline":
            ctrl.token_diag = {"structure": [], "affinity": []}
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
        ev = evaluate_mode(mode, rec, raw_recs["baseline"], eval_ligs, ctrl)
        if mode == "baseline":
            base_fwd_1_11 = ev["ligs_1_to_11_total_fwd_s_mol"]
        ev["ligs_1_to_11_fwd_speedup"] = base_fwd_1_11 / ev["ligs_1_to_11_total_fwd_s_mol"]
        results[mode] = ev
        print(
            f"  {mode:22s}: steady={ev['steady_s_mol']:.3f}s ({ev['speedup_vs_baseline']:.2f}x) | "
            f"amort_fwd={ev['ligs_1_to_11_total_fwd_s_mol']:.3f}s ({ev['ligs_1_to_11_fwd_speedup']:.2f}x) | "
            f"val_rho={ev['affinity_pred_value']['spearman_rho']:.4f} "
            f"prob_rho={ev['affinity_probability_binary']['spearman_rho']:.4f} | "
            f"med_ca={ev['median_ca_rmsd_A']:.3f}A med_lig={ev['median_ligand_rmsd_A']:.3f}A",
            flush=True,
        )

    token_summary = {
        st: {
            "n_ligs": len(rows),
            "n_prot_mean": float(np.mean([r["n_prot"] for r in rows])) if rows else None,
            "n_lig_mean": float(np.mean([r["n_lig"] for r in rows])) if rows else None,
            "matched_prot_ratio_min": float(np.min([r["matched_prot_ratio"] for r in rows])) if rows else None,
            "matched_prot_ratio_mean": float(np.mean([r["matched_prot_ratio"] for r in rows])) if rows else None,
        }
        for st, rows in ctrl.token_diag.items()
    }
    print(f"=== Token Diag: {json.dumps(token_summary)} ===", flush=True)

    out_json = args.out_dir / "cross_ligand_cache.json"
    out_json.write_text(
        json.dumps(
            {
                "config": "dpm2m_16_affkern",
                "n_ligands": args.n_ligands,
                "ref_ligand": ref_lig,
                "eval_ligands": eval_ligs,
                "token_diag": token_summary,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved summary to {out_json}", flush=True)


if __name__ == "__main__":
    main_cli()
