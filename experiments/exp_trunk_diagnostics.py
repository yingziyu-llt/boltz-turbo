import argparse
import functools
import gc
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import boltz.main as main
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from analyze_resident_compare import kabsch_align, load_cif_atoms
from exp_resident_compare import PRESET_CONFIGS, ResidentBoltzRunner


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    denom = float(af.norm() * bf.norm())
    if denom == 0.0:
        return 1.0 if torch.equal(af, bf) else 0.0
    return float(torch.dot(af, bf) / denom)


def _rel_l1(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = float(ref.abs().sum())
    if denom == 0.0:
        return 0.0
    return float(diff.abs().sum() / denom)


def compare_checkpoints(boltz_cache: Path) -> dict:
    conf_path = boltz_cache / "boltz2_conf.ckpt"
    aff_path = boltz_cache / "boltz2_aff.ckpt"
    conf_sd = torch.load(conf_path, map_location="cpu", weights_only=False)["state_dict"]
    aff_sd = torch.load(aff_path, map_location="cpu", weights_only=False)["state_dict"]

    conf_keys = set(conf_sd)
    aff_keys = set(aff_sd)
    shared = sorted(conf_keys & aff_keys)

    grouped = defaultdict(list)
    for k in shared:
        grouped[k.split(".")[0]].append(k)

    prefix_stats = {}
    for prefix, keys in sorted(grouped.items()):
        exact = 0
        max_abs = 0.0
        rel_l2s = []
        cos_sims = []
        shape_mismatches = 0
        for k in keys:
            v1, v2 = conf_sd[k], aff_sd[k]
            if v1.shape != v2.shape:
                shape_mismatches += 1
                continue
            if torch.equal(v1, v2):
                exact += 1
                rel_l2s.append(0.0)
                cos_sims.append(1.0)
                continue
            f1, f2 = v1.float(), v2.float()
            d = (f1 - f2).abs()
            max_abs = max(max_abs, float(d.max()))
            rel_l2s.append(float(d.norm() / (f1.norm() + 1e-12)))
            cos_sims.append(_cos_sim(f1, f2))
        prefix_stats[prefix] = {
            "shared_keys": len(keys),
            "exact_equal_keys": exact,
            "shape_mismatches": shape_mismatches,
            "max_abs_diff": max_abs,
            "mean_rel_l2": float(np.mean(rel_l2s)) if rel_l2s else 0.0,
            "min_cos_sim": float(min(cos_sims)) if cos_sims else 1.0,
        }

    only_conf = sorted({k.split(".")[0] for k in (conf_keys - aff_keys)})
    only_aff = sorted({k.split(".")[0] for k in (aff_keys - conf_keys)})
    del conf_sd, aff_sd
    gc.collect()
    return {
        "conf_total_keys": len(conf_keys),
        "aff_total_keys": len(aff_keys),
        "shared_keys": len(shared),
        "only_conf_prefixes": only_conf,
        "only_aff_prefixes": only_aff,
        "by_prefix": prefix_stats,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _aggregate_records(records: list[dict], keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in records if k in r and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    return out


class TrunkDiagnosticController:
    def __init__(self, first_ligand: str):
        self.first_ligand = first_ligand
        self.collect_dynamics = False
        self.msa_probe = None  # None, "msa_r0_only", "msa_top512", "msa_r0_top512"
        self.msa_dynamics = defaultdict(lambda: defaultdict(list))
        self.pf_dynamics = defaultdict(lambda: defaultdict(list))
        self.layer_dynamics = defaultdict(list)  # stage -> list of records (first ligand)
        self.cross_ligand = defaultdict(dict)  # key -> {ligand: tensor}
        self.token_counts = defaultdict(dict)

    def wrap_boltz2_forward(self, orig_forward):
        ctrl = self

        @functools.wraps(orig_forward)
        def wrapped_forward(model, feats, *args, **kwargs):
            stage = "affinity" if getattr(model, "affinity_prediction", False) else "structure"
            lig = feats["record"][0].id
            collect = ctrl.collect_dynamics
            probe = ctrl.msa_probe

            if not collect and probe is None:
                return orig_forward(model, feats, *args, **kwargs)

            tok_valid = feats["token_pad_mask"][0].bool()
            mol_type = feats["mol_type"][0]
            prot_idx = torch.where((mol_type == 0) & tok_valid)[0]
            lig_idx = torch.where((mol_type == 3) & tok_valid)[0]
            if collect:
                ctrl.token_counts[stage][lig] = {
                    "n_prot": int(prot_idx.numel()),
                    "n_lig": int(lig_idx.numel()),
                    "n_total": int(tok_valid.sum()),
                }

            msa_mod = getattr(model.msa_module, "_orig_mod", model.msa_module)
            pf_mod = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
            orig_msa_fwd = msa_mod.forward
            orig_pf_fwd = pf_mod.forward

            state = {"msa_r": 0, "pf_r": 0, "msa_delta0": None, "prev_s": None, "prev_z": None}
            layer_patches = []

            def msa_hook(z, emb, f, *a, **kw):
                r = state["msa_r"]
                state["msa_r"] += 1
                if probe in ("msa_r0_only", "msa_r0_top512") and r >= 1:
                    return z
                if probe in ("msa_top512", "msa_r0_top512"):
                    f = dict(f)
                    for k in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
                        if k in f and f[k] is not None and f[k].shape[1] > 512:
                            f[k] = f[k][:, :512]
                z_out = orig_msa_fwd(z, emb, f, *a, **kw)
                if collect:
                    delta = z_out - z
                    d_pp = delta[:, prot_idx][:, :, prot_idx]
                    d_pl = delta[:, prot_idx][:, :, lig_idx]
                    z_pp = z[:, prot_idx][:, :, prot_idx]
                    z_pl = z[:, prot_idx][:, :, lig_idx]
                    rec = {
                        "rel_l1_pp": _rel_l1(d_pp, z_pp),
                        "rel_l1_pl": _rel_l1(d_pl, z_pl),
                    }
                    if r == 0:
                        state["msa_delta0"] = delta.detach()
                        rec["cos_vs_r0"] = 1.0
                        if stage == "structure":
                            ctrl.cross_ligand["msa_delta0_pp"][lig] = d_pp.detach().float().cpu()
                    else:
                        rec["cos_vs_r0"] = _cos_sim(delta, state["msa_delta0"])
                    ctrl.msa_dynamics[stage][r].append(rec)
                return z_out

            def pf_hook(s, z, *a, **kw):
                r = state["pf_r"]
                state["pf_r"] += 1
                s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
                if collect:
                    if r >= 1 and state["prev_z"] is not None:
                        dz = z_out - state["prev_z"]
                        ds = s_out - state["prev_s"]
                        ctrl.pf_dynamics[stage][r].append({
                            "rel_l1_z_pp": _rel_l1(dz[:, prot_idx][:, :, prot_idx], state["prev_z"][:, prot_idx][:, :, prot_idx]),
                            "rel_l1_z_pl": _rel_l1(dz[:, prot_idx][:, :, lig_idx], state["prev_z"][:, prot_idx][:, :, lig_idx]),
                            "rel_l1_s_p": _rel_l1(ds[:, prot_idx], state["prev_s"][:, prot_idx]),
                            "rel_l1_s_l": _rel_l1(ds[:, lig_idx], state["prev_s"][:, lig_idx]),
                            "cos_z_pp": _cos_sim(z_out[:, prot_idx][:, :, prot_idx], state["prev_z"][:, prot_idx][:, :, prot_idx]),
                            "cos_z_pl": _cos_sim(z_out[:, prot_idx][:, :, lig_idx], state["prev_z"][:, prot_idx][:, :, lig_idx]),
                        })
                    state["prev_s"] = s_out.detach()
                    state["prev_z"] = z_out.detach()
                    if stage == "structure":
                        if r == 0:
                            ctrl.cross_ligand["pf_r0_z_pp"][lig] = z_out[:, prot_idx][:, :, prot_idx].detach().float().cpu()
                        elif r == 3:
                            ctrl.cross_ligand["pf_r3_z_pp"][lig] = z_out[:, prot_idx][:, :, prot_idx].detach().float().cpu()
                            ctrl.cross_ligand["pf_r3_s_p"][lig] = s_out[:, prot_idx].detach().float().cpu()
                return s_out, z_out

            if collect and lig == ctrl.first_ligand:
                prev_layer_delta = {}
                for l_idx, layer in enumerate(pf_mod.layers):
                    orig_l_fwd = layer.forward

                    def make_layer_hook(idx, fn):
                        def l_hook(s_in, z_in, *a, **kw):
                            r = state["pf_r"] - 1  # incremented after pf_hook? Wait: pf_hook calls orig_pf_fwd BEFORE incrementing? No, pf_r += 1 is BEFORE orig_pf_fwd!
                            s_o, z_o = fn(s_in, z_in, *a, **kw)
                            dz_l = (z_o - z_in).detach()
                            dz_pp = dz_l[:, prot_idx][:, :, prot_idx]
                            dz_pl = dz_l[:, prot_idx][:, :, lig_idx]
                            rec = {
                                "recycle": r,
                                "layer": idx,
                                "rel_l1_pp": _rel_l1(dz_pp, z_in[:, prot_idx][:, :, prot_idx]),
                                "rel_l1_pl": _rel_l1(dz_pl, z_in[:, prot_idx][:, :, lig_idx]),
                            }
                            if idx in prev_layer_delta:
                                rec["cos_pp_vs_prev_r"] = _cos_sim(dz_pp, prev_layer_delta[idx][0])
                                rec["cos_pl_vs_prev_r"] = _cos_sim(dz_pl, prev_layer_delta[idx][1])
                            prev_layer_delta[idx] = (dz_pp, dz_pl)
                            ctrl.layer_dynamics[stage].append(rec)
                            return s_o, z_o
                        return l_hook

                    layer_patches.append((layer, orig_l_fwd))
                    layer.forward = make_layer_hook(l_idx, orig_l_fwd)

            msa_mod.forward = msa_hook
            pf_mod.forward = pf_hook
            try:
                return orig_forward(model, feats, *args, **kwargs)
            finally:
                msa_mod.forward = orig_msa_fwd
                pf_mod.forward = orig_pf_fwd
                for layer, orig_l_fwd in layer_patches:
                    layer.forward = orig_l_fwd

        return wrapped_forward


def summarize_cross_ligand(tensors_by_lig: dict[str, torch.Tensor]) -> dict:
    ligs = sorted(tensors_by_lig)
    cos_list = []
    rel_l1_list = []
    for i in range(len(ligs)):
        for j in range(i + 1, len(ligs)):
            a = tensors_by_lig[ligs[i]]
            b = tensors_by_lig[ligs[j]]
            cos_list.append(_cos_sim(a, b))
            rel_l1_list.append(_rel_l1(a - b, 0.5 * (a.abs() + b.abs())))
    stacked = torch.stack([tensors_by_lig[k].squeeze(0) for k in ligs], dim=0)  # (L, N, N, C) or (L, N, C)
    mean_t = stacked.mean(dim=0, keepdim=True)
    rel_dev = (stacked - mean_t).abs() / (mean_t.abs().mean() + 1e-8)
    if stacked.ndim == 4:
        # Per-residue row mean relative deviation across ligands
        per_res = rel_dev.mean(dim=(0, 2, 3)).numpy()
    else:
        per_res = rel_dev.mean(dim=(0, 2)).numpy()
    return {
        "n_ligands": len(ligs),
        "n_pairs": len(cos_list),
        "pair_cos_mean": float(np.mean(cos_list)),
        "pair_cos_min": float(np.min(cos_list)),
        "pair_rel_l1_mean": float(np.mean(rel_l1_list)),
        "pair_rel_l1_max": float(np.max(rel_l1_list)),
        "per_residue_rel_dev_quantiles": {
            q: float(np.quantile(per_res, p))
            for q, p in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("max", 1.0))
        },
    }


def evaluate_probe(label: str, rec: dict, ref_rec: dict) -> dict:
    ligs = sorted(ref_rec["scores"])
    n = rec["molecules"]
    ref_n = ref_rec["molecules"]
    out = {
        "label": label,
        "steady_s_mol": rec["gpu_seconds_per_molecule"],
        "speedup_vs_baseline": ref_rec["gpu_seconds_per_molecule"] / rec["gpu_seconds_per_molecule"],
        "struct_fwd_s_mol": rec["stages"]["structure"]["model_forward_seconds"] / n,
        "aff_fwd_s_mol": rec["stages"]["affinity"]["model_forward_seconds"] / n,
        "peak_allocated_gib": max(
            rec["stages"]["structure"]["peak_allocated_gib"],
            rec["stages"]["affinity"]["peak_allocated_gib"],
        ),
        "all_cache_hits": rec["all_cache_hits"],
    }
    for m in ("affinity_pred_value", "affinity_probability_binary"):
        va = np.array([ref_rec["scores"][k][m] for k in ligs], dtype=float)
        vb = np.array([rec["scores"][k][m] for k in ligs], dtype=float)
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
    out["median_ligand_rmsd_A"] = float(np.median(lig_rmsds)) if lig_rmsds else None
    return out


def main_cli():
    parser = argparse.ArgumentParser(description="4-in-1 Trunk Feature Extraction Diagnostics on HSP90")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-ligands", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    boltz_cache = Path(os.environ.get("BOLTZ_CACHE", "/home/lhlai_pkuhpc/.boltz"))

    print("=== PART 1: CHECKPOINT WEIGHT DIFF ===", flush=True)
    ckpt_diff = compare_checkpoints(boltz_cache)
    for p, st in ckpt_diff["by_prefix"].items():
        print(
            f"  {p:26s} shared={st['shared_keys']:4d} exact={st['exact_equal_keys']:4d} "
            f"max_abs={st['max_abs_diff']:.6f} mean_rel_l2={st['mean_rel_l2']:.6f} min_cos={st['min_cos_sim']:.6f}",
            flush=True,
        )

    yamls = sorted(args.input.glob("hsp90_*.yaml"), key=lambda p: int(p.stem.split("_")[1]))[: args.n_ligands]
    assert len(yamls) == args.n_ligands
    batch_dir = args.out_dir / "inputs6"
    batch_dir.mkdir(exist_ok=True)
    for y in yamls:
        dst = batch_dir / y.name
        if not dst.exists():
            dst.symlink_to(y.resolve())

    ctrl = TrunkDiagnosticController(first_ligand=yamls[0].stem)
    orig_boltz2_fwd = main.Boltz2.forward
    main.Boltz2.forward = ctrl.wrap_boltz2_forward(orig_boltz2_fwd)

    runner = ResidentBoltzRunner(num_workers=2, preprocessing_threads=2)
    base_cfg = PRESET_CONFIGS["dpm2m_16_affkern"]

    # Warmup
    runner.run_once("warmup", base_cfg, yamls[0], args.out_dir / "_warmup", seed=0, repeat=0, tag="warmup")

    print("\n=== PART 2 & 3: BASELINE + RECYCLE/LAYER DYNAMICS + CROSS-LIGAND SIMILARITY ===", flush=True)
    ctrl.collect_dynamics = True
    base_rec = runner.run_once("dpm2m_16_affkern", base_cfg, batch_dir, args.out_dir / "baseline", seed=0, repeat=0)
    ctrl.collect_dynamics = False

    msa_dyn_summary = {
        st: {
            str(r): _aggregate_records(recs, ("rel_l1_pp", "rel_l1_pl", "cos_vs_r0"))
            for r, recs in sorted(by_r.items())
        }
        for st, by_r in ctrl.msa_dynamics.items()
    }
    pf_dyn_summary = {
        st: {
            str(r): _aggregate_records(recs, ("rel_l1_z_pp", "rel_l1_z_pl", "rel_l1_s_p", "rel_l1_s_l", "cos_z_pp", "cos_z_pl"))
            for r, recs in sorted(by_r.items())
        }
        for st, by_r in ctrl.pf_dynamics.items()
    }
    cross_lig_summary = {k: summarize_cross_ligand(v) for k, v in ctrl.cross_ligand.items()}
    print("[CROSS_LIGAND]", json.dumps(cross_lig_summary, indent=2), flush=True)

    print("\n=== PART 4: MSA ALGORITHMIC PROBES ===", flush=True)
    probes = {}
    for probe_name in ("msa_r0_only", "msa_top512", "msa_r0_top512"):
        ctrl.msa_probe = probe_name
        rec = runner.run_once(probe_name, base_cfg, batch_dir, args.out_dir / probe_name, seed=0, repeat=0)
        ctrl.msa_probe = None
        probes[probe_name] = evaluate_probe(probe_name, rec, base_rec)
        p = probes[probe_name]
        print(
            f"[PROBE] {probe_name:14s}: steady={p['steady_s_mol']:.2f}s/mol ({p['speedup_vs_baseline']:.2f}x) "
            f"struct={p['struct_fwd_s_mol']:.2f}s aff={p['aff_fwd_s_mol']:.2f}s "
            f"val_rho={p['affinity_pred_value']['spearman_rho']:.4f} "
            f"prob_rho={p['affinity_probability_binary']['spearman_rho']:.4f} "
            f"prob_mae={p['affinity_probability_binary']['mean_abs_diff']:.4f} "
            f"ca_rmsd={p['median_ca_rmsd_A']:.3f}A lig_rmsd={p['median_ligand_rmsd_A']:.3f}A",
            flush=True,
        )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "ligands": [y.stem for y in yamls],
        "token_counts": dict(ctrl.token_counts),
        "checkpoint_diff": ckpt_diff,
        "msa_dynamics": msa_dyn_summary,
        "pairformer_dynamics": pf_dyn_summary,
        "pairformer_layer_dynamics": dict(ctrl.layer_dynamics),
        "cross_ligand_similarity": cross_lig_summary,
        "baseline": {
            "steady_s_mol": base_rec["gpu_seconds_per_molecule"],
            "struct_fwd_s_mol": base_rec["stages"]["structure"]["model_forward_seconds"] / len(yamls),
            "aff_fwd_s_mol": base_rec["stages"]["affinity"]["model_forward_seconds"] / len(yamls),
            "peak_allocated_gib": max(
                base_rec["stages"]["structure"]["peak_allocated_gib"],
                base_rec["stages"]["affinity"]["peak_allocated_gib"],
            ),
        },
        "msa_probes": probes,
    }
    out_json = args.out_dir / "trunk_diag.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print("[DONE] Wrote", out_json, flush=True)


if __name__ == "__main__":
    main_cli()
