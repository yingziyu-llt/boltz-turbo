import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

REPO_ROOT = Path(__file__).resolve().parent.parent


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = values[order]
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ra = rankdata(np.asarray(a, dtype=float))
    rb = rankdata(np.asarray(b, dtype=float))
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def kabsch_align(moving: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m_center = moving[mask].mean(axis=0)
    r_center = reference[mask].mean(axis=0)
    cov = (moving[mask] - m_center).T @ (reference[mask] - r_center)
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return (moving - m_center) @ rot + r_center


def load_cif_atoms(path: str):
    parser = MMCIFParser(QUIET=True)
    atoms = list(parser.get_structure("s", path).get_atoms())
    coords = np.array([a.coord for a in atoms], dtype=float)
    keys = [(a.get_parent().get_parent().id, a.get_parent().id, a.name) for a in atoms]
    is_protein = np.array([a.get_parent().id[0] == " " for a in atoms])
    is_ca = is_protein & np.array([a.name == "CA" for a in atoms])
    is_ligand = ~is_protein
    return coords, keys, is_ca, is_ligand


def geometry_vs_reference(cand_cifs: dict[str, str], ref_cifs: dict[str, str]) -> dict:
    ca_rmsds = []
    lig_rmsds = []
    for name, ref_path in ref_cifs.items():
        cand_path = cand_cifs.get(name)
        if not cand_path or not Path(cand_path).is_file() or not Path(ref_path).is_file():
            continue
        r_xyz, r_keys, r_ca, r_lig = load_cif_atoms(ref_path)
        c_xyz, c_keys, _, _ = load_cif_atoms(cand_path)
        if r_keys != c_keys:
            continue
        aligned = kabsch_align(c_xyz, r_xyz, r_ca)
        diff_sq = np.sum((aligned - r_xyz) ** 2, axis=-1)
        ca_rmsds.append(float(np.sqrt(np.mean(diff_sq[r_ca]))))
        if np.any(r_lig):
            lig_rmsds.append(float(np.sqrt(np.mean(diff_sq[r_lig]))))
    return {
        "evaluated_ligands": len(ca_rmsds),
        "median_protein_ca_rmsd_A": float(np.median(ca_rmsds)) if ca_rmsds else None,
        "median_ligand_rmsd_A": float(np.median(lig_rmsds)) if lig_rmsds else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze persistent-loading fair comparison results.")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "results/resident_compare")
    parser.add_argument("--reference", type=str, default="stock_200")
    args = parser.parse_args()

    out_dir = args.dir
    summary_path = out_dir / "summary.json"
    jsonl_path = out_dir / "runs.jsonl"

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        runs = summary["runs"]
        metadata = summary.get("metadata", {})
        cold_loads = summary.get("cold_loads", [])
        verification = summary.get("cache_verification", {})
    else:
        assert jsonl_path.is_file(), f"Neither {summary_path} nor {jsonl_path} found."
        runs = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        metadata, cold_loads, verification = {}, [], {}

    by_label = defaultdict(list)
    order = []
    for row in runs:
        lbl = row["label"]
        if lbl not in by_label:
            order.append(lbl)
        by_label[lbl].append(row)

    assert args.reference in by_label, f"Reference '{args.reference}' not in {order}"
    ref_runs = by_label[args.reference]
    ref_steady = float(np.median([r["steady_state_wall_seconds"] for r in ref_runs]))
    ref_diff = float(np.median([r["total_diffusion_seconds"] for r in ref_runs]))
    ref_rep0 = min(ref_runs, key=lambda r: r["repeat"])
    ligands = ref_rep0["ligand_names"]
    ref_probs = [ref_rep0["scores"][k]["affinity_probability_binary"] for k in ligands]
    ref_vals = [ref_rep0["scores"][k]["affinity_pred_value"] for k in ligands]

    rows = []
    for lbl in order:
        group = sorted(by_label[lbl], key=lambda r: r["repeat"])
        steady_list = [r["steady_state_wall_seconds"] for r in group]
        s_per_mol_list = [r["gpu_seconds_per_molecule"] for r in group]
        diff_list = [r["total_diffusion_seconds"] for r in group]
        pre_list = [r["preprocessing_seconds"] for r in group]
        struct_tr = [r["stages"]["structure"]["trainer_wall_seconds"] for r in group]
        struct_diff = [r["stages"]["structure"]["diffusion_sample_seconds"] for r in group]
        aff_tr = [r["stages"]["affinity"]["trainer_wall_seconds"] for r in group]
        aff_diff = [r["stages"]["affinity"]["diffusion_sample_seconds"] for r in group]
        peak_alloc = max(
            max(r["stages"]["structure"]["peak_allocated_gib"], r["stages"]["affinity"]["peak_allocated_gib"])
            for r in group
        )

        med_steady = float(np.median(steady_list))
        med_s_per_mol = float(np.median(s_per_mol_list))
        med_diff = float(np.median(diff_list))

        rep0 = group[0]
        probs = [rep0["scores"][k]["affinity_probability_binary"] for k in ligands]
        vals = [rep0["scores"][k]["affinity_pred_value"] for k in ligands]
        prob_diffs = [abs(a - b) for a, b in zip(probs, ref_probs)]
        val_diffs = [abs(a - b) for a, b in zip(vals, ref_vals)]

        repeat_prob_drift = 0.0
        if len(group) >= 2:
            for r_other in group[1:]:
                for k in ligands:
                    repeat_prob_drift = max(
                        repeat_prob_drift,
                        abs(r_other["scores"][k]["affinity_probability_binary"] - rep0["scores"][k]["affinity_probability_binary"]),
                    )

        geom = (
            {"evaluated_ligands": len(ligands), "median_protein_ca_rmsd_A": 0.0, "median_ligand_rmsd_A": 0.0}
            if lbl == args.reference
            else geometry_vs_reference(rep0["cif_paths"], ref_rep0["cif_paths"])
        )

        rows.append({
            "config": lbl,
            "repeats": len(group),
            "molecules": rep0["molecules"],
            "median_steady_seconds": med_steady,
            "steady_seconds_per_repeat": steady_list,
            "median_gpu_seconds_per_molecule": med_s_per_mol,
            "median_molecules_per_second": float(rep0["molecules"] / med_steady),
            "steady_speedup_vs_ref": float(ref_steady / med_steady),
            "median_diffusion_seconds": med_diff,
            "diffusion_speedup_vs_ref": float(ref_diff / med_diff),
            "diffusion_share": float(med_diff / med_steady),
            "non_diffusion_share": float(1.0 - med_diff / med_steady),
            "median_preprocessing_seconds": float(np.median(pre_list)),
            "median_structure_trainer_seconds": float(np.median(struct_tr)),
            "median_structure_diffusion_seconds": float(np.median(struct_diff)),
            "median_affinity_trainer_seconds": float(np.median(aff_tr)),
            "median_affinity_diffusion_seconds": float(np.median(aff_diff)),
            "peak_allocated_gib": float(peak_alloc),
            "repeat_max_abs_prob_diff": float(repeat_prob_drift),
            "vs_ref_prob_mae": float(np.mean(prob_diffs)),
            "vs_ref_prob_max_abs_diff": float(np.max(prob_diffs)),
            "vs_ref_pred_value_mae": float(np.mean(val_diffs)),
            "vs_ref_prob_spearman": spearman_corr(probs, ref_probs),
            "geometry_vs_ref": geom,
        })

    comparison = {
        "metadata": metadata,
        "reference": args.reference,
        "cold_loads": cold_loads,
        "cache_verification": verification,
        "summary_rows": rows,
        "notes": [
            "Steady-state wall time excludes one-time checkpoint loading and includes preprocessing, structure, affinity, and CIF/JSON output.",
            "Repeats run in alternating forward/reverse order in the same resident process after warmup.",
            "Geometry RMSD is computed relative to the stock reference prediction (not experimental crystal structures), matching ligand atoms by name without symmetry correction.",
        ],
    }
    comp_path = out_dir / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))

    # Print clean Markdown table
    print("\n### Persistent-Loading Steady-State Comparison\n")
    if verification:
        print(
            f"- Cold checkpoint load: **{verification.get('cold_checkpoint_load_seconds', 0):.2f} s** | "
            f"Warm cache hit load: **{verification.get('warm_checkpoint_load_seconds', 0):.4f} s** | "
            f"Exact cache match: **{verification.get('exact_match')}**"
        )
    print(
        "\n| Config | Steady s/mol | Mol/s | Steady Speedup | Diff Speedup | Diff Share | "
        "Prob MAE | Spearman | Cα RMSD (Å) | Lig RMSD (Å) | Peak GiB |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        sp = f"{r['vs_ref_prob_spearman']:.3f}" if r["vs_ref_prob_spearman"] is not None else "—"
        ca = f"{r['geometry_vs_ref']['median_protein_ca_rmsd_A']:.3f}" if r["geometry_vs_ref"]["median_protein_ca_rmsd_A"] is not None else "—"
        lg = f"{r['geometry_vs_ref']['median_ligand_rmsd_A']:.3f}" if r["geometry_vs_ref"]["median_ligand_rmsd_A"] is not None else "—"
        print(
            f"| {r['config']} | {r['median_gpu_seconds_per_molecule']:.2f} | "
            f"{r['median_molecules_per_second']:.3f} | {r['steady_speedup_vs_ref']:.2f}× | "
            f"{r['diffusion_speedup_vs_ref']:.2f}× | {100*r['diffusion_share']:.1f}% | "
            f"{r['vs_ref_prob_mae']:.4f} | {sp} | {ca} | {lg} | {r['peak_allocated_gib']:.2f} |"
        )
    print(f"\nSaved detailed report to {comp_path}")


if __name__ == "__main__":
    main()
