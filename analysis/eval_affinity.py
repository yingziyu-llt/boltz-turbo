import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

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
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denom) if denom > 0 else float("nan")


def bootstrap_ci(fn, x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Boltz-2 resident comparison runs against experimental FEP+ affinities.")
    parser.add_argument("--dir", type=Path, required=True, help="Result directory containing summary.json or runs.jsonl.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to data/fep/<target>/labels.json.")
    parser.add_argument("--reference", type=str, default="stock_200")
    args = parser.parse_args()

    out_dir = args.dir
    labels = json.loads(args.labels.read_text())
    label_by_name = {item["name"]: item for item in labels["ligands"]}

    summary_path = out_dir / "summary.json"
    jsonl_path = out_dir / "runs.jsonl"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        runs = summary["runs"]
        metadata = summary.get("metadata", {})
    else:
        assert jsonl_path.is_file(), f"Neither {summary_path} nor {jsonl_path} found."
        runs = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        metadata = {}

    by_label = defaultdict(list)
    order = []
    for r in runs:
        lbl = r["label"]
        if lbl not in by_label:
            order.append(lbl)
        by_label[lbl].append(r)

    assert args.reference in by_label, f"Reference '{args.reference}' not in {order}"
    ref_runs = by_label[args.reference]
    ref_steady = float(np.median([r["steady_state_wall_seconds"] for r in ref_runs]))
    ref_diff = float(np.median([r["total_diffusion_seconds"] for r in ref_runs]))
    ref_rep0 = min(ref_runs, key=lambda r: r["repeat"])

    ligands = sorted(k for k in ref_rep0["scores"] if k in label_by_name)
    assert ligands, "No overlapping ligands between predictions and labels.json"
    exp_dg = np.array([label_by_name[k]["exp_dg_kcal"] for k in ligands], dtype=float)
    exp_log10_uM = np.array([label_by_name[k]["exp_log10_uM"] for k in ligands], dtype=float)
    ref_pred_val = np.array([ref_rep0["scores"][k]["affinity_pred_value"] for k in ligands], dtype=float)
    ref_prob = np.array([ref_rep0["scores"][k]["affinity_probability_binary"] for k in ligands], dtype=float)

    top_k = max(3, len(ligands) // 4)
    exp_top_idx = set(np.argsort(exp_dg)[:top_k].tolist())

    rows = []
    for lbl in order:
        group = sorted(by_label[lbl], key=lambda r: r["repeat"])
        med_steady = float(np.median([r["steady_state_wall_seconds"] for r in group]))
        med_s_mol = float(np.median([r["gpu_seconds_per_molecule"] for r in group]))
        med_diff = float(np.median([r["total_diffusion_seconds"] for r in group]))

        # Average across repeats if multiple repeats exist; also report repeat 0
        pred_val = np.mean([[r["scores"][k]["affinity_pred_value"] for k in ligands] for r in group], axis=0)
        pred_prob = np.mean([[r["scores"][k]["affinity_probability_binary"] for k in ligands] for r in group], axis=0)
        # Boltz-2 official conversion: pIC50 = 6 - affinity_pred_value -> dG = -1.364 * pIC50 = 1.364 * val - 8.184
        pred_dg = 1.364 * pred_val - 8.184

        r_val = pearson(pred_dg, exp_dg)
        rho_val = spearman(pred_dg, exp_dg)
        tau_val = kendall_tau_b(pred_dg, exp_dg)
        rmse_raw = float(np.sqrt(np.mean((pred_dg - exp_dg) ** 2)))
        rmse_centered = float(np.sqrt(np.mean(((pred_dg - pred_dg.mean()) - (exp_dg - exp_dg.mean())) ** 2)))
        mae_centered = float(np.mean(np.abs((pred_dg - pred_dg.mean()) - (exp_dg - exp_dg.mean()))))
        pred_top_idx = set(np.argsort(pred_val)[:top_k].tolist())
        top_k_recall = len(exp_top_idx & pred_top_idx) / top_k

        rows.append({
            "config": lbl,
            "repeats": len(group),
            "ligands": len(ligands),
            "median_gpu_seconds_per_molecule": med_s_mol,
            "median_molecules_per_second": float(len(ligands) / med_steady),
            "steady_speedup_vs_ref": float(ref_steady / med_steady),
            "diffusion_speedup_vs_ref": float(ref_diff / med_diff),
            "diffusion_share": float(med_diff / med_steady),
            "pearson_r_vs_exp": r_val,
            "pearson_ci95": bootstrap_ci(pearson, pred_dg, exp_dg),
            "spearman_rho_vs_exp": rho_val,
            "spearman_ci95": bootstrap_ci(spearman, pred_dg, exp_dg),
            "kendall_tau_vs_exp": tau_val,
            "dg_rmse_raw_kcal": rmse_raw,
            "dg_rmse_centered_kcal": rmse_centered,
            "dg_mae_centered_kcal": mae_centered,
            f"top{top_k}_strongest_recall": float(top_k_recall),
            "median_binder_prob": float(np.median(pred_prob)),
            "vs_ref_pred_value_mae": float(np.mean(np.abs(pred_val - ref_pred_val))),
            "vs_ref_pred_value_spearman": spearman(pred_val, ref_pred_val),
            "vs_ref_prob_mae": float(np.mean(np.abs(pred_prob - ref_prob))),
        })

    result = {
        "target": labels["target"],
        "pdb": labels["pdb"],
        "sequence_length": labels["sequence_length"],
        "msa": labels["msa"],
        "num_ligands": len(ligands),
        "exp_dg_kcal_range": labels["exp_dg_kcal_range"],
        "top_k": top_k,
        "metadata": metadata,
        "rows": rows,
        "notes": [
            "Boltz-2 affinity_pred_value is log10(IC50/Ki in uM), converted via pred_dg = 1.364 * affinity_pred_value - 8.184 kcal/mol.",
            "Lower affinity_pred_value / lower exp_dg_kcal indicates stronger binding; positive Pearson/Spearman/Kendall indicates correct experimental ranking.",
        ],
    }
    eval_path = out_dir / "affinity_eval.json"
    eval_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n### Experimental Affinity Evaluation: {labels['target'].upper()} (PDB {labels['pdb']}, n={len(ligands)}, msa={labels['msa']})\n")
    print(
        f"| Config | Steady s/mol | Speedup | Diff Share | Pearson r | Spearman ρ | Kendall τ | "
        f"cRMSE (kcal/mol) | Top-{top_k} Recall | Med Prob | vs Stock ρ |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['config']} | {r['median_gpu_seconds_per_molecule']:.2f} | "
            f"{r['steady_speedup_vs_ref']:.2f}× | {100*r['diffusion_share']:.1f}% | "
            f"{r['pearson_r_vs_exp']:.3f} | {r['spearman_rho_vs_exp']:.3f} | {r['kendall_tau_vs_exp']:.3f} | "
            f"{r['dg_rmse_centered_kcal']:.2f} | {r[f'top{top_k}_strongest_recall']:.2f} | "
            f"{r['median_binder_prob']:.3f} | {r['vs_ref_pred_value_spearman']:.3f} |"
        )
    print(f"\nSaved affinity evaluation to {eval_path}")


if __name__ == "__main__":
    main()
