import argparse
import csv
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
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    hits = np.cumsum(y == 1)
    prec = hits / np.arange(1, len(y) + 1)
    return float(prec[y == 1].sum() / n_pos)


def enrichment_and_recall(scores: np.ndarray, labels: np.ndarray, frac: float) -> tuple[float, float]:
    n = len(labels)
    n_pos = int((labels == 1).sum())
    k = max(1, int(round(n * frac)))
    if n_pos == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_hits = int((labels[order[:k]] == 1).sum())
    ef = (top_hits / k) / (n_pos / n)
    rec = top_hits / n_pos
    return float(ef), float(rec)


def _opt_float(value: str) -> float:
    text = value.strip()
    return float(text) if text else float("nan")


def load_hsp90_labels(csv_path: Path) -> dict[str, dict]:
    rows = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["ligand_id"]] = {
                "ligand_id": row["ligand_id"],
                "chemotype": row["chemotype"],
                "ic50_uM": float(row["ic50_uM"]),
                "relation": row["relation"].strip(),
                "pIC50_M": _opt_float(row["pIC50_M"]),
                "boltz_log10_IC50_uM": _opt_float(row["boltz_log10_IC50_uM"]),
                "label_group": row["label_group"].strip(),
            }
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Boltz-2 runs on HSP90 experimental affinity & screening dataset.")
    parser.add_argument("--dir", type=Path, required=True, help="Result dir containing summary.json or runs.jsonl.")
    parser.add_argument("--labels-csv", type=Path,
                        default=Path("/lustre3/lhlai_pkuhpc/llt/hsp90_affinity/hsp90_labels_primary.csv"))
    parser.add_argument("--reference", type=str, default="stock_200")
    args = parser.parse_args()

    labels = load_hsp90_labels(args.labels_csv)
    summary_path = args.dir / "summary.json"
    jsonl_path = args.dir / "runs.jsonl"
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

    ligs = sorted(k for k in ref_rep0["scores"] if k in labels)
    assert ligs, "No overlapping ligand_id between predictions and HSP90 CSV"

    quant_idx = np.array(
        [i for i, k in enumerate(ligs)
         if labels[k]["relation"] == "=" and np.isfinite(labels[k]["boltz_log10_IC50_uM"])],
        dtype=int,
    )
    exp_log10 = np.array([labels[k]["boltz_log10_IC50_uM"] for k in ligs], dtype=float)
    exp_dg = 1.364 * exp_log10 - 8.184
    y_1uM = np.array([1 if (labels[k]["relation"] == "=" and labels[k]["ic50_uM"] <= 1.0) else 0 for k in ligs])
    y_100nM = np.array([1 if (labels[k]["relation"] == "=" and labels[k]["ic50_uM"] <= 0.1) else 0 for k in ligs])

    ref_val = np.array([ref_rep0["scores"][k]["affinity_pred_value"] for k in ligs], dtype=float)
    ref_prob = np.array([ref_rep0["scores"][k]["affinity_probability_binary"] for k in ligs], dtype=float)

    chemotypes = sorted({labels[k]["chemotype"] for k in ligs})
    rows = []
    for lbl in order:
        group = sorted(by_label[lbl], key=lambda r: r["repeat"])
        med_steady = float(np.median([r["steady_state_wall_seconds"] for r in group]))
        med_s_mol = float(np.median([r["gpu_seconds_per_molecule"] for r in group]))
        med_diff = float(np.median([r["total_diffusion_seconds"] for r in group]))

        pred_val = np.mean([[r["scores"][k]["affinity_pred_value"] for k in ligs] for r in group], axis=0)
        pred_prob = np.mean([[r["scores"][k]["affinity_probability_binary"] for k in ligs] for r in group], axis=0)
        pred_dg = 1.364 * pred_val - 8.184

        # Quantitative regression metrics (147 quantitative ligands)
        pv_q, ev_q = pred_val[quant_idx], exp_log10[quant_idx]
        pdg_q, edg_q = pred_dg[quant_idx], exp_dg[quant_idx]
        r_q = pearson(pv_q, ev_q)
        rho_q = spearman(pv_q, ev_q)
        tau_q = kendall_tau_b(pv_q, ev_q)
        crmse_kcal = float(np.sqrt(np.mean(((pdg_q - pdg_q.mean()) - (edg_q - edg_q.mean())) ** 2)))
        raw_rmse_kcal = float(np.sqrt(np.mean((pdg_q - edg_q) ** 2)))

        # Screening metrics using BOTH affinity_probability_binary (higher=binder) AND -affinity_pred_value (higher=stronger)
        ef10_prob_1uM, rec10_prob_1uM = enrichment_and_recall(pred_prob, y_1uM, 0.10)
        ef10_val_1uM, rec10_val_1uM = enrichment_and_recall(-pred_val, y_1uM, 0.10)
        ef10_prob_100nM, rec10_prob_100nM = enrichment_and_recall(pred_prob, y_100nM, 0.10)
        ef10_val_100nM, rec10_val_100nM = enrichment_and_recall(-pred_val, y_100nM, 0.10)

        per_chemo = {}
        for ch in chemotypes:
            cidx = np.array([i for i in quant_idx if labels[ligs[i]]["chemotype"] == ch], dtype=int)
            if len(cidx) >= 3:
                per_chemo[ch] = {
                    "n_quant": int(len(cidx)),
                    "pearson_r": pearson(pred_val[cidx], exp_log10[cidx]),
                    "spearman_rho": spearman(pred_val[cidx], exp_log10[cidx]),
                }

        rows.append({
            "config": lbl,
            "repeats": len(group),
            "total_ligands": len(ligs),
            "quant_ligands": int(len(quant_idx)),
            "median_gpu_seconds_per_molecule": med_s_mol,
            "median_molecules_per_second": float(len(ligs) / med_steady),
            "steady_speedup_vs_ref": float(ref_steady / med_steady),
            "diffusion_speedup_vs_ref": float(ref_diff / med_diff),
            "diffusion_share": float(med_diff / med_steady),
            "quant_pearson_r": r_q,
            "quant_spearman_rho": rho_q,
            "quant_kendall_tau": tau_q,
            "quant_crmse_kcal": crmse_kcal,
            "quant_raw_rmse_kcal": raw_rmse_kcal,
            "screen_1uM_by_prob": {
                "actives": int(y_1uM.sum()),
                "auroc": auroc(pred_prob, y_1uM),
                "ap": average_precision(pred_prob, y_1uM),
                "ef10": ef10_prob_1uM,
                "recall10": rec10_prob_1uM,
            },
            "screen_1uM_by_neg_val": {
                "actives": int(y_1uM.sum()),
                "auroc": auroc(-pred_val, y_1uM),
                "ap": average_precision(-pred_val, y_1uM),
                "ef10": ef10_val_1uM,
                "recall10": rec10_val_1uM,
            },
            "screen_100nM_by_prob": {
                "actives": int(y_100nM.sum()),
                "auroc": auroc(pred_prob, y_100nM),
                "ap": average_precision(pred_prob, y_100nM),
                "ef10": ef10_prob_100nM,
                "recall10": rec10_prob_100nM,
            },
            "screen_100nM_by_neg_val": {
                "actives": int(y_100nM.sum()),
                "auroc": auroc(-pred_val, y_100nM),
                "ap": average_precision(-pred_val, y_100nM),
                "ef10": ef10_val_100nM,
                "recall10": rec10_val_100nM,
            },
            "vs_ref_val_spearman": spearman(pred_val, ref_val),
            "vs_ref_val_mae": float(np.mean(np.abs(pred_val - ref_val))),
            "vs_ref_prob_spearman": spearman(pred_prob, ref_prob),
            "vs_ref_prob_mae": float(np.mean(np.abs(pred_prob - ref_prob))),
            "per_chemotype": per_chemo,
        })

    out = {
        "dataset": "HSP90 (4YKR) experimental IC50",
        "labels_csv": str(args.labels_csv),
        "total_ligands": len(ligs),
        "quant_ligands": int(len(quant_idx)),
        "actives_le_1uM": int(y_1uM.sum()),
        "actives_le_100nM": int(y_100nM.sum()),
        "metadata": metadata,
        "rows": rows,
    }
    eval_path = args.dir / "hsp90_eval.json"
    eval_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n### HSP90 Experimental Affinity & Screening Evaluation (n={len(ligs)}, quant={len(quant_idx)}, <=1uM={int(y_1uM.sum())}, <=100nM={int(y_100nM.sum())})\n")
    print("| Config | s/mol | Speedup | Diff% | Quant r | Quant ρ | Quant τ | cRMSE | AUROC(1µM prob/val) | AP(1µM prob/val) | EF10%(100nM prob/val) | vs Stock ρ(val/prob) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| {r['config']} | {r['median_gpu_seconds_per_molecule']:.2f} | {r['steady_speedup_vs_ref']:.2f}× | "
            f"{100*r['diffusion_share']:.1f}% | {r['quant_pearson_r']:.3f} | {r['quant_spearman_rho']:.3f} | "
            f"{r['quant_kendall_tau']:.3f} | {r['quant_crmse_kcal']:.2f} | "
            f"{r['screen_1uM_by_prob']['auroc']:.3f} / {r['screen_1uM_by_neg_val']['auroc']:.3f} | "
            f"{r['screen_1uM_by_prob']['ap']:.3f} / {r['screen_1uM_by_neg_val']['ap']:.3f} | "
            f"{r['screen_100nM_by_prob']['ef10']:.2f}× / {r['screen_100nM_by_neg_val']['ef10']:.2f}× | "
            f"{r['vs_ref_val_spearman']:.3f} / {r['vs_ref_prob_spearman']:.3f} |"
        )
    print(f"\nSaved evaluation to {eval_path}")


if __name__ == "__main__":
    main()
