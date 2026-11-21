import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "results/training_free"
reports = []
for path in sorted(root.glob("dpm2m_*/summary.json")):
    report = json.loads(path.read_text())
    if "dpm_parameters" not in report:
        continue
    row = {"configuration": path.parent.name, "seed": report["seed"],
           "parameters": report["dpm_parameters"], "scores": report["scores"],
           "warm_sample_seconds": {s["stage"]: s["warm_sample_seconds"] for s in report["stages"]},
           "studies": {s["stage"]: s["dpm_study"] for s in report["stages"] if "dpm_study" in s}}
    reports.append(row)
result = {"reports": reports, "reference": "Heun512 under identical conditioning and restored initial RNG",
          "limitations": ["All ODE variants use fixed frame, no churn, no reverse alignment, no guidance and unit step scale.",
                          "Same-frame coordinate RMS is per coordinate, not per-atom Euclidean RMSD.",
                          "Heun256 versus Heun512 measures reference sensitivity, not an exact-solution guarantee.",
                          "Ligand pair-distance error is a numerical geometry diagnostic, not physical validity.",
                          "Full predictions change both stages; their score effects are not independently attributed.",
                          "Unlabeled smoke input; no screening-quality conclusion."]}
(root / "dpm_comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
