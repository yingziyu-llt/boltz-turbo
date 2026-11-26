#!/usr/bin/env bash
#SBATCH --job-name=boltz2-hsp90-final-h800
#SBATCH --partition=gpu_h800
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlaih800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:55:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/hsp90-final-h800-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE="${BOLTZ_CACHE:-/home/lhlai_pkuhpc/.boltz}"
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
printf 'JOB_ID=%s NODE=%s\n' "$SLURM_JOB_ID" "$HOSTNAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

OUT_DIR="results/hsp90_h800_final_${SLURM_JOB_ID}"
python experiments/exp_resident_compare.py \
  --input data/hsp90_strat60 \
  --configs stock_200,dpm2m_16_affkern,dpm2m_16_affkern_tc05 \
  --repeats 1 \
  --no-verify-cache \
  --out-dir "$OUT_DIR"

python analysis/analyze_resident_compare.py --dir "$OUT_DIR" --reference stock_200
python analysis/eval_hsp90.py --dir "$OUT_DIR" --reference stock_200

python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from analysis.analyze_resident_compare import kabsch_align, load_cif_atoms

out_dir = Path(sys.argv[1])
summary = json.loads((out_dir / "summary.json").read_text())
runs = {r["label"]: r for r in summary["runs"]}
ref = runs["stock_200"]
ak = runs["dpm2m_16_affkern"]
tc = runs["dpm2m_16_affkern_tc05"]
ligs = sorted(ref["scores"])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

for label, r in (("stock_200", ref), ("dpm2m_16_affkern", ak), ("dpm2m_16_affkern_tc05", tc)):
    n = r["molecules"]
    s_st = r["stages"]["structure"]
    a_st = r["stages"]["affinity"]
    print(
        f"[STAGE] {label}: steady={r['gpu_seconds_per_molecule']:.3f}s/mol "
        f"speedup_vs_stock={ref['gpu_seconds_per_molecule']/r['gpu_seconds_per_molecule']:.2f}x "
        f"struct_fwd={s_st['model_forward_seconds']/n:.3f}s aff_fwd={a_st['model_forward_seconds']/n:.3f}s "
        f"mean_msa_aff={a_st.get('mean_msa_recycles')} mean_pf_aff={a_st.get('mean_pf_recycles')} "
        f"all_cache_hits={r['all_cache_hits']}"
    )

for metric in ("affinity_pred_value", "affinity_probability_binary"):
    va = np.array([ak["scores"][k][metric] for k in ligs], dtype=float)
    vb = np.array([tc["scores"][k][metric] for k in ligs], dtype=float)
    diff = np.abs(va - vb)
    print(
        f"[TC05_VS_AFFKERN] {metric}: n={len(ligs)} max_abs={diff.max():.6f} "
        f"mean_abs={diff.mean():.6f} pearson_r={np.corrcoef(va, vb)[0,1]:.6f} "
        f"spearman_rho={spearman(va, vb):.6f}"
    )

ca_rmsds, lig_rmsds = [], []
for k in ligs:
    c_a, k_a, ca_a, lg_a = load_cif_atoms(ak["cif_paths"][k])
    c_b, k_b, ca_b, lg_b = load_cif_atoms(tc["cif_paths"][k])
    if k_a == k_b:
        aligned = kabsch_align(c_b, c_a, ca_a)
        ca_rmsds.append(float(np.sqrt(np.mean(np.sum((aligned[ca_a] - c_a[ca_a]) ** 2, axis=-1)))))
        lig_rmsds.append(float(np.sqrt(np.mean(np.sum((aligned[lg_a] - c_a[lg_a]) ** 2, axis=-1)))))
print(f"[TC05_VS_AFFKERN_RMSD] median_ca={np.median(ca_rmsds):.6f}A median_lig={np.median(lig_rmsds):.6f}A")
PY
