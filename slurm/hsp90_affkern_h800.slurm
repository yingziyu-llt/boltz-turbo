#!/usr/bin/env bash
#SBATCH --job-name=boltz2-hsp90-affkern-h800
#SBATCH --partition=gpu_h800
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlaih800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:35:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/hsp90-affkern-h800-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE="${BOLTZ_CACHE:-/home/lhlai_pkuhpc/.boltz}"
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
printf 'JOB_ID=%s NODE=%s\n' "$SLURM_JOB_ID" "$HOSTNAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

OUT_DIR="results/hsp90_h800_affkern_${SLURM_JOB_ID}"
python experiments/exp_resident_compare.py \
  --input data/hsp90_strat60 \
  --configs dpm2m_16,dpm2m_16_affkern \
  --repeats 1 \
  --no-verify-cache \
  --out-dir "$OUT_DIR"

python analysis/analyze_resident_compare.py --dir "$OUT_DIR" --reference dpm2m_16
python analysis/eval_hsp90.py --dir "$OUT_DIR" --reference dpm2m_16

python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1])
summary = json.loads((out_dir / "summary.json").read_text())
runs = {r["label"]: r for r in summary["runs"]}
a = runs["dpm2m_16"]
b = runs["dpm2m_16_affkern"]
ligs = sorted(a["scores"])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

for label, r in (("dpm2m_16", a), ("dpm2m_16_affkern", b)):
    n = r["molecules"]
    s_st = r["stages"]["structure"]
    a_st = r["stages"]["affinity"]
    print(
        f"[STAGE] {label}: steady={r['gpu_seconds_per_molecule']:.3f}s/mol "
        f"struct_fwd={s_st['model_forward_seconds']/n:.3f}s(kern={s_st['use_kernels']}) "
        f"aff_fwd={a_st['model_forward_seconds']/n:.3f}s(kern={a_st['use_kernels']}) "
        f"peak_gib={max(s_st['peak_allocated_gib'], a_st['peak_allocated_gib']):.2f}"
    )

for metric in ("affinity_pred_value", "affinity_probability_binary"):
    va = np.array([a["scores"][k][metric] for k in ligs], dtype=float)
    vb = np.array([b["scores"][k][metric] for k in ligs], dtype=float)
    diff = np.abs(va - vb)
    print(
        f"[DIFF] {metric}: n={len(ligs)} max_abs={diff.max():.6f} "
        f"mean_abs={diff.mean():.6f} pearson_r={np.corrcoef(va, vb)[0,1]:.6f} "
        f"spearman_rho={spearman(va, vb):.6f}"
    )
PY
