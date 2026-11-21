#!/usr/bin/env bash
#SBATCH --job-name=boltz2-resident-cmp
#SBATCH --partition=gpu_l40
#SBATCH --nodelist=s60b05n02
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/resident-cmp-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE="${BOLTZ_CACHE:-/home/lhlai_pkuhpc/.boltz}"
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SUITE="${1:-fair}"
OUT_DIR="results/resident_compare_${SUITE}"
python experiments/exp_resident_compare.py --suite "$SUITE" --repeats 2 --out-dir "$OUT_DIR"
python analysis/analyze_resident_compare.py --dir "$OUT_DIR"
