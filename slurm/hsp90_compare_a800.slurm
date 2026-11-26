#!/usr/bin/env bash
#SBATCH --job-name=boltz2-hsp90-a800
#SBATCH --partition=gpu_a800
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlaia800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/hsp90-a800-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE="${BOLTZ_CACHE:-/home/lhlai_pkuhpc/.boltz}"
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
printf 'JOB_ID=%s NODE=%s\n' "$SLURM_JOB_ID" "$HOSTNAME"
nvidia-smi --query-gpu=name,memory.total,driver_version,ecc.errors.uncorrected.volatile.total --format=csv

SUITE="${1:-fair}"
INPUT_DIR="${2:-/lustre3/lhlai_pkuhpc/llt/hsp90_affinity/inputs}"
OUT_DIR="results/hsp90_a800_${SUITE}"
python experiments/exp_resident_compare.py \
  --input "$INPUT_DIR" \
  --suite "$SUITE" \
  --repeats 1 \
  --no-verify-cache \
  --out-dir "$OUT_DIR"
python analysis/analyze_resident_compare.py --dir "$OUT_DIR"
python analysis/eval_hsp90.py --dir "$OUT_DIR"
