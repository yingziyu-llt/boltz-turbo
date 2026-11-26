#!/usr/bin/env bash
#SBATCH --job-name=boltz2-hsp90-profile
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/hsp90-profile-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE="${BOLTZ_CACHE:-/home/lhlai_pkuhpc/.boltz}"
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
printf 'JOB_ID=%s NODE=%s\n' "$SLURM_JOB_ID" "$HOSTNAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "${1:-experiments}/exp_profile_hsp90.py" \
  --input /lustre3/lhlai_pkuhpc/llt/hsp90_affinity/inputs \
  --out-dir "results/hsp90_profile_${SLURM_JOB_ID}"
