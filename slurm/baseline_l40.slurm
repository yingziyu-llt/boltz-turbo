#!/usr/bin/env bash
#SBATCH --job-name=boltz2-baseline
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/baseline-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export CUDA_CACHE_PATH="$PWD/cache/cuda"
export TMPDIR="$PWD/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python experiments/exp_baseline.py
python experiments/exp_baseline.py --profile
