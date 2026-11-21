#!/usr/bin/env bash
#SBATCH --job-name=boltz2-dpm
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/dpm-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python experiments/exp_training_free.py --method dpm2m --steps 32 --ligand-history-weight 0.5 --dpm-study
python experiments/exp_training_free.py --method dpm2m --steps 32 --seed 1 --dpm-study
python experiments/exp_training_free.py --method dpm2m --steps 32 --seed 1 --ligand-history-weight 0.5
python analysis/analyze_dpm.py
