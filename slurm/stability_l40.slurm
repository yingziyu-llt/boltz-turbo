#!/usr/bin/env bash
#SBATCH --job-name=boltz2-stability
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/stability-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for seed in 0 1 2 3 4; do
    for system in smoke 3PTB 1IEP 1STP; do
        python experiments/exp_training_free.py --input "data/stability/$system.yaml" --single-pass --method stock --steps 200 --schedule karras --seed "$seed"
        python experiments/exp_training_free.py --input "data/stability/$system.yaml" --single-pass --method dpm2m --steps 16 --seed "$seed"
        python experiments/exp_training_free.py --input "data/stability/$system.yaml" --single-pass --method dpm2m --steps 32 --seed "$seed"
    done
done
python analysis/analyze_stability.py
