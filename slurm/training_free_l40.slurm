#!/usr/bin/env bash
#SBATCH --job-name=boltz2-training-free
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/training-free-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export XDG_CACHE_HOME="$PWD/cache" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ "${1:-all}" = all ]; then
    python experiments/exp_training_free.py --method stock --steps 200 --schedule karras
    python experiments/exp_training_free.py --method stock --steps 100 --schedule karras
    python experiments/exp_training_free.py --method stock --steps 50 --schedule karras
    python experiments/exp_training_free.py --method geometric --steps 50
fi
python experiments/exp_training_free.py --method euler --steps 200 --schedule karras
python experiments/exp_training_free.py --method euler --steps 64 --schedule karras
python experiments/exp_training_free.py --method euler --steps 64
python experiments/exp_training_free.py --method heun --steps 32
python experiments/exp_training_free.py --method dpm2m --steps 64
python experiments/exp_training_free.py --method euler --steps 32
python experiments/exp_training_free.py --method heun --steps 16
python experiments/exp_training_free.py --method dpm2m --steps 32
python analysis/analyze_training_free.py
