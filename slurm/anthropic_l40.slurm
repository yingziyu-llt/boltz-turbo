#!/usr/bin/env bash
#SBATCH --job-name=boltz2-anthropic
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/anthropic-%j.log
set -euo pipefail
ROOT="$PWD"
export PATH="$ROOT/anthropic-env/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export MODEL_OPT_JIT_ROOT="$ROOT/cache/anthropic"
export TRITON_CACHE_DIR="$ROOT/cache/anthropic-triton"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache/anthropic-xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/anthropic-extensions"
export CUDA_CACHE_PATH="$ROOT/cache/anthropic-cuda"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
export LD_LIBRARY_PATH="$ROOT/anthropic-env/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/results/anthropic"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability())'
for mode in exact fast big; do
    set +e
    bash "$ROOT/anthropic-kit/boltz2/run.sh" check --mode "$mode" > "$ROOT/results/anthropic/check_$mode.log" 2>&1
    rc=$?
    set -e
    printf 'CHECK mode=%s rc=%s\n' "$mode" "$rc"
done
for mode in off exact fast big; do
    for repeat in 0 1; do
        set +e
        /usr/bin/time -p bash "$ROOT/anthropic-kit/boltz2/run.sh" pred --mode "$mode" \
          --input "$ROOT/data/smoke.yaml" --out_dir "$ROOT/results/anthropic/${mode}_${repeat}" \
          --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
          --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
          --num_workers 2 --preprocessing-threads 2 --override \
          > "$ROOT/results/anthropic/${mode}_${repeat}.log" 2>&1
        rc=$?
        set -e
        printf 'PRED mode=%s repeat=%s rc=%s\n' "$mode" "$repeat" "$rc"
    done
done
set +e
/usr/bin/time -p bash "$ROOT/anthropic-kit/boltz2/run.sh" pred --mode big --n_gpu 2 \
  --input "$ROOT/data/smoke.yaml" --out_dir "$ROOT/results/anthropic/big_x2" \
  --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
  --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
  --num_workers 2 --preprocessing-threads 2 --override \
  > "$ROOT/results/anthropic/big_x2.log" 2>&1
rc=$?
set -e
printf 'PRED mode=big_x2 rc=%s\n' "$rc"
