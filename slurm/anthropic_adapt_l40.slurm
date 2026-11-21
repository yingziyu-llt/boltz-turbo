#!/usr/bin/env bash
#SBATCH --job-name=boltz2-kit-l40
#SBATCH --partition=gpu_l40
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/anthropic-adapt-%j.log
set -euo pipefail
ROOT="$PWD"
export PATH="$ROOT/anthropic-env/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export MODEL_OPT_JIT_ROOT="$ROOT/cache/anthropic" TRITON_CACHE_DIR="$ROOT/cache/anthropic-triton"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache/anthropic-xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/anthropic-extensions" CUDA_CACHE_PATH="$ROOT/cache/anthropic-cuda"
export LD_LIBRARY_PATH="$ROOT/anthropic-env/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
for mode in exact fast big big_x2; do
    ngpu=1
    actual="$mode"
    if [ "$mode" = big_x2 ]; then ngpu=2; actual=big; fi
    for repeat in 0 1; do
        set +e
        /usr/bin/time -p python experiments/exp_anthropic_l40.py pred --mode "$actual" --n_gpu "$ngpu" --allow-partial \
          --input "$ROOT/data/smoke.yaml" --out_dir "$ROOT/results/anthropic/adapt_${mode}_${repeat}" \
          --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
          --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
          --num_workers 2 --preprocessing-threads 2 --override \
          > "$ROOT/results/anthropic/adapt_${mode}_${repeat}.log" 2>&1
        rc=$?
        set -e
        printf 'PRED mode=%s repeat=%s rc=%s\n' "$mode" "$repeat" "$rc"
    done
done
