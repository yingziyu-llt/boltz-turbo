#!/usr/bin/env bash
#SBATCH --job-name=boltz2-sm89
#SBATCH --partition=gpu_l40
#SBATCH --nodelist=s60b05n02
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlail40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/sm89-%j.log
set -euo pipefail
ROOT="$PWD"
export PATH="$ROOT/anthropic-env/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BOLTZ_CACHE=/home/lhlai_pkuhpc/.boltz
export MODEL_OPT_JIT_ROOT="$ROOT/cache/anthropic" TRITON_CACHE_DIR="$ROOT/cache/anthropic-triton"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache/anthropic-xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/anthropic-extensions" CUDA_CACHE_PATH="$ROOT/cache/anthropic-cuda"
export LD_LIBRARY_PATH="$ROOT/anthropic-env/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$MODEL_OPT_JIT_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$ROOT/results/sm89"
entry="${1:-sm89/smoke_sm89_opm.py}"
if [ "$#" -gt 0 ]; then shift; fi
case "$entry" in
    *.sh) bash "$entry" "$@" ;;
    *) python "$entry" "$@" ;;
esac
