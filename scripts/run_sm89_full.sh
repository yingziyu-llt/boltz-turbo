#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
python "$ROOT/sm89/smoke_sm89_pair.py"
export SM89_ACTIVE=1 SM89_FULL=1
if [ "$#" -eq 0 ]; then set -- exact fast; fi
for mode in "$@"; do
    label=full
    if [ "$mode" = exact ]; then label=exact; fi
    /usr/bin/time -p python "$ROOT/experiments/exp_sm89.py" pred --mode "$mode" --n_gpu 1 --allow-partial \
      --input "$ROOT/data/smoke.yaml" --out_dir "$ROOT/results/anthropic/sm89_${label}_0" \
      --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
      --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
      --num_workers 2 --preprocessing-threads 2 --override \
      > "$ROOT/results/anthropic/sm89_${label}_0.log" 2>&1
done
