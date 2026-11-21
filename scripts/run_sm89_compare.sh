#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
python "$ROOT/sm89/install_sm89.py"
for repeat in 0 1; do
    for variant in baseline opm; do
        active=0
        entry="$ROOT/experiments/exp_anthropic_l40.py"
        if [ "$variant" = opm ]; then active=1; entry="$ROOT/experiments/exp_sm89.py"; fi
        set +e
        SM89_ACTIVE="$active" /usr/bin/time -p python "$entry" pred --mode fast --n_gpu 1 --allow-partial \
          --input "$ROOT/data/smoke.yaml" --out_dir "$ROOT/results/anthropic/sm89_${variant}_${repeat}" \
          --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
          --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
          --num_workers 2 --preprocessing-threads 2 --override \
          > "$ROOT/results/anthropic/sm89_${variant}_${repeat}.log" 2>&1
        rc=$?
        set -e
        printf 'PRED variant=%s repeat=%s rc=%s\n' "$variant" "$repeat" "$rc"
        if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    done
done
