#!/usr/bin/env bash
# Run from the project root on an allocated GPU node with Boltz 2.2.1 active.
set -euo pipefail
: "${BOLTZ_CACHE:?Set BOLTZ_CACHE to the shared Boltz-2 weights directory}"
mkdir -p results/stock_smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import sys, torch; from importlib.metadata import version; print("python", sys.version); print("boltz", version("boltz")); print("torch", torch.__version__); print("cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name()); assert version("boltz") == "2.2.1"'
/usr/bin/time -p boltz predict data/smoke.yaml \
  --model boltz2 --seed 0 \
  --recycling_steps 3 \
  --sampling_steps 200 --diffusion_samples 1 \
  --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
  --num_workers 2 --preprocessing-threads 2 \
  --out_dir results/stock_smoke --override
python - <<'PY'
import json
import math
from pathlib import Path

prediction = Path("results/stock_smoke/boltz_results_smoke/predictions/smoke")
structure = prediction / "smoke_model_0.cif"
affinity = prediction / "affinity_smoke.json"
assert structure.is_file(), structure
scores = json.loads(affinity.read_text())
for key in ("affinity_probability_binary", "affinity_pred_value"):
    assert math.isfinite(scores[key]), (key, scores[key])
assert 0 <= scores["affinity_probability_binary"] <= 1
print(json.dumps({"structure": str(structure), "affinity": str(affinity), "scores": scores}, indent=2))
PY
