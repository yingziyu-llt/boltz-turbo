#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$PWD}"
BASE="${BASE:-$(command -v python3)}"
ENV="$ROOT/anthropic-env"
KIT="$ROOT/anthropic-kit/boltz2"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ROOT/cache/pip"
mkdir -p "$ROOT/tmp" "$PIP_CACHE_DIR"
"$BASE" -m venv "$ENV"
export PATH="$ENV/bin:$PATH"
ROOT="$ROOT" python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['ROOT'])
lines = (root / 'anthropic-kit/boltz2/environment/requirements.lock').read_text().splitlines()
(root / 'tmp/anthropic-bootstrap.txt').write_text('\n'.join(x for x in lines if x.startswith(('pip==', 'wheel=='))) + '\n')
(root / 'tmp/anthropic-stack.txt').write_text('\n'.join(x for x in lines if x and not x.startswith(('#', 'pip==', 'wheel==', 'boltz=='))) + '\n')
(root / 'tmp/anthropic-build.txt').write_text('setuptools==82.0.1\n')
PY
python -m pip install --no-deps -r "$ROOT/tmp/anthropic-bootstrap.txt"
python -m pip install --no-deps --build-constraint "$ROOT/tmp/anthropic-build.txt" -r "$ROOT/tmp/anthropic-stack.txt"
python -m pip install --no-deps "$KIT/stock/boltz-2.2.1-py3-none-any.whl"
bash "$KIT/run.sh" install
python -m pip freeze --all
