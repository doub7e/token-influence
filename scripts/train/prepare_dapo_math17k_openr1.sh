#!/usr/bin/env bash
set -euo pipefail

# Prepare Archer-format JSON training data from the HF dataset:
#   open-r1/DAPO-Math-17k-Processed
#
# This script is intended to be run inside the Archer container via RunAI
# (CPU-only is fine). It overwrites the default math training JSON.

cd "$(dirname "$0")/../.."

PY="/opt/archer-venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

echo "[INFO] Using python: $PY"
"$PY" -V

echo "[INFO] Ensuring deps (datasets, huggingface_hub)"
if ! "$PY" -c "import datasets, huggingface_hub" >/dev/null 2>&1; then
  # Try regular install first (works in most container venvs), then fall back to user site.
  if ! "$PY" -m pip install -q datasets huggingface_hub; then
    "$PY" -m pip install -q --user datasets huggingface_hub
  fi
fi

echo "[INFO] datasets version:"
"$PY" -c "import datasets; print(datasets.__version__)"

echo "[INFO] Dataset revision (best-effort):"
"$PY" - <<'PY'
try:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info("open-r1/DAPO-Math-17k-Processed")
    print("id", info.id)
    print("sha", info.sha)
except Exception as exc:
    print("[WARN] cannot query dataset sha:", repr(exc))
PY

OUT="data/train/archer2.0-math-1.5b-train.json"
LOG="output/prepare_dapo_math17k_open-r1.log"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

echo "[INFO] Writing: $OUT"
"$PY" scripts/train/prepare_dapo_math17k.py \
  --dataset open-r1/DAPO-Math-17k-Processed \
  --split train \
  --output "$OUT" \
  2>&1 | tee "$LOG"

echo "[INFO] Sanity check:"
"$PY" - <<'PY'
import json
from pathlib import Path

p = Path("data/train/archer2.0-math-1.5b-train.json")
obj = json.loads(p.read_text(encoding="utf-8"))
print("num_samples", len(obj))
print("first_keys", list(obj[0].keys()))
PY

