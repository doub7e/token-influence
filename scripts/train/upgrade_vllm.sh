#!/usr/bin/env bash
set -euo pipefail
# Upgrade vLLM to support Qwen3 models.
# Installs to user-writable location and prepends to PYTHONPATH.

VLLM_TARGET="/scratch/cvlab/home/shuli/.local/vllm-0.8.5"

if python -c "import vllm; exit(0 if vllm.__version__.startswith('0.8') else 1)" 2>/dev/null; then
    echo "[INFO] vLLM 0.8.x already available, skipping upgrade."
    exit 0
fi

echo "[INFO] Installing vLLM 0.8.5.post1 to ${VLLM_TARGET}..."
pip install --target "${VLLM_TARGET}" 'vllm==0.8.5.post1' --no-deps 2>&1 | tail -5

echo "[INFO] vLLM upgrade installed to ${VLLM_TARGET}"
echo "[INFO] Set PYTHONPATH=${VLLM_TARGET}:\$PYTHONPATH before running training."
