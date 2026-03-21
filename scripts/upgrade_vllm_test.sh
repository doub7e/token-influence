#!/usr/bin/env bash
set -euo pipefail

LOGFILE=/scratch/cvlab/home/shuli/agentic-research/Archer2.0/output/vllm_upgrade_test.log
mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee "$LOGFILE") 2>&1

echo "=== Current versions ==="
python -c 'import vllm; print("vLLM:", vllm.__version__)'
python -c 'import torch; print("torch:", torch.__version__)'
python -c 'import transformers; print("transformers:", transformers.__version__)'

echo ""
echo "=== Attempting vLLM upgrade (no-deps first) ==="
pip install 'vllm>=0.8.4,<0.9.0' --no-deps 2>&1 | tail -20 || echo "FAILED: vllm 0.8.x no-deps"

echo ""
echo "=== Check new version ==="
python -c 'import vllm; print("vLLM:", vllm.__version__)' 2>&1 || echo "FAILED: cannot import vllm after upgrade"

echo ""
echo "=== Check Qwen3 support ==="
python -c '
try:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    qwen = {k:v for k,v in _VLLM_MODELS.items() if "Qwen" in k}
    print("Qwen models:", qwen)
    print("Qwen3ForCausalLM supported:", "Qwen3ForCausalLM" in _VLLM_MODELS)
except Exception as e:
    print("Registry check failed:", e)
    # Try alternative registry location
    try:
        from vllm.model_executor.models import ModelRegistry
        print("ModelRegistry found, checking Qwen3...")
        # newer vllm versions may have different registry
    except:
        pass
' 2>&1

echo ""
echo "=== Check verl compatibility ==="
python -c '
import sys
sys.path.insert(0, "/scratch/cvlab/home/shuli/agentic-research/Archer2.0")
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
print("vLLMRollout import OK")
from vllm import LLM, SamplingParams
print("LLM, SamplingParams import OK")
' 2>&1 || echo "FAILED: verl compatibility check"

echo ""
echo "=== DONE ==="
