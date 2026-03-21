#!/usr/bin/env python3
"""Check vLLM version and Qwen3 model support."""
import vllm
print("vLLM:", vllm.__version__)

import transformers
print("transformers:", transformers.__version__)

# Check model registry for Qwen support
try:
    from vllm.model_executor.models import _MODELS
    qwen = {k: v for k, v in _MODELS.items() if "Qwen" in k}
    print("Qwen models in _MODELS:", qwen)
except Exception as e:
    print("_MODELS approach failed:", e)

try:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    qwen = {k: v for k, v in _VLLM_MODELS.items() if "Qwen" in k}
    print("Qwen models in _VLLM_MODELS:", qwen)
except Exception as e:
    print("_VLLM_MODELS approach failed:", e)

# Check if Qwen3 specifically exists
try:
    import vllm.model_executor.models.qwen3 as q3
    print("qwen3 module exists:", dir(q3))
except ImportError:
    print("No vllm.model_executor.models.qwen3 module")

# List all qwen-related files
import os
models_dir = os.path.dirname(vllm.model_executor.models.__file__)
qwen_files = [f for f in os.listdir(models_dir) if "qwen" in f.lower()]
print("Qwen-related files in vllm models dir:", qwen_files)
