#!/usr/bin/env python3
"""Read vLLM 0.7.3 qwen2.py and registry.py for Qwen3 patching."""
import os

vllm_models_dir = "/opt/archer-venv/lib/python3.10/site-packages/vllm/model_executor/models"

# Read qwen2.py
qwen2_path = os.path.join(vllm_models_dir, "qwen2.py")
with open(qwen2_path) as f:
    content = f.read()
print("=== qwen2.py ===")
print(content)

# Read registry.py (just Qwen entries and the dict structure)
registry_path = os.path.join(vllm_models_dir, "registry.py")
with open(registry_path) as f:
    reg_content = f.read()
print("\n=== registry.py ===")
print(reg_content)
