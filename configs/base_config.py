"""
Base configuration for models, hardware, and PEFT/LoRA settings.
"""
import os
import torch

MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Hardware / Attention Backend
_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_dtype_name = os.environ.get("LLM_TORCH_DTYPE", "bfloat16").lower()
if _dtype_name not in _TORCH_DTYPES:
    raise ValueError(f"Unsupported LLM_TORCH_DTYPE: {_dtype_name}")
TORCH_DTYPE = _TORCH_DTYPES[_dtype_name]
ATTN_IMPLEMENTATION = os.environ.get("LLM_ATTN_IMPLEMENTATION", "sdpa")

# PEFT / LoRA Configuration
# Target all 7 linear projections across attention and MLP subspaces
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

LORA_R = int(os.environ.get("LLM_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LLM_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LLM_LORA_DROPOUT", "0.05"))
LORA_BIAS = os.environ.get("LLM_LORA_BIAS", "none")

# Maximum sequence length (strictly filtered before training)
MAX_SEQ_LENGTH = int(os.environ.get("LLM_MAX_SEQ_LENGTH", "1024"))

# Output paths
BASE_DIR = os.environ.get("LLM_OUTPUT_ROOT", "c:/Users/vedan/PythonWS/CS769")
OUTPUT_DATA_DIR = os.environ.get("LLM_OUTPUT_DATA_DIR", f"{BASE_DIR}/outputs/datasets")
OUTPUT_ADAPTERS_DIR = os.environ.get("LLM_OUTPUT_ADAPTERS_DIR", f"{BASE_DIR}/outputs/adapters")
OUTPUT_CHECKPOINTS_DIR = os.environ.get("LLM_OUTPUT_CHECKPOINTS_DIR", f"{BASE_DIR}/outputs/checkpoints")
OUTPUT_RESULTS_DIR = os.environ.get("LLM_OUTPUT_RESULTS_DIR", f"{BASE_DIR}/outputs/results")
OUTPUT_DIAGNOSTICS_DIR = os.environ.get("LLM_OUTPUT_DIAGNOSTICS_DIR", f"{BASE_DIR}/outputs/diagnostics")
