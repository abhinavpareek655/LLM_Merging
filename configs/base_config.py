"""
Base configuration for models, hardware, and PEFT/LoRA settings.
"""
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Hardware / Attention Backend
TORCH_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "sdpa"  # Native PyTorch SDPA: fast, 0-compilation needed

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

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# Maximum sequence length (strictly filtered before training)
MAX_SEQ_LENGTH = 1024

# Output paths
BASE_DIR = "c:/Users/vedan/PythonWS/CS769"
OUTPUT_DATA_DIR = f"{BASE_DIR}/outputs/datasets"
OUTPUT_ADAPTERS_DIR = f"{BASE_DIR}/outputs/adapters"
OUTPUT_CHECKPOINTS_DIR = f"{BASE_DIR}/outputs/checkpoints"
OUTPUT_RESULTS_DIR = f"{BASE_DIR}/outputs/results"
OUTPUT_DIAGNOSTICS_DIR = f"{BASE_DIR}/outputs/diagnostics"
