"""
Bake LoRA adapters into base model weights and save standalone bfloat16 checkpoints.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import argparse
import torch
torch.set_num_threads(2)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# Ensure root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE, ATTN_IMPLEMENTATION


def bake_adapter(adapter_path: str, output_path: str):
    print(f"Loading base model: {MODEL_ID} in {TORCH_DTYPE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="cpu"  # Merge on CPU to conserve GPU VRAM
    )
    
    print(f"Loading adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter into base model weights (merge_and_unload)...")
    merged_model = model.merge_and_unload()
    
    # Explicitly enforce bfloat16 to avoid precision drift or storage bloat
    merged_model = merged_model.to(TORCH_DTYPE)
    
    print(f"Saving merged checkpoint to: {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    print(f"Successfully baked adapter into: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake LoRA adapter into full model weights.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged checkpoint")
    args = parser.parse_args()

    bake_adapter(args.adapter_path, args.output_path)
