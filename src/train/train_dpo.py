"""
DPO Training Pipeline for Preference Specialist (E_pref).
Utilizes PEFT LoRA and sets ref_model=None so TRL reuses the base model with adapters disabled,
keeping peak VRAM under 3.5 GB on an 8 GB GPU.
"""
import os
import sys

# Prevent Windows CLOCK_WATCHDOG_TIMEOUT (0x101) by capping OpenMP/MKL worker threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import yaml
import torch

# Strictly limit PyTorch CPU threads so Windows watchdog interrupts are never starved
torch.set_num_threads(2)

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import (
    MODEL_ID,
    TORCH_DTYPE,
    ATTN_IMPLEMENTATION,
    LORA_TARGET_MODULES,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_BIAS,
    MAX_SEQ_LENGTH
)


def train_dpo(dataset_path: str, output_adapter_dir: str, config_path: str = None, max_steps: int = None):
    beta = 0.1
    lr = 5.0e-5
    epochs = 2
    batch_size = 1
    grad_accum = 16
    grad_ckpt = True
    max_seq_len = MAX_SEQ_LENGTH
    seed = 42

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            beta = float(cfg.get("beta", beta))
            lr = float(cfg.get("learning_rate", lr))
            epochs = int(cfg.get("num_train_epochs", epochs))
            batch_size = int(cfg.get("per_device_train_batch_size", batch_size))
            grad_accum = int(cfg.get("gradient_accumulation_steps", grad_accum))
            grad_ckpt = bool(cfg.get("gradient_checkpointing", True))
            max_seq_len = int(cfg.get("max_length", MAX_SEQ_LENGTH))
            seed = int(cfg.get("seed", seed))

    print(f"Loading preference training dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} preference pairs.")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {MODEL_ID} in {TORCH_DTYPE} with {ATTN_IMPLEMENTATION} directly onto CUDA...")
    # Do NOT pass device_map="auto" to DPOTrainer; it attaches Accelerate CPU-offload hooks
    # that flood the PCIe memory arbiter and cause CLOCK_WATCHDOG_TIMEOUT BSODs.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL: LoRA + Gradient Checkpointing requires enabling input gradients
    # to prevent autograd backward spinlocks on frozen base parameters
    if grad_ckpt:
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    dpo_config = DPOConfig(
        output_dir=output_adapter_dir,
        beta=beta,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=max_seq_len,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Non-reentrant to prevent autograd deadlocks
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        report_to="none"
    )

    if max_steps is not None and max_steps > 0:
        dpo_config.max_steps = max_steps

    print(f"Using max sequence length: {max_seq_len}")

    # Note: ref_model=None is CRUCIAL. TRL automatically calls model.disable_adapter()
    # in-place for reference passes, preventing memory duplication.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving trained DPO LoRA adapter to: {output_adapter_dir}...")
    os.makedirs(output_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print("DPO Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO preference specialist with LoRA.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input JSONL preference dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--config_path", type=str, default="configs/dpo_config.yaml", help="Path to config YAML")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for testing/validation")
    args = parser.parse_args()

    train_dpo(args.dataset_path, args.output_dir, args.config_path, max_steps=args.max_steps)
