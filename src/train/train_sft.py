"""
SFT Training Pipeline for Consortium Domain Experts (E_math, E_code, E_instr).
Utilizes PEFT LoRA on all 7 linear projections with native PyTorch SDPA in bfloat16.
"""
import os
import sys
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

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


def train_sft(dataset_path: str, output_adapter_dir: str, config_path: str = None):
    # Load hyperparameter defaults
    lr = 2.0e-4
    epochs = 3
    batch_size = 2
    grad_accum = 8
    seed = 42

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            lr = float(cfg.get("learning_rate", lr))
            epochs = int(cfg.get("num_train_epochs", epochs))
            batch_size = int(cfg.get("per_device_train_batch_size", batch_size))
            grad_accum = int(cfg.get("gradient_accumulation_steps", grad_accum))
            seed = int(cfg.get("seed", seed))

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if "text" in dataset.column_names:
        dataset = dataset.select_columns(["text"])
    print(f"Loaded {len(dataset)} training examples.")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {MODEL_ID} in {TORCH_DTYPE} with {ATTN_IMPLEMENTATION}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=output_adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        gradient_checkpointing=False,
        seed=seed,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving trained LoRA adapter to: {output_adapter_dir}...")
    os.makedirs(output_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print("SFT Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT specialist with LoRA.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--config_path", type=str, default="configs/sft_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    train_sft(args.dataset_path, args.output_dir, args.config_path)
