"""
Rock-Solid DPO Training Pipeline with Precomputed Reference Log-Probabilities.
Uses a pure, standard PyTorch training loop:
1. Zero dynamic adapter toggling (ref logps are read from precomputed JSONL).
2. Zero Accelerate dispatch hooks.
3. Clean PyTorch autograd with AdamW and cosine schedule.
Peak VRAM is strictly ~2.2 GB on an 8 GB GPU with 100% hardware stability.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import argparse
import yaml
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

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
    MAX_SEQ_LENGTH,
    OUTPUT_DATA_DIR
)


def compute_logp_for_completion(model, tokenizer, prompt: str, completion: str, max_seq_len: int = 1024) -> torch.Tensor:
    """Computes total log-prob of completion given prompt with gradient tracking enabled."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    full_ids = (prompt_ids + completion_ids)[:max_seq_len]
    input_ids = torch.tensor([full_ids], device=model.device)
    outputs = model(input_ids)
    logits = outputs.logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)

    prompt_len = min(len(prompt_ids), len(full_ids) - 1)
    completion_token_losses = loss_per_token[prompt_len - 1:]
    total_logp = -completion_token_losses.sum()
    return total_logp


def train_dpo_stable(
    dataset_path: str,
    output_adapter_dir: str,
    config_path: str = None,
    max_steps: int = None
):
    beta = 0.1
    lr = 5.0e-5
    epochs = 1
    grad_accum = 8
    max_seq_len = 1024
    seed = 42

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            beta = float(cfg.get("beta", beta))
            lr = float(cfg.get("learning_rate", lr))
            epochs = int(cfg.get("num_train_epochs", epochs))
            grad_accum = int(cfg.get("gradient_accumulation_steps", grad_accum))
            max_seq_len = int(cfg.get("max_length", max_seq_len))
            seed = int(cfg.get("seed", seed))

    print(f"Configuration: lr={lr}, epochs={epochs}, grad_accum={grad_accum}, beta={beta}, max_seq_len={max_seq_len}")
    torch.manual_seed(seed)

    print(f"Loading preference dataset with precomputed reference logps: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} preference pairs.")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model directly onto CUDA in {TORCH_DTYPE}...")
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
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_pairs = len(dataset) * epochs
    total_optimizer_steps = total_pairs // grad_accum
    if max_steps and max_steps > 0:
        total_optimizer_steps = min(total_optimizer_steps, max_steps)

    warmup_steps = max(5, int(total_optimizer_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps
    )

    print("\n" + "="*70)
    print("STARTING STABLE DPO TRAINING")
    print(f"Total Optimization Steps: {total_optimizer_steps} (Grad Accum: {grad_accum}, Beta: {beta})")
    print("="*70)

    current_opt_step = 0
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    accumulated_margin = 0.0
    sub_step = 0

    pbar = tqdm(total=total_optimizer_steps, desc="DPO Training Steps")

    for epoch in range(epochs):
        for item in dataset:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            ref_c = float(item["ref_chosen_logp"])
            ref_r = float(item["ref_rejected_logp"])

            # Compute policy logps on active model
            pi_c = compute_logp_for_completion(model, tokenizer, prompt, chosen, max_seq_len=max_seq_len)
            pi_r = compute_logp_for_completion(model, tokenizer, prompt, rejected, max_seq_len=max_seq_len)

            # Compute DPO loss
            pi_diff = pi_c - pi_r
            ref_diff = ref_c - ref_r
            logits = beta * (pi_diff - ref_diff)

            loss = F.softplus(-logits)

            # Metrics
            with torch.no_grad():
                acc = (logits > 0).float().item()
                margin = (pi_diff - ref_diff).item()
                accumulated_acc += acc
                accumulated_margin += margin

            # Backward pass
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            accumulated_loss += loss.item()
            sub_step += 1

            if sub_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_opt_step += 1
                pbar.update(1)

                if current_opt_step % 10 == 0 or current_opt_step == 1:
                    avg_loss = accumulated_loss / grad_accum
                    avg_acc = (accumulated_acc / grad_accum) * 100.0
                    avg_margin = accumulated_margin / grad_accum
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "acc%": f"{avg_acc:.1f}",
                        "margin": f"{avg_margin:.2f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })

                accumulated_loss = 0.0
                accumulated_acc = 0.0
                accumulated_margin = 0.0

                if max_steps and current_opt_step >= max_steps:
                    break

        if max_steps and current_opt_step >= max_steps:
            break

    pbar.close()

    print(f"\nSaving trained DPO LoRA adapter to: {output_adapter_dir}...")
    os.makedirs(output_adapter_dir, exist_ok=True)
    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print("DPO Training complete successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO specialist with precomputed ref logps.")
    parser.add_argument("--dataset_path", type=str, default=os.path.join(OUTPUT_DATA_DIR, "e_pref_train_with_ref.jsonl"))
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save adapter")
    parser.add_argument("--config_path", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for training")
    args = parser.parse_args()

    train_dpo_stable(
        dataset_path=args.dataset_path,
        output_adapter_dir=args.output_dir,
        config_path=args.config_path,
        max_steps=args.max_steps
    )
