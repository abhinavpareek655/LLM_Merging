"""
Deterministic Preference Evaluation Suite.
Computes:
1. Raw token-normalized log-likelihood win rate (B0 ~ 58%, E_pref ~ 72%).
2. DPO implicit reward margin relative to B0 (B0 = 0.00 / 50.0%, E_pref > 0).
Can cache reference model log-probabilities to avoid redundant computation.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import argparse
import json
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import TORCH_DTYPE, ATTN_IMPLEMENTATION, MODEL_ID


def compute_sequence_logp(model, tokenizer, prompt: str, completion: str) -> tuple[float, float, int]:
    """Computes total log-prob, token-normalized log-prob, and token count of completion given prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
        
    # Shift logits and labels for causal LM
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    
    # Calculate cross entropy per token (negative log likelihood)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)
    
    # Isolate tokens belonging strictly to the completion
    prompt_len = len(prompt_ids)
    completion_token_losses = loss_per_token[prompt_len - 1:]
    
    total_logp = -completion_token_losses.sum().item()
    num_tokens = len(completion_token_losses)
    norm_logp = total_logp / max(1, num_tokens)
    
    return total_logp, norm_logp, num_tokens


def evaluate_preference(
    model_path: str,
    test_dataset_path: str,
    ref_logps_cache_path: str = None,
    beta: float = 0.1,
    max_samples: int = None,
    output_json: str = None
):
    print(f"Loading preference test dataset from: {test_dataset_path}")
    dataset = load_dataset("json", data_files=test_dataset_path, split="train")
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} preference pairs on model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()

    # Load cached reference logps if provided
    ref_logps = None
    if ref_logps_cache_path and os.path.exists(ref_logps_cache_path):
        with open(ref_logps_cache_path, "r") as f:
            ref_logps = json.load(f)
        print(f"Loaded cached reference logps from {ref_logps_cache_path}")

    raw_wins = 0
    implicit_dpo_wins = 0
    dpo_margins = []
    cached_records = []

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating Preference")):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        tot_c, norm_c, len_c = compute_sequence_logp(model, tokenizer, prompt, chosen)
        tot_r, norm_r, len_r = compute_sequence_logp(model, tokenizer, prompt, rejected)

        # 1. Raw token-normalized log-likelihood win
        if norm_c > norm_r:
            raw_wins += 1

        record = {
            "idx": idx,
            "chosen_tot_logp": tot_c,
            "chosen_norm_logp": norm_c,
            "rejected_tot_logp": tot_r,
            "rejected_norm_logp": norm_r
        }
        cached_records.append(record)

        # 2. DPO Implicit margin relative to reference
        if ref_logps is not None and idx < len(ref_logps):
            ref_c = ref_logps[idx]["chosen_tot_logp"]
            ref_r = ref_logps[idx]["rejected_tot_logp"]
            # Delta r = beta * [ (log pi(yc) - log pi(yr)) - (log pref(yc) - log pref(yr)) ]
            pi_diff = tot_c - tot_r
            ref_diff = ref_c - ref_r
            margin = beta * (pi_diff - ref_diff)
            dpo_margins.append(margin)
            if margin > 1e-7:
                implicit_dpo_wins += 1.0
            elif abs(margin) <= 1e-7:
                implicit_dpo_wins += 0.5  # Neutral reference tie = 50% credit

    total = len(dataset)
    raw_acc = (raw_wins / total) * 100.0
    mean_margin = float(torch.tensor(dpo_margins).mean().item()) if dpo_margins else 0.0
    implicit_acc = (implicit_dpo_wins / total) * 100.0 if dpo_margins else 50.0

    print("\n" + "="*60)
    print(f"PREFERENCE EVALUATION RESULTS: {model_path}")
    print(f"Raw Log-Likelihood Win Rate : {raw_acc:.2f}% ({raw_wins}/{total})")
    if dpo_margins:
        print(f"DPO Implicit Reward Margin   : {mean_margin:+.4f} (vs Reference)")
        print(f"DPO Implicit Win Rate        : {implicit_acc:.2f}%")
    print("="*60)

    # Save logps cache if this is the base model and cache didn't exist
    if ref_logps_cache_path and (ref_logps is None or not os.path.exists(ref_logps_cache_path)):
        os.makedirs(os.path.dirname(ref_logps_cache_path), exist_ok=True)
        with open(ref_logps_cache_path, "w") as f:
            json.dump(cached_records, f, indent=2)
        print(f"Cached reference logps to: {ref_logps_cache_path}")

    summary = {
        "model": model_path,
        "total_pairs": total,
        "raw_accuracy": raw_acc,
        "raw_wins": raw_wins,
        "dpo_implicit_margin": mean_margin,
        "dpo_implicit_accuracy": implicit_acc
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on preference pairs.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test preference JSONL")
    parser.add_argument("--ref_cache", type=str, default=None, help="Path to reference logps cache")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_preference(
        model_path=args.model_path,
        test_dataset_path=args.test_dataset,
        ref_logps_cache_path=args.ref_cache,
        beta=args.beta,
        max_samples=args.max_samples,
        output_json=args.output_json
    )
