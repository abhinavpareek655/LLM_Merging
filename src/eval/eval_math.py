"""
GSM8K Exact Match (EM) Evaluation Suite.
Evaluates mathematical reasoning by comparing extracted numerical answers
against GSM8K ground truth values.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import re
import argparse
import json
import torch
torch.set_num_threads(2)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from configs.base_config import TORCH_DTYPE, ATTN_IMPLEMENTATION
except ImportError:
    TORCH_DTYPE = torch.bfloat16
    ATTN_IMPLEMENTATION = "sdpa"


def extract_gold_answer(answer_str: str) -> str:
    """Extracts final numeric ground truth from GSM8K solution (following ####)."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def extract_predicted_answer(text: str) -> str:
    """Extracts predicted numerical answer from generated response."""
    # Look for explicit markers first
    patterns = [
        r"####\s*(-?[\d,]+(?:\.\d+)?)",
        r"(?:the\s+answer\s+is|final\s+answer\s+is)[:\s]*\$?(-?[\d,]+(?:\.\d+)?)",
        r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}",
        r"= ?(-?[\d,]+(?:\.\d+)?)\s*$"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()

    # Fallback to the last standalone number
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def is_equivalent(pred: str, gold: str) -> bool:
    """Checks numerical equivalence between predicted and ground truth answers."""
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


def evaluate_gsm8k(
    model_path: str,
    max_samples: int = None,
    output_json: str = None
):
    print(f"Loading GSM8K dataset...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        dataset = load_dataset("gsm8k", "main", split="test")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} GSM8K problems on model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()

    correct_count = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating GSM8K"):
        question = item["question"]
        gold_raw = item["answer"]
        gold = extract_gold_answer(gold_raw)

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": question}
        ]

        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_predicted_answer(gen_text)

        correct = is_equivalent(pred, gold)
        if correct:
            correct_count += 1

        results.append({
            "question": question,
            "gold_raw": gold_raw,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "response": gen_text
        })

    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accuracy = (correct_count / len(dataset)) * 100.0
    print(f"\n==========================================")
    print(f"GSM8K Exact Match Accuracy: {accuracy:.2f}% ({correct_count}/{len(dataset)})")
    print(f"==========================================")

    final_payload = {
        "model": model_path,
        "exact_match_accuracy": accuracy,
        "total": len(dataset),
        "correct": correct_count,
        "details": results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(final_payload, f, indent=2)
        print(f"Saved evaluation results to: {output_json}")

    return final_payload


def evaluate_svamp(
    model_path: str,
    max_samples: int = None,
    output_json: str = None,
    batch_size: int = 16
):
    """Evaluates mathematical reasoning on the SVAMP test benchmark (300 problems)."""
    print(f"Loading SVAMP dataset...")
    try:
        dataset = load_dataset("ChilleD/SVAMP", split="test")
    except Exception:
        dataset = load_dataset("Chillee/SVAMP", split="test")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} SVAMP problems on model: {model_path} (batch_size={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()

    items = []
    for item in dataset:
        body = item.get("Body", "").strip()
        q = item.get("Question", "").strip()
        full_q = f"{body} {q}".strip() if body else q
        gold = str(item.get("Answer", "")).strip()
        if gold.endswith(".0"):
            gold = gold[:-2]
        items.append({"question": full_q, "gold": gold})

    total = len(items)
    correct_count = 0
    results = []

    for i in tqdm(range(0, total, batch_size), desc="Evaluating SVAMP (Batched)"):
        batch = items[i : i + batch_size]
        prompts = []
        for it in batch:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": it["question"]}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for it, gen_text in zip(batch, decoded):
            pred = extract_predicted_answer(gen_text)
            correct = is_equivalent(pred, it["gold"])
            if correct:
                correct_count += 1
            results.append({
                "question": it["question"],
                "gold": it["gold"],
                "pred": pred,
                "correct": correct,
                "response": gen_text
            })

    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accuracy = (correct_count / total) * 100.0
    print(f"\n==========================================")
    print(f"SVAMP Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"==========================================")

    final_payload = {
        "model": model_path,
        "benchmark": "svamp",
        "accuracy": accuracy,
        "total": total,
        "correct": correct_count,
        "details": results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(final_payload, f, indent=2)
        print(f"Saved evaluation results to: {output_json}")

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on Math benchmarks (SVAMP or GSM8K).")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--benchmark", type=str, default="svamp", choices=["svamp", "gsm8k"], help="Benchmark dataset (default: svamp)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    if args.benchmark == "svamp":
        evaluate_svamp(args.model_path, args.max_samples, args.output_json, args.batch_size)
    else:
        evaluate_gsm8k(args.model_path, args.max_samples, args.output_json)
