"""
Automated Dual-Benchmark Evaluation Harness for Merged Models (E2).
Evaluates:
1. Math Retention: SVAMP (Full 300 Test Problems, Batched batch_size=16) -> Computes CR_math
2. Preference Retention: UltraFeedback (200 Held-Out Pairs, Sequence Logps) -> Computes PRR
Formats and exports Table 2 (Midpoint Evaluation Table).
"""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.path.append(os.path.abspath("."))
import torch
torch.set_num_threads(2)
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.eval_math import extract_predicted_answer, is_equivalent

# Baseline anchors established in Experiment 1
B0_MATH_ACC = 53.3333
EMATH_MATH_ACC = 58.6667
DELTA_MATH = EMATH_MATH_ACC - B0_MATH_ACC  # +5.3334%

B0_PREF_MARGIN = 0.0000
EPREF_PREF_MARGIN = 0.31489
DELTA_PREF_MARGIN = EPREF_PREF_MARGIN - B0_PREF_MARGIN  # +0.31489

B0_RAW_WIN = 51.5000
EPREF_RAW_WIN = 54.5000
DELTA_RAW_WIN = EPREF_RAW_WIN - B0_RAW_WIN  # +3.00%

def evaluate_svamp(model, tokenizer, n_samples=300, batch_size=16) -> float:
    print(f"  -> Evaluating SVAMP ({n_samples} problems, batch_size={batch_size})...")
    ds = load_dataset("ChilleD/SVAMP", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    items = []
    for item in ds:
        body = item.get("Body", "").strip()
        q = item.get("Question", "").strip()
        full_q = f"{body} {q}".strip() if body else q
        gold = str(item.get("Answer", "")).strip()
        items.append({"question": full_q, "gold": gold})

    total = len(items)
    correct = 0

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start:batch_start + batch_size]
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
            if is_equivalent(pred, it["gold"]):
                correct += 1

    acc = (correct / total) * 100.0
    print(f"  -> SVAMP Result: {acc:.2f}% ({correct}/{total})")
    return acc

def compute_sequence_logp(model, tokenizer, prompt: str, completion: str) -> tuple[float, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)

    prompt_len = len(prompt_ids)
    completion_losses = loss_per_token[prompt_len - 1:]

    total_logp = -completion_losses.sum().item()
    norm_logp = total_logp / max(1, len(completion_losses))
    return total_logp, norm_logp

def evaluate_preference(model, tokenizer, ref_cache: list, n_pairs=200, beta=0.1) -> tuple[float, float]:
    print(f"  -> Evaluating UltraFeedback Preference ({n_pairs} pairs)...")
    test_path = "outputs/datasets/e_pref_test.jsonl"
    dataset = load_dataset("json", data_files=test_path, split="train")
    if n_pairs < len(dataset):
        dataset = dataset.select(range(n_pairs))

    raw_wins = 0
    dpo_margins = []

    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        tot_c, norm_c = compute_sequence_logp(model, tokenizer, prompt, chosen)
        tot_r, norm_r = compute_sequence_logp(model, tokenizer, prompt, rejected)

        if norm_c > norm_r:
            raw_wins += 1

        ref_c = ref_cache[i]["chosen_tot_logp"]
        ref_r = ref_cache[i]["rejected_tot_logp"]

        margin = beta * ((tot_c - tot_r) - (ref_c - ref_r))
        dpo_margins.append(margin)

    raw_win_rate = (raw_wins / n_pairs) * 100.0
    mean_margin = float(sum(dpo_margins) / len(dpo_margins))
    print(f"  -> Preference Result: Raw Win Rate={raw_win_rate:.2f}%, DPO Margin={mean_margin:+.4f}")
    return raw_win_rate, mean_margin

def evaluate_model_pipeline(model_path: str, ref_cache: list) -> dict:
    print(f"\n{'='*70}\nEVALUATING MODEL: {model_path}\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    math_acc = evaluate_svamp(model, tokenizer, n_samples=300, batch_size=16)
    raw_win, dpo_margin = evaluate_preference(model, tokenizer, ref_cache, n_pairs=200, beta=0.1)

    # Compute Normalized Capability Retention (CR_math)
    cr_math = ((math_acc - B0_MATH_ACC) / DELTA_MATH) * 100.0

    # Compute Preference Retention Ratio (PRR)
    prr = (dpo_margin / DELTA_PREF_MARGIN) * 100.0

    del model
    torch.cuda.empty_cache()

    return {
        "model_path": model_path,
        "math_acc": math_acc,
        "cr_math": cr_math,
        "raw_win_rate": raw_win,
        "dpo_margin": dpo_margin,
        "prr": prr
    }

def run_phase_2a_eval():
    """Runs dual-benchmark evaluation for Phase 2A merged models and compiles Table 2."""
    # 1. Preload B0 reference log-probability cache
    print("Loading or computing B0 reference cache for UltraFeedback...")
    ref_cache_path = "outputs/results/b0_pref_200_cache.json"
    if os.path.exists(ref_cache_path):
        with open(ref_cache_path, "r") as f:
            ref_cache = json.load(f)
        print(f"Loaded existing B0 reference cache from {ref_cache_path}")
    else:
        # Precompute and cache B0 logps
        print("Precomputing B0 logps on GPU...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        b0_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to("cuda")
        b0_model.eval()

        test_path = "outputs/datasets/e_pref_test.jsonl"
        dataset = load_dataset("json", data_files=test_path, split="train").select(range(200))
        ref_cache = []
        for item in dataset:
            tot_c, norm_c = compute_sequence_logp(b0_model, tokenizer, item["prompt"], item["chosen"])
            tot_r, norm_r = compute_sequence_logp(b0_model, tokenizer, item["prompt"], item["rejected"])
            ref_cache.append({"chosen_tot_logp": tot_c, "rejected_tot_logp": tot_r})

        with open(ref_cache_path, "w") as f:
            json.dump(ref_cache, f, indent=2)
        print(f"Saved B0 reference cache to {ref_cache_path}")
        del b0_model
        torch.cuda.empty_cache()

    # Models to benchmark in Phase 2A
    models = [
        ("Base Model (B0)", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("Math Specialist (E_math)", "outputs/checkpoints/e_math"),
        ("Preference Specialist (E_pref)", "outputs/checkpoints/e_pref"),
        ("Linear (Naive, α=0.5)", "outputs/checkpoints/merge_linear_naive"),
        ("Linear (Norm-Balanced, α=0.5)", "outputs/checkpoints/merge_linear_balanced"),
        ("SLERP (t=0.5)", "outputs/checkpoints/merge_slerp"),
        ("TIES (Density=0.2, α=0.5)", "outputs/checkpoints/merge_ties"),
        ("DARE-TIES (Drop=0.5, Density=0.2, α=0.5)", "outputs/checkpoints/merge_dare_ties"),
    ]

    results = []
    for label, path in models:
        metrics = evaluate_model_pipeline(path, ref_cache)
        metrics["label"] = label
        results.append(metrics)

    # Format Table 2
    md_lines = [
        "# Table 2: Model Merging Midpoint Evaluation (Phase 2A: α = 0.5)",
        "",
        "| Merging Method | SVAMP Accuracy (%) | CR_math (%) | UltraFeedback Margin | PRR (%) | Raw Win Rate (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in results:
        md_lines.append(
            f"| **{r['label']}** | {r['math_acc']:.2f} | {r['cr_math']:.1f}% | {r['dpo_margin']:+.4f} | {r['prr']:.1f}% | {r['raw_win_rate']:.2f}% |"
        )

    md_content = "\n".join(md_lines)
    print("\n" + "="*70)
    print(md_content)
    print("="*70)

    # Save outputs
    out_json = "outputs/results/table2_merging_midpoint.json"
    out_md = "outputs/results/table2_merging_midpoint.md"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_md, "w") as f:
        f.write(md_content + "\n")
    print(f"Table 2 exported to {out_md} and {out_json}")

if __name__ == "__main__":
    run_phase_2a_eval()
