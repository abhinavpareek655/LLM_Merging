"""
Experiment 2 (E2): Full Two-Model Preference Retention Sweep.
Sweeps alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] across:
1. Linear Task Arithmetic
2. SLERP
3. TIES-Merging
4. DARE-TIES (Deterministic seed = 10)

Evaluates:
- UltraFeedback (200 held-out pairs) -> PRR (Preference Retention Ratio)
- SVAMP (300 test problems, batched) -> CR_math (Capability Retention)
Outputs: Table 2 (Markdown & LaTeX) + JSON results.
Fully resumable, in-memory streaming, 0 matplotlib imports.
"""
import os
import sys

# Ensure repository root is on sys.path FIRST
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import copy
import traceback
import numpy as np

import torch
torch.set_num_threads(2)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.eval_math import extract_predicted_answer, is_equivalent

RANDOM_SEED = 10  # Explicit seed requested by user
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Baseline Anchors established in Experiment 1
B0_MATH_ACC = 53.3333
EMATH_MATH_ACC = 58.6667
DELTA_MATH = EMATH_MATH_ACC - B0_MATH_ACC  # +5.3334%

B0_PREF_MARGIN = 0.0000
EPREF_PREF_MARGIN = 0.31489
DELTA_PREF_MARGIN = EPREF_PREF_MARGIN - B0_PREF_MARGIN  # +0.31489

B0_RAW_WIN = 51.5000
EPREF_RAW_WIN = 54.5000

PROGRESS_FILE = "outputs/results/e2_sweep_progress.json"
FINAL_RESULTS_FILE = "outputs/results/e2_sweep_results.json"
TABLE2_MD_FILE = "outputs/results/table2_merging_full.md"
TABLE2_TEX_FILE = "outputs/results/table2_merging_full.tex"
CRASH_LOG_FILE = "scratch/crash_log.txt"

def load_state_dict_cpu(path: str):
    print(f"Loading state dict to CPU: {path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd

def slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    orig_shape = v0.shape
    v0_f = v0.flatten().float()
    v1_f = v1.flatten().float()
    n0 = torch.norm(v0_f)
    n1 = torch.norm(v1_f)
    if n0 < eps or n1 < eps:
        return ((1.0 - t) * v0_f + t * v1_f).view(orig_shape).to(v0.dtype)
    u0 = v0_f / n0
    u1 = v1_f / n1
    dot = torch.clamp(torch.sum(u0 * u1), -1.0, 1.0)
    if torch.abs(dot) > (1.0 - 1e-4):
        return ((1.0 - t) * v0_f + t * v1_f).view(orig_shape).to(v0.dtype)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    c0 = torch.sin((1.0 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega
    u_interp = c0 * u0 + c1 * u1
    n_interp = (1.0 - t) * n0 + t * n1
    res = n_interp * (u_interp / torch.norm(u_interp))
    return res.view(orig_shape).to(v0.dtype)

def merge_state_dict(
    method: str,
    alpha: float,
    base_sd: dict,
    math_sd: dict,
    pref_sd: dict,
    density: float = 0.20,
    drop_rate: float = 0.50
) -> dict:
    if abs(alpha - 0.0) < 1e-5:
        return {k: v.clone() for k, v in math_sd.items()}
    if abs(alpha - 1.0) < 1e-5:
        return {k: v.clone() for k, v in pref_sd.items()}

    w_math = 1.0 - alpha
    w_pref = alpha

    if method == "dare_ties":
        torch.manual_seed(RANDOM_SEED)

    merged = {}
    for k in base_sd.keys():
        b_val = base_sd[k]
        m_val = math_sd[k]
        p_val = pref_sd[k]

        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            merged[k] = b_val.clone()
            continue

        tau_m = m_val - b_val
        tau_p = p_val - b_val

        if method == "linear":
            tau = w_math * tau_m + w_pref * tau_p
            merged[k] = b_val + tau

        elif method == "slerp":
            # Standard SLERP on full weights (matching canonical MergeKit)
            merged[k] = slerp_tensor(m_val, p_val, t=alpha)

        elif method == "ties" or method == "dare_ties":
            if method == "dare_ties":
                keep_prob = 1.0 - drop_rate
                mask_m = torch.bernoulli(torch.full_like(tau_m.float(), keep_prob)).to(torch.bool)
                tau_m_proc = (tau_m.float() * mask_m) / keep_prob
                mask_p = torch.bernoulli(torch.full_like(tau_p.float(), keep_prob)).to(torch.bool)
                tau_p_proc = (tau_p.float() * mask_p) / keep_prob
            else:
                tau_m_proc = tau_m.float()
                tau_p_proc = tau_p.float()

            k_m = int(density * tau_m_proc.numel())
            if k_m > 0:
                t_m_thresh = torch.topk(torch.abs(tau_m_proc.flatten()), k_m)[0][-1]
                t_m_trim = tau_m_proc * (torch.abs(tau_m_proc) >= t_m_thresh)
            else:
                t_m_trim = tau_m_proc

            k_p = int(density * tau_p_proc.numel())
            if k_p > 0:
                t_p_thresh = torch.topk(torch.abs(tau_p_proc.flatten()), k_p)[0][-1]
                t_p_trim = tau_p_proc * (torch.abs(tau_p_proc) >= t_p_thresh)
            else:
                t_p_trim = tau_p_proc

            consensus_sum = w_math * t_m_trim + w_pref * t_p_trim
            consensus_sign = torch.sign(consensus_sum)

            agree_m = (torch.sign(t_m_trim) == consensus_sign) & (consensus_sign != 0)
            agree_p = (torch.sign(t_p_trim) == consensus_sign) & (consensus_sign != 0)

            val_m = t_m_trim * agree_m
            val_p = t_p_trim * agree_p

            tau_ties = w_math * val_m + w_pref * val_p
            merged[k] = b_val + tau_ties.to(b_val.dtype)

        else:
            raise ValueError(f"Unknown merge method: {method}")

    return merged

def compute_sequence_logp(model, tokenizer, prompt: str, completion: str) -> tuple[float, float]:
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    c_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = torch.tensor([p_ids + c_ids], device=model.device)

    with torch.no_grad():
        logits = model(input_ids).logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits, shift_labels)
    c_losses = losses[len(p_ids) - 1:]

    tot_logp = -c_losses.sum().item()
    norm_logp = tot_logp / max(1, len(c_losses))
    return tot_logp, norm_logp

def eval_preference(model, tokenizer, pairs, ref_cache, beta=0.1) -> tuple[float, float]:
    raw_wins = 0
    margins = []
    for i, it in enumerate(pairs):
        tot_c, norm_c = compute_sequence_logp(model, tokenizer, it["prompt"], it["chosen"])
        tot_r, norm_r = compute_sequence_logp(model, tokenizer, it["prompt"], it["rejected"])
        if norm_c > norm_r:
            raw_wins += 1
        ref_c = ref_cache[i]["chosen"]
        ref_r = ref_cache[i]["rejected"]
        margin = beta * ((tot_c - tot_r) - (ref_c - ref_r))
        margins.append(margin)

    raw_rate = (raw_wins / len(pairs)) * 100.0
    mean_margin = float(sum(margins) / len(margins))
    return raw_rate, mean_margin

def eval_svamp(model, tokenizer, svamp_items, batch_size=16) -> float:
    total = len(svamp_items)
    correct = 0
    for b_idx in range(0, total, batch_size):
        batch = svamp_items[b_idx:b_idx + batch_size]
        prompts = []
        for it in batch:
            msgs = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": it["question"]}
            ]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        in_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gens = tokenizer.batch_decode(outputs[:, in_len:], skip_special_tokens=True)
        for it, gen in zip(batch, gens):
            pred = extract_predicted_answer(gen)
            if is_equivalent(pred, it["gold"]):
                correct += 1

    return (correct / total) * 100.0

def export_tables(results_dict: dict):
    os.makedirs("outputs/results", exist_ok=True)
    methods = ["linear", "slerp", "ties", "dare_ties"]
    method_labels = {
        "linear": "Linear Task Arithmetic",
        "slerp": "SLERP",
        "ties": "TIES-Merging",
        "dare_ties": "DARE-TIES"
    }

    alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    md_lines = [
        "# Table 2: Experiment 2 (E2) Two-Model Merging Retention Master Table",
        "",
        "| $\\alpha$ | Method | SVAMP EM (%) | $\\text{CR}_{\\text{math}}$ (%) | DPO Margin | PRR (%) | Raw Win (%) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Preference Retention Ratio (PRR) and Math Capability Retention ($\text{CR}_{\text{math}}$) across merge coefficients $\alpha \in [0.0, 1.0]$.}",
        r"\label{tab:table2_merging_full}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$\alpha$ & Method & SVAMP (\%) & $\text{CR}_{\text{math}}$ (\%) & DPO Margin & PRR (\%) & Win Rate (\%) \\",
        r"\midrule"
    ]

    for a in alphas:
        a_str = f"{a:.1f}"
        for m in methods:
            key = f"{m}_{a_str}"
            if key in results_dict:
                d = results_dict[key]
                lbl = method_labels[m]
                md_lines.append(f"| {a_str} | **{lbl}** | {d['svamp_acc']:.2f} | {d['cr_math']:+.1f}% | {d['dpo_margin']:+.4f} | {d['prr']:+.1f}% | {d['raw_win_rate']:.2f}% |")
                tex_lines.append(f"{a_str} & {lbl} & {d['svamp_acc']:.2f} & {d['cr_math']:+.1f}\\% & {d['dpo_margin']:+.4f} & {d['prr']:+.1f}\\% & {d['raw_win_rate']:.2f}\\% \\\\")

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    with open(TABLE2_MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(TABLE2_TEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Exported Table 2 to {TABLE2_MD_FILE} and {TABLE2_TEX_FILE}", flush=True)

def main():
    print("="*70, flush=True)
    print(f"EXPERIMENT 2 (E2): FULL RETENTION SWEEP (SEED={RANDOM_SEED})", flush=True)
    print("="*70, flush=True)

    progress = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)
            print(f"Resuming from existing progress file: {len(progress)} steps completed.", flush=True)
        except Exception:
            progress = {}

    print("\nLoading test datasets into RAM...", flush=True)
    pref_ds = load_dataset("json", data_files="outputs/datasets/e_pref_test.jsonl", split="train").select(range(200))
    pref_pairs = list(pref_ds)

    svamp_ds = load_dataset("ChilleD/SVAMP", split="test")
    svamp_items = []
    for it in svamp_ds:
        b = it.get("Body", "").strip()
        q = it.get("Question", "").strip()
        full_q = f"{b} {q}".strip() if b else q
        svamp_items.append({"question": full_q, "gold": str(it.get("Answer", "")).strip()})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Instantiating base model on GPU...", flush=True)
    gpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to("cuda")
    gpu_model.eval()

    ref_cache_file = "outputs/results/b0_pref_200_cache.json"
    if os.path.exists(ref_cache_file):
        with open(ref_cache_file, "r") as f:
            ref_cache = json.load(f)
        print(f"Loaded B0 reference cache from {ref_cache_file}", flush=True)
    else:
        print("Computing B0 reference log-probabilities for UltraFeedback...", flush=True)
        ref_cache = []
        for it in pref_pairs:
            c_logp, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["chosen"])
            r_logp, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["rejected"])
            ref_cache.append({"chosen": c_logp, "rejected": r_logp})
        os.makedirs("outputs/results", exist_ok=True)
        with open(ref_cache_file, "w") as f:
            json.dump(ref_cache, f, indent=2)
        print(f"Cached B0 reference logps to {ref_cache_file}", flush=True)

    base_sd = load_state_dict_cpu(MODEL_ID)
    math_sd = load_state_dict_cpu("outputs/checkpoints/e_math")
    pref_sd = load_state_dict_cpu("outputs/checkpoints/e_pref")

    methods = ["linear", "slerp", "ties", "dare_ties"]
    alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]

    total_steps = len(methods) * len(alphas)
    step_num = 0

    for m in methods:
        for a in alphas:
            step_num += 1
            key = f"{m}_{a:.1f}"
            print(f"\n[{step_num}/{total_steps}] METHOD: {m.upper()} | ALPHA: {a:.1f}", flush=True)

            if key in progress:
                print(f"  -> Already completed. Skipping. (PRR={progress[key]['prr']:.1f}%, CR={progress[key]['cr_math']:.1f}%)", flush=True)
                continue

            merged_sd = merge_state_dict(m, a, base_sd, math_sd, pref_sd)
            gpu_model.load_state_dict(merged_sd)

            raw_win, dpo_margin = eval_preference(gpu_model, tokenizer, pref_pairs, ref_cache, beta=0.1)
            prr = (dpo_margin / DELTA_PREF_MARGIN) * 100.0

            svamp_acc = eval_svamp(gpu_model, tokenizer, svamp_items, batch_size=16)
            cr_math = ((svamp_acc - B0_MATH_ACC) / DELTA_MATH) * 100.0

            print(f"  -> RESULT: SVAMP={svamp_acc:.2f}% (CR={cr_math:+.1f}%) | DPO Margin={dpo_margin:+.4f} (PRR={prr:+.1f}%) | Raw Win={raw_win:.2f}%", flush=True)

            progress[key] = {
                "method": m,
                "alpha": a,
                "svamp_acc": svamp_acc,
                "cr_math": cr_math,
                "dpo_margin": dpo_margin,
                "prr": prr,
                "raw_win_rate": raw_win
            }

            with open(PROGRESS_FILE, "w") as f:
                json.dump(progress, f, indent=2)

            del merged_sd
            gc.collect()
            torch.cuda.empty_cache()

    with open(FINAL_RESULTS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"\nSaved final results to {FINAL_RESULTS_FILE}", flush=True)

    export_tables(progress)
    print("\nEXPERIMENT 2 SWEEP COMPLETE!", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        os.makedirs("scratch", exist_ok=True)
        with open(CRASH_LOG_FILE, "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
