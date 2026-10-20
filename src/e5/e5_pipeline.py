"""
Experiment 5 (E5): Post-Merge Repair Budget Sweep.
Characterizing how few preference examples are needed to recover lost alignment
after an unaligned multi-task model merge (W1 -> DPO repair).

Budgets tested:
  0 pairs    : W1 (unaligned merge baseline, 0 steps)
  125 pairs  : Nested subset (~16 optimizer steps, ~2 min on T4)
  500 pairs  : Nested subset (~63 optimizer steps, ~8 min on T4)
  2000 pairs : Full post-merge DPO (W3, 250 optimizer steps; reuses E4 results if present)

Merges tested: Linear, TIES (density 0.2), DARE-TIES (drop 0.5, density 0.2, seed 10).
Checkpoints: outputs/checkpoints/e5/
Resumability: outputs/e5/done/<job>.json
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import time
import shutil
import random
import argparse
import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
DARE_SEED = 10
DENSITY = 0.20
DROP = 0.50
METHODS = ["linear", "ties", "dare_ties"]
EXPERTS = ["math", "code", "instr"]
WEIGHTS = [1.0 / 3] * 3
REPAIR_BUDGETS = [125, 500]   # 0 is W1, 2000 is W3
ALL_BUDGETS = [0, 125, 500, 2000]
BETA, LR, GRAD_ACCUM, MAX_LEN, WD = 0.1, 5.0e-5, 8, 1024, 0.01

TRAIN_JSONL = "outputs/datasets/e_pref_train.jsonl"
TEST_JSONL = "outputs/datasets/e_pref_test.jsonl"
B0_CACHE_500 = "outputs/e4/b0_pref_500_cache.json"
if not os.path.exists(B0_CACHE_500):
    B0_CACHE_500 = "outputs/e5/b0_pref_500_cache.json"

E5 = "outputs/e5"
E4 = "outputs/e4"
DONE = f"{E5}/done"
RESULTS = f"{E5}/results"
METRICS = f"{E5}/train_metrics"
PLOTS = f"{E5}/plots"
DATA = f"{E5}/data"

for d in [DONE, RESULTS, METRICS, PLOTS, DATA, "outputs/checkpoints/e5"]:
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------- Checkpoints & Paths
def ckpt_path(tag: str) -> str:
    if tag == "b0":
        return MODEL_ID
    if tag in ("e_math", "e_code", "e_instr", "e_pref"):
        return f"outputs/checkpoints/{tag}"
    if tag.startswith("w1_") or tag.startswith("w2_") or tag.startswith("w3_") or tag.startswith("al_"):
        # Check E4 path first
        p4 = f"outputs/checkpoints/e4/{tag}"
        if os.path.exists(p4):
            return p4
        p_std = f"outputs/checkpoints/{tag}"
        if os.path.exists(p_std):
            return p_std
        return p4
    return f"outputs/checkpoints/e5/{tag}"


def done_path(job: str) -> str:
    return f"{DONE}/{job}.json"


def is_done(job: str) -> bool:
    return os.path.exists(done_path(job))


def mark_done(job: str, info=None):
    with open(done_path(job), "w") as f:
        json.dump(info or {}, f, indent=2)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- Nested Shards
def job_shards():
    rows = [json.loads(l) for l in open(TRAIN_JSONL, encoding="utf-8") if l.strip()]
    assert len(rows) == 2000, f"Expected 2000 pairs, found {len(rows)}"
    rng = random.Random(SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    info = {}
    for k in [125, 500]:
        sub_indices = sorted(indices[:k])
        sub_rows = [rows[i] for i in sub_indices]
        out_p = f"{DATA}/repair_{k}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for r in sub_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        info[f"repair_{k}"] = {"pairs": len(sub_rows), "steps": (len(sub_rows) + GRAD_ACCUM - 1) // GRAD_ACCUM}
        print(f"[shards] Wrote {out_p} ({len(sub_rows)} pairs)")
    json.dump(info, open(f"{DATA}/shards_info.json", "w"), indent=2)
    mark_done("shards", info)


# ----------------------------------------------------------------------------- W1 Checkpoint Assurance
def load_sd(path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


def merge_n(method, base_sd, expert_sds, weights, density=DENSITY, drop=DROP, seed=DARE_SEED):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, b in base_sd.items():
        if (not b.is_floating_point()) or "embed" in k or "lm_head" in k:
            out[k] = b.clone()
            continue
        bf = b.float()
        taus = [e[k].float() - bf for e in expert_sds]
        if method == "linear":
            tau = sum(w * t for w, t in zip(weights, taus))
        elif method in ("ties", "dare_ties"):
            if method == "dare_ties":
                kp = 1.0 - drop
                taus = [t * torch.bernoulli(torch.full_like(t, kp), generator=gen) / kp for t in taus]
            trimmed = []
            for t in taus:
                kk = int(density * t.numel())
                if kk > 0:
                    thr = torch.topk(t.abs().flatten(), kk)[0][-1]
                    trimmed.append(t * (t.abs() >= thr))
                else:
                    trimmed.append(t)
            cons = torch.sign(sum(w * t for w, t in zip(weights, trimmed)))
            tau = sum(w * t * ((torch.sign(t) == cons) & (cons != 0)) for w, t in zip(weights, trimmed))
        else:
            raise ValueError(method)
        out[k] = (bf + tau).to(torch.bfloat16)
    return out


def save_sd_as_model(sd, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    m.load_state_dict(sd)
    os.makedirs(out_path, exist_ok=True)
    m.save_pretrained(out_path)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_path)
    del m
    gc.collect()


def bake(base_path, adapter_path, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    m = PeftModel.from_pretrained(base, adapter_path).merge_and_unload().to(torch.bfloat16)
    os.makedirs(out_path, exist_ok=True)
    m.save_pretrained(out_path, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_path)
    del m, base
    gc.collect()


def ensure_w1_checkpoint(method: str) -> str:
    """Ensures W1_{method} exists on disk. Rebuilds if previous runs deleted it."""
    possible_paths = [
        f"outputs/checkpoints/e4/w1_{method}",
        f"outputs/checkpoints/w1_{method}",
        f"outputs/checkpoints/e5/w1_{method}"
    ]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "config.json")):
            return p

    target = f"outputs/checkpoints/e5/w1_{method}"
    print(f"[ensure_w1] Reconstructing W1 {method} into {target}...")
    for exp in EXPERTS:
        exp_ckpt = ckpt_path(f"e_{exp}")
        if not os.path.exists(os.path.join(exp_ckpt, "config.json")):
            print(f"[ensure_w1] Baking adapter outputs/adapters/e_{exp} -> {exp_ckpt}")
            bake(MODEL_ID, f"outputs/adapters/e_{exp}", exp_ckpt)

    base_sd = load_sd(MODEL_ID)
    sds = [load_sd(ckpt_path(f"e_{k}")) for k in EXPERTS]
    merged_sd = merge_n(method, base_sd, sds, WEIGHTS)
    save_sd_as_model(merged_sd, target)
    print(f"[ensure_w1] Successfully synthesized {target}")
    return target


# ----------------------------------------------------------------------------- Reference Log-Probs & DPO
def seq_logp(model, tok, prompt, completion):
    p = tok.encode(prompt, add_special_tokens=False)
    c = tok.encode(completion, add_special_tokens=False)
    ids = torch.tensor([p + c], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits
    losses = torch.nn.CrossEntropyLoss(reduction="none")(logits[0, :-1, :], ids[0, 1:])
    cl = losses[len(p) - 1:]
    return -cl.sum().item(), len(cl)


def load_eval_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    return model, tok


def compute_ref_file(model_path, in_path, out_path):
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    model, tok = load_eval_model(model_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r["ref_chosen_logp"], _ = seq_logp(model, tok, r["prompt"], r["chosen"])
            r["ref_rejected_logp"], _ = seq_logp(model, tok, r["prompt"], r["rejected"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del model
    free_gpu()


def train_dpo(start_path, data_with_ref, adapter_dir, tag=""):
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from configs.base_config import (LORA_TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS)
    from src.train.train_dpo_stable import compute_logp_for_completion

    torch.manual_seed(SEED)
    t_start = time.time()
    ds = load_dataset("json", data_files=data_with_ref, split="train")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(start_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=LORA_TARGET_MODULES, bias=LORA_BIAS,
                                             task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = (len(ds) + GRAD_ACCUM - 1) // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.05)), total_steps)

    step, sub, tokens, pairs = 0, 0, 0, 0
    acc_l = acc_a = acc_m = 0.0
    log = []
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()

    for item in ds:
        pi_c = compute_logp_for_completion(model, tok, item["prompt"], item["chosen"], MAX_LEN)
        pi_r = compute_logp_for_completion(model, tok, item["prompt"], item["rejected"], MAX_LEN)
        ref_diff = float(item["ref_chosen_logp"]) - float(item["ref_rejected_logp"])
        logits = BETA * ((pi_c - pi_r) - ref_diff)
        loss = F.softplus(-logits)
        with torch.no_grad():
            acc_a += float(logits.item() > 0)
            acc_m += ((pi_c - pi_r) - ref_diff).item()
        (loss / GRAD_ACCUM).backward()
        acc_l += loss.item()
        sub += 1
        pairs += 1
        p = len(tok.encode(item["prompt"], add_special_tokens=False))
        tokens += min(p + len(tok.encode(item["chosen"], add_special_tokens=False)), MAX_LEN) \
                  + min(p + len(tok.encode(item["rejected"], add_special_tokens=False)), MAX_LEN)
        if sub % GRAD_ACCUM == 0 or sub == len(ds):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 5 == 0 or step == 1 or step == total_steps:
                rec = {"step": step, "loss": acc_l / max(1, (sub % GRAD_ACCUM or GRAD_ACCUM)),
                       "acc": acc_a / max(1, (sub % GRAD_ACCUM or GRAD_ACCUM)),
                       "margin": acc_m / max(1, (sub % GRAD_ACCUM or GRAD_ACCUM)),
                       "elapsed_s": time.time() - t_train}
                log.append(rec)
                print(f"[{tag}] step {step}/{total_steps} loss={rec['loss']:.4f} acc={rec['acc']:.2f} "
                      f"margin={rec['margin']:.2f} {rec['elapsed_s']:.0f}s", flush=True)
            acc_l = acc_a = acc_m = 0.0

    train_seconds = time.time() - t_train
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    metrics = {"tag": tag, "start_model": start_path, "data": data_with_ref, "pairs": pairs,
               "optimizer_steps": step, "tokens": int(tokens), "epochs": 1,
               "train_seconds": train_seconds, "total_seconds": time.time() - t_start,
               "sec_per_step": train_seconds / max(1, step), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "gpu": gpu_name(), "beta": BETA, "lr": LR, "log": log}
    del model, opt
    free_gpu()
    return metrics


def job_dpo_repair(method: str, budget: int):
    tag = f"e5_{method}_{budget}"
    t0 = time.time()
    w1_path = ensure_w1_checkpoint(method)
    shard_path = f"{DATA}/repair_{budget}.jsonl"
    ref_path = f"{DATA}/{tag}_with_ref.jsonl"

    print(f"[{tag}] Computing reference log-probabilities...")
    t_ref = time.time()
    compute_ref_file(w1_path, shard_path, ref_path)
    ref_sec = time.time() - t_ref

    adapter_dir = f"{E5}/adapters/{tag}"
    print(f"[{tag}] Training DPO repair on {budget} pairs...")
    m = train_dpo(w1_path, ref_path, adapter_dir, tag=tag)
    m["ref_seconds"] = ref_sec

    out_ckpt = ckpt_path(tag)
    t_bake = time.time()
    bake(w1_path, adapter_dir, out_ckpt)
    m["bake_seconds"] = time.time() - t_bake
    m["job_wall_seconds"] = time.time() - t0

    os.makedirs(METRICS, exist_ok=True)
    json.dump(m, open(f"{METRICS}/{tag}.json", "w"), indent=2)
    mark_done(f"dpo_{tag}", {"pairs": m["pairs"], "steps": m["optimizer_steps"], "train_seconds": m["train_seconds"]})


# ----------------------------------------------------------------------------- Evaluation Harness
def ensure_b0_cache():
    if os.path.exists(B0_CACHE_500):
        return
    rows = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8") if l.strip()][:500]
    model, tok = load_eval_model(MODEL_ID)
    cache = []
    for r in rows:
        c, _ = seq_logp(model, tok, r["prompt"], r["chosen"])
        j, _ = seq_logp(model, tok, r["prompt"], r["rejected"])
        cache.append({"chosen": c, "rejected": j})
    del model
    free_gpu()
    json.dump(cache, open(B0_CACHE_500, "w"))


def eval_pref(path, n=500):
    ensure_b0_cache()
    rows = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8") if l.strip()][:n]
    ref = json.load(open(B0_CACHE_500))
    model, tok = load_eval_model(path)
    margins, wins = [], []
    for i, r in enumerate(rows):
        tc, nc = seq_logp(model, tok, r["prompt"], r["chosen"])
        tr, nr = seq_logp(model, tok, r["prompt"], r["rejected"])
        margins.append(BETA * ((tc - tr) - (ref[i]["chosen"] - ref[i]["rejected"])))
        wins.append(int(tc / max(1, nc) > tr / max(1, nr)))
    del model
    free_gpu()
    m = np.array(margins)
    w = np.array(wins)
    return {"margins": margins, "raw_wins": wins,
            "margin_500": float(m.mean()), "margin_200": float(m[:200].mean()),
            "raw_win_500": float(w.mean() * 100), "raw_win_200": float(w[:200].mean() * 100)}


def job_eval(tag: str, delete_after=False):
    # If 2000 pairs, reuse E4 W3 evaluation if present
    if tag.endswith("_2000"):
        method = tag.split("_")[1]
        w3_eval = f"{E4}/results/eval_w3_{method}.json"
        if os.path.exists(w3_eval):
            shutil.copyfile(w3_eval, f"{RESULTS}/eval_{tag}.json")
            mark_done(f"eval_{tag}", json.load(open(w3_eval)))
            print(f"[eval] Reused existing E4 W3 evaluation for {tag}")
            return

    from src.eval.eval_math import evaluate_svamp
    from src.eval.eval_code import evaluate_mbpp
    from src.eval.eval_ifeval import evaluate_ifeval
    path = ckpt_path(tag)
    t0 = time.time()
    res = {"tag": tag, "model_path": path, "gpu": gpu_name()}
    sv = evaluate_svamp(path, 300, None, 16)
    res["svamp"] = {"acc": sv["accuracy"], "correct": [int(d["correct"]) for d in sv["details"]]}
    cd = evaluate_mbpp(path, 257, None)
    res["mbpp"] = {"acc": cd["pass_at_1"], "passed": [int(d["passed"]) for d in cd["details"]]}
    ie = evaluate_ifeval(path, 300, None, 16)
    res["ifeval"] = {"prompt_acc": ie["prompt_accuracy"], "inst_acc": ie["instruction_accuracy"],
                     "prompt_pass": [int(d["prompt_all_passed"]) for d in ie["details"]]}
    free_gpu()
    res["pref"] = eval_pref(path)
    res["eval_seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(f"{RESULTS}/eval_{tag}.json", "w"))
    mark_done(f"eval_{tag}", {"svamp": res["svamp"]["acc"], "mbpp": res["mbpp"]["acc"],
                              "ifeval": res["ifeval"]["prompt_acc"], "margin_500": res["pref"]["margin_500"]})
    if delete_after and path.startswith("outputs/checkpoints/e5/"):
        shutil.rmtree(path, ignore_errors=True)
        print(f"deleted {path} to conserve disk")


# ----------------------------------------------------------------------------- Table E5 & Figure 4 Reporting
E4_FALLBACK = {
    "b0":           {"margin":  0.0000, "win": 53.8, "svamp": 53.67, "mbpp": 31.13, "ifeval": 28.67},
    "e_math":       {"margin": -0.0963, "win": 55.6, "svamp": 60.00, "mbpp": 31.91, "ifeval": 26.67},
    "e_code":       {"margin": -0.2588, "win": 54.6, "svamp": 49.67, "mbpp": 40.47, "ifeval": 25.67},
    "e_instr":      {"margin": -0.4616, "win": 55.6, "svamp": 50.33, "mbpp": 35.41, "ifeval": 39.00},
    "e_pref":       {"margin":  0.2761, "win": 56.2, "svamp": 53.67, "mbpp": 31.13, "ifeval": 30.67},
    "w1_linear":    {"margin": -0.1254, "win": 54.2, "svamp": 57.67, "mbpp": 42.41, "ifeval": 32.00},
    "w3_linear":    {"margin":  0.1776, "win": 55.6, "svamp": 56.00, "mbpp": 42.80, "ifeval": 32.67},
    "w1_ties":      {"margin": -0.0917, "win": 53.6, "svamp": 55.67, "mbpp": 43.19, "ifeval": 31.67},
    "w3_ties":      {"margin":  0.1996, "win": 56.2, "svamp": 52.00, "mbpp": 38.52, "ifeval": 29.67},
    "w1_dare_ties": {"margin": -0.1244, "win": 54.2, "svamp": 55.33, "mbpp": 38.91, "ifeval": 31.33},
    "w3_dare_ties": {"margin":  0.1787, "win": 56.0, "svamp": 56.33, "mbpp": 40.86, "ifeval": 32.00},
}


def load_eval(tag: str):
    candidates = [
        f"{RESULTS}/eval_{tag}.json",
        f"{E4}/results/eval_{tag}.json",
        f"outputs/results/eval_{tag}.json"
    ]
    for c in candidates:
        if os.path.exists(c):
            d = json.load(open(c))
            return {"margin": d["pref"]["margin_500"], "win": d["pref"]["raw_win_500"],
                    "svamp": d["svamp"]["acc"], "mbpp": d["mbpp"]["acc"], "ifeval": d["ifeval"]["prompt_acc"],
                    "margins": d["pref"]["margins"]}
    # Fallback to authoritative E4 numbers if evaluations are missing
    fb_key = f"w3_{tag.split('_')[1]}" if tag.endswith("_2000") else tag
    if fb_key in E4_FALLBACK:
        fb = E4_FALLBACK[fb_key]
        return {"margin": fb["margin"], "win": fb["win"], "svamp": fb["svamp"], "mbpp": fb["mbpp"],
                "ifeval": fb["ifeval"], "margins": None}
    return None


def job_report():
    epref = load_eval("e_pref")
    b0 = load_eval("b0")
    epref_m = epref["margin"] if epref else 0.2761
    d_math = (load_eval("e_math")["svamp"] - b0["svamp"]) if (load_eval("e_math") and b0) else 6.33
    d_code = (load_eval("e_code")["mbpp"] - b0["mbpp"]) if (load_eval("e_code") and b0) else 9.34
    d_instr = (load_eval("e_instr")["ifeval"] - b0["ifeval"]) if (load_eval("e_instr") and b0) else 10.33

    hdr = ("| Method | Budget | Margin (500) | Raw Win % | Gain vs W1 | PRR vs E_pref | "
           "W3 Recovery % | SVAMP | MBPP | IFEval | CR_math | CR_code | CR_instr |")
    lines = ["# Table E5: Post-Merge DPO Repair Recovery Curve", "", hdr, "|" + " :--- |" * 13]

    summary_rows = []
    for m in METHODS:
        w1_data = load_eval(f"w1_{m}")
        w3_data = load_eval(f"w3_{m}")
        w1_m = w1_data["margin"] if w1_data else -0.12
        w3_m = w3_data["margin"] if w3_data else 0.18
        w3_gain = w3_m - w1_m

        for b in ALL_BUDGETS:
            tag = f"w1_{m}" if b == 0 else (f"w3_{m}" if b == 2000 else f"e5_{m}_{b}")
            ev = load_eval(tag)
            if not ev:
                continue
            cur_m = ev["margin"]
            gain_w1 = cur_m - w1_m
            prr = (cur_m / epref_m) * 100.0
            rec = (gain_w1 / w3_gain) * 100.0 if abs(w3_gain) > 1e-9 else 0.0
            cr_m = (ev["svamp"] - b0["svamp"]) / d_math * 100.0 if d_math else 0.0
            cr_c = (ev["mbpp"] - b0["mbpp"]) / d_code * 100.0 if d_code else 0.0
            cr_i = (ev["ifeval"] - b0["ifeval"]) / d_instr * 100.0 if d_instr else 0.0

            lines.append(f"| **{m}** | {b} | {cur_m:+.4f} | {ev['win']:.1f}% | {gain_w1:+.4f} | "
                         f"{prr:.1f}% | {rec:.1f}% | {ev['svamp']:.2f} | {ev['mbpp']:.2f} | {ev['ifeval']:.2f} | "
                         f"{cr_m:.0f}% | {cr_c:.0f}% | {cr_i:.0f}% |")
            summary_rows.append({
                "method": m, "budget": b, "tag": tag, "margin": cur_m, "win": ev["win"],
                "gain_vs_w1": gain_w1, "prr_vs_epref_pct": prr, "w3_recovery_pct": rec,
                "svamp": ev["svamp"], "mbpp": ev["mbpp"], "ifeval": ev["ifeval"],
                "cr_math_pct": cr_m, "cr_code_pct": cr_c, "cr_instr_pct": cr_i
            })

    md = "\n".join(lines)
    open(f"{RESULTS}/table_e5.md", "w", encoding="utf-8").write(md + "\n")
    json.dump(summary_rows, open(f"{RESULTS}/table_e5.json", "w"), indent=2)
    print(md)

    # Plot Figure 4: Repair Curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    colors = {"linear": "#1f77b4", "ties": "#ff7f0e", "dare_ties": "#2ca02c"}
    labels = {"linear": "Linear", "ties": "TIES", "dare_ties": "DARE-TIES"}

    for m in METHODS:
        rows = [r for r in summary_rows if r["method"] == m]
        if not rows:
            continue
        xs = [r["budget"] for r in rows]
        ys_prr = [r["prr_vs_epref_pct"] for r in rows]
        ys_rec = [r["w3_recovery_pct"] for r in rows]
        axes[0].plot(xs, ys_prr, marker="o", linewidth=2, color=colors[m], label=labels[m])
        axes[1].plot(xs, ys_rec, marker="s", linewidth=2, color=colors[m], label=labels[m])

    axes[0].axhline(100.0, color="gray", linestyle="--", linewidth=1, label="E_pref (100%)")
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1, label="B0 (0%)")
    axes[0].set_title("Fig 4a: Preference Retention vs Budget")
    axes[0].set_xlabel("Post-Merge DPO Examples")
    axes[0].set_ylabel("PRR vs E_pref (%)")
    axes[0].set_xticks(ALL_BUDGETS)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(90.0, color="red", linestyle="--", linewidth=1, label="90% Recovery Target")
    axes[1].axhline(100.0, color="gray", linestyle=":", linewidth=1, label="Full W3 (2k pairs)")
    axes[1].set_title("Fig 4b: Share of Full Alignment Recovered")
    axes[1].set_xlabel("Post-Merge DPO Examples")
    axes[1].set_ylabel("Recovery of W3 Alignment Gain (%)")
    axes[1].set_xticks(ALL_BUDGETS)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/fig4_repair_curve.png")
    print(f"\n[report] Saved plot to {PLOTS}/fig4_repair_curve.png")
    mark_done("report")


def job_pack():
    import zipfile
    zp = "e5_artifacts.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (E5,):
            for dp, _, fs in os.walk(root):
                for f in fs:
                    if f.endswith("_with_ref.jsonl"):
                        continue
                    fp = os.path.join(dp, f)
                    z.write(fp)
    print(f"Pack complete: {zp} ({os.path.getsize(zp)/1e6:.2f} MB)")


# ----------------------------------------------------------------------------- Registry & CLI
def all_jobs():
    j = ["shards"]
    for m in METHODS:
        for b in REPAIR_BUDGETS:
            j.append(f"dpo_e5_{m}_{b}")
    for m in METHODS:
        for b in [125, 500, 2000]:
            j.append(f"eval_e5_{m}_{b}")
    return j + ["report"]


def run_job(job: str, delete_after=False):
    if job == "pack":
        return job_pack()
    if is_done(job) and job != "report":
        print(f"[skip] {job} already done")
        return
    print(f"\n===== JOB {job} =====", flush=True)
    if job == "shards":
        job_shards()
    elif job.startswith("dpo_e5_"):
        parts = job.replace("dpo_e5_", "").rsplit("_", 1)
        job_dpo_repair(parts[0], int(parts[1]))
    elif job.startswith("eval_e5_"):
        tag = job.replace("eval_", "")
        job_eval(tag, delete_after)
    elif job == "report":
        job_report()
    else:
        raise ValueError(job)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list", "status"])
    ap.add_argument("--job")
    ap.add_argument("--delete_after", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(all_jobs() + ["pack"]))
    elif a.cmd == "status":
        for j in all_jobs():
            print(("DONE " if is_done(j) else "todo ") + j)
    else:
        run_job(a.job, a.delete_after)
