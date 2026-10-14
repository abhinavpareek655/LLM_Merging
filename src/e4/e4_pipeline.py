"""
Experiment 4 (E4): Order of operations at an equal DPO budget.

  W1  SFT experts -> merge                      (unaligned reference)
  W2  each SFT expert -> DPO(shard) -> merge    (align first, then merge)
  W3  SFT experts -> merge -> DPO(full pool)    (merge first, then align)

Budget (same preference pool, 2,000 UltraFeedback pairs, same hyper-parameters as E_pref):
  W2: 3 experts x one epoch on disjoint, token-balanced shards of 664/664/672 pairs
      = 2,000 pairs, 83+83+84 = 250 optimizer steps in total
  W3: one epoch over the whole 2,000-pair pool = 250 optimizer steps
Pair counts are multiples of 8 (= grad accumulation), so no partial batch is dropped.

Everything is a resumable *job* (marker file in outputs/e4/done/).  Run one job with
    python src/e4/e4_pipeline.py run --job <job_id>
and list them with
    python src/e4/e4_pipeline.py list
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
import argparse
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42            # DPO seed (same as configs/dpo_config.yaml)
DARE_SEED = 10       # same as run_e2_sweep.py
DENSITY = 0.20       # TIES density (E2)
DROP = 0.50          # DARE drop rate (E2)
METHODS = ["linear", "ties", "dare_ties"]
EXPERTS = ["math", "code", "instr"]
SHARD_SIZES = {"math": 664, "code": 664, "instr": 672}
WEIGHTS = [1.0 / 3] * 3
BETA, LR, GRAD_ACCUM, MAX_LEN, WD = 0.1, 5.0e-5, 8, 1024, 0.01   # == dpo_config.yaml

TRAIN_JSONL = "outputs/datasets/e_pref_train.jsonl"
TEST_JSONL = "outputs/datasets/e_pref_test.jsonl"
E2_B0_CACHE = "outputs/results/b0_pref_200_cache.json"
B0_CACHE_500 = "outputs/e4/b0_pref_500_cache.json"
E4 = "outputs/e4"
DONE = f"{E4}/done"
RESULTS = f"{E4}/results"
METRICS = f"{E4}/train_metrics"


# ----------------------------------------------------------------------------- paths
def ckpt_path(tag: str) -> str:
    if tag == "b0":
        return MODEL_ID
    if tag in ("e_math", "e_code", "e_instr", "e_pref"):
        return f"outputs/checkpoints/{tag}"
    return f"outputs/checkpoints/e4/{tag}"       # al_*, w1_*, w2_*, w3_*


def all_eval_tags():
    tags = ["b0", "e_math", "e_code", "e_instr", "e_pref"]
    tags += [f"al_{k}" for k in EXPERTS]
    for w in ("w1", "w2", "w3"):
        tags += [f"{w}_{m}" for m in METHODS]
    return tags


def done_path(job):
    return f"{DONE}/{job}.json"


def is_done(job):
    return os.path.exists(done_path(job))


def mark_done(job, info=None):
    os.makedirs(DONE, exist_ok=True)
    with open(done_path(job), "w") as f:
        json.dump(info or {}, f, indent=2)


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- data shards
def split_balanced(tokens, sizes, seed=SEED, iters=40000):
    """Random disjoint split of len(tokens) items into shards of the given sizes, then
    hill-climb pair swaps so every shard's total tokens is proportional to its size
    (i.e. equal tokens per pair / per optimizer step)."""
    tokens = np.asarray(tokens, dtype=np.float64)
    assert sum(sizes) == len(tokens)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tokens))
    bounds = np.cumsum([0] + list(sizes))
    shards = [list(perm[bounds[i]:bounds[i + 1]]) for i in range(len(sizes))]
    target = [tokens.sum() * s / len(tokens) for s in sizes]
    sums = [tokens[s].sum() for s in shards]
    cost = lambda ss: sum((a - t) ** 2 for a, t in zip(ss, target))
    cur = cost(sums)
    for _ in range(iters):
        a, b = rng.choice(len(sizes), 2, replace=False)
        i, j = rng.randint(len(shards[a])), rng.randint(len(shards[b]))
        d = tokens[shards[b][j]] - tokens[shards[a][i]]
        new = list(sums)
        new[a] += d
        new[b] -= d
        c = cost(new)
        if c < cur:
            shards[a][i], shards[b][j] = shards[b][j], shards[a][i]
            sums, cur = new, c
    return shards, sums


def pair_tokens(tok, r):
    p = len(tok.encode(r["prompt"], add_special_tokens=False))
    c = len(tok.encode(r["chosen"], add_special_tokens=False))
    j = len(tok.encode(r["rejected"], add_special_tokens=False))
    return min(p + c, MAX_LEN) + min(p + j, MAX_LEN)


def job_shards():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = [json.loads(l) for l in open(TRAIN_JSONL, encoding="utf-8") if l.strip()]
    assert len(rows) == 2000, f"expected 2000 train pairs, got {len(rows)}"
    toks = np.array([pair_tokens(tok, r) for r in rows])
    shards, sums = split_balanced(toks, [SHARD_SIZES[k] for k in EXPERTS])
    os.makedirs(f"{E4}/data", exist_ok=True)
    info = {"total_pairs": 2000, "total_tokens": int(toks.sum()), "shards": {}}
    for k, idx, s in zip(EXPERTS, shards, sums):
        with open(f"{E4}/data/shard_{k}.jsonl", "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        info["shards"][k] = {"pairs": len(idx), "tokens": int(s),
                             "tokens_per_pair": float(s / len(idx)), "indices": [int(i) for i in idx]}
    assert len(set(i for k in info["shards"] for i in info["shards"][k]["indices"])) == 2000
    json.dump(info, open(f"{E4}/data/shards_info.json", "w"), indent=1)
    print({k: (v["pairs"], v["tokens"], round(v["tokens_per_pair"], 1)) for k, v in info["shards"].items()},
          "total tokens", info["total_tokens"])
    mark_done("shards", {k: v["pairs"] for k, v in info["shards"].items()})


# ----------------------------------------------------------------------------- baking
def bake(base_path, adapter_path, out_path):
    """Bake a LoRA adapter into base_path weights (CPU, bf16 - same as bake_adapters.py)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cpu",
                                                attn_implementation="sdpa")
    m = PeftModel.from_pretrained(base, adapter_path).merge_and_unload().to(torch.bfloat16)
    os.makedirs(out_path, exist_ok=True)
    m.save_pretrained(out_path, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_path)
    del m, base
    gc.collect()


def job_bake_sft(name):          # name in math/code/instr/pref
    out = ckpt_path(f"e_{name}")
    if not os.path.exists(os.path.join(out, "config.json")):
        bake(MODEL_ID, f"outputs/adapters/e_{name}", out)
    mark_done(f"bake_sft_{name}")


# ----------------------------------------------------------------------------- merging
def load_sd(path):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu",
                                             low_cpu_mem_usage=True)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


def merge_n(method, base_sd, expert_sds, weights, density=DENSITY, drop=DROP, seed=DARE_SEED):
    """N-way merge, same conventions as run_e2_sweep.py generalised from 2 to N experts:
    weights are applied to task vectors; TIES = per-tensor top-`density` trim, sign elected from
    the weighted sum, weighted sum of agreeing entries (no divide-by-count; weights sum to 1);
    DARE = Bernoulli drop + 1/(1-drop) rescale (seeded) before TIES.  embed/lm_head stay at base.
    Arithmetic is fp32 with a single rounding to bf16 at the end."""
    import torch
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu",
                                             low_cpu_mem_usage=True)
    m.load_state_dict(sd)
    os.makedirs(out_path, exist_ok=True)
    m.save_pretrained(out_path)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_path)
    del m
    gc.collect()


def job_merge(stage, method):    # stage w1 -> SFT experts, w2 -> DPO-aligned experts
    t0 = time.time()
    src_tags = [f"e_{k}" for k in EXPERTS] if stage == "w1" else [f"al_{k}" for k in EXPERTS]
    base_sd = load_sd(MODEL_ID)
    sds = [load_sd(ckpt_path(t)) for t in src_tags]
    merged = merge_n(method, base_sd, sds, WEIGHTS)
    out = ckpt_path(f"{stage}_{method}")
    save_sd_as_model(merged, out)
    mark_done(f"merge_{stage}_{method}", {"sources": src_tags, "weights": WEIGHTS, "density": DENSITY,
                                          "drop": DROP, "dare_seed": DARE_SEED,
                                          "merge_seconds": time.time() - t0})


# ----------------------------------------------------------------------------- reference log-probs
def seq_logp(model, tok, prompt, completion):
    """Total log-prob + #completion tokens (identical maths to run_e2_sweep.compute_sequence_logp)."""
    import torch
    p = tok.encode(prompt, add_special_tokens=False)
    c = tok.encode(completion, add_special_tokens=False)
    ids = torch.tensor([p + c], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits
    losses = torch.nn.CrossEntropyLoss(reduction="none")(logits[0, :-1, :], ids[0, 1:])
    cl = losses[len(p) - 1:]
    return -cl.sum().item(), len(cl)


def load_eval_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model.eval()
    return model, tok


def compute_ref_file(model_path, in_path, out_path, limit=None):
    """Adds ref_chosen_logp / ref_rejected_logp (from model_path) to every pair of in_path."""
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    model, tok = load_eval_model(model_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r["ref_chosen_logp"], _ = seq_logp(model, tok, r["prompt"], r["chosen"])
            r["ref_rejected_logp"], _ = seq_logp(model, tok, r["prompt"], r["rejected"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del model
    free_gpu()


def job_b0_ref500():
    rows = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8") if l.strip()]
    assert len(rows) >= 500, f"need 500 test pairs, got {len(rows)}"
    rows = rows[:500]
    model, tok = load_eval_model(MODEL_ID)
    cache = []
    for r in rows:
        c, _ = seq_logp(model, tok, r["prompt"], r["chosen"])
        j, _ = seq_logp(model, tok, r["prompt"], r["rejected"])
        cache.append({"chosen": c, "rejected": j})
    os.makedirs(E4, exist_ok=True)
    json.dump(cache, open(B0_CACHE_500, "w"))
    info = {}
    if os.path.exists(E2_B0_CACHE):
        old = json.load(open(E2_B0_CACHE))
        n = min(len(old), 200)
        d = np.array([[abs(old[i]["chosen"] - cache[i]["chosen"]), abs(old[i]["rejected"] - cache[i]["rejected"])]
                      for i in range(n)])
        info = {"vs_e2_cache_mean_abs_diff": float(d.mean()), "vs_e2_cache_max_abs_diff": float(d.max())}
        print("B0 cache vs E2 cache (first 200):", info)
    del model
    free_gpu()
    mark_done("b0_ref500", info)


# ----------------------------------------------------------------------------- DPO training
def train_dpo(start_path, data_with_ref, adapter_dir, max_steps=None, tag=""):
    """Same loop / optimiser / schedule as src/train/train_dpo_stable.py, but starting from
    `start_path` instead of the hard-coded base, and logging pairs / steps / tokens / runtime."""
    import torch
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
    model = AutoModelForCausalLM.from_pretrained(start_path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=LORA_TARGET_MODULES, bias=LORA_BIAS,
                                             task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = len(ds) // GRAD_ACCUM
    if max_steps:
        total_steps = min(total_steps, max_steps)
    sched = get_cosine_schedule_with_warmup(opt, max(5, int(total_steps * 0.05)), total_steps)

    step, sub, tokens, pairs = 0, 0, 0, 0
    acc_l = acc_a = acc_m = 0.0
    log = []
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    done = False
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
        if sub % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == 1 or step == total_steps:
                rec = {"step": step, "loss": acc_l / GRAD_ACCUM, "acc": acc_a / GRAD_ACCUM,
                       "margin": acc_m / GRAD_ACCUM, "lr": sched.get_last_lr()[0],
                       "elapsed_s": time.time() - t_train}
                log.append(rec)
                print(f"[{tag}] step {step}/{total_steps} loss={rec['loss']:.4f} acc={rec['acc']:.2f} "
                      f"margin={rec['margin']:.2f} {rec['elapsed_s']:.0f}s", flush=True)
            acc_l = acc_a = acc_m = 0.0
            if max_steps and step >= max_steps:
                done = True
                break
        if done:
            break
    train_seconds = time.time() - t_train
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    metrics = {"tag": tag, "start_model": start_path, "data": data_with_ref, "pairs": pairs,
               "optimizer_steps": step, "tokens": int(tokens), "epochs": 1,
               "train_seconds": train_seconds, "total_seconds_incl_load": time.time() - t_start,
               "sec_per_step": train_seconds / max(1, step), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "gpu": gpu_name(), "beta": BETA, "lr": LR, "log": log}
    del model, opt
    free_gpu()
    return metrics


def _save_metrics(name, m):
    os.makedirs(METRICS, exist_ok=True)
    json.dump(m, open(f"{METRICS}/{name}.json", "w"), indent=1)


def job_dpo(kind, key):
    """kind='exp'  : DPO on SFT expert `key` with its shard   -> al_{key}   (W2 building block)
       kind='w3'   : DPO on merged SFT model w1_{key}, full pool -> w3_{key}"""
    t0 = time.time()
    os.makedirs(f"{E4}/data", exist_ok=True)
    if kind == "exp":
        start, data_in, out_tag = ckpt_path(f"e_{key}"), f"{E4}/data/shard_{key}.jsonl", f"al_{key}"
    else:
        start, data_in, out_tag = ckpt_path(f"w1_{key}"), TRAIN_JSONL, f"w3_{key}"
    data_ref = f"{E4}/data/{out_tag}_with_ref.jsonl"
    t_ref = time.time()
    compute_ref_file(start, data_in, data_ref)
    ref_seconds = time.time() - t_ref
    adapter = f"{E4}/adapters/{out_tag}"
    m = train_dpo(start, data_ref, adapter, tag=out_tag)
    m["ref_precompute_seconds"] = ref_seconds
    t_bake = time.time()
    bake(start, adapter, ckpt_path(out_tag))
    m["bake_seconds"] = time.time() - t_bake
    m["job_wall_seconds"] = time.time() - t0
    _save_metrics(out_tag, m)
    mark_done(f"dpo_{kind}_{key}", {k: m[k] for k in ("pairs", "optimizer_steps", "tokens", "train_seconds")})


def job_bench():
    """3 optimizer steps (24 pairs) on B0 to time a step and check memory - nothing is kept."""
    rows = [json.loads(l) for l in open(TRAIN_JSONL, encoding="utf-8") if l.strip()][:24]
    os.makedirs(f"{E4}/bench", exist_ok=True)
    with open(f"{E4}/bench/bench.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    compute_ref_file(MODEL_ID, f"{E4}/bench/bench.jsonl", f"{E4}/bench/bench_ref.jsonl")
    m = train_dpo(MODEL_ID, f"{E4}/bench/bench_ref.jsonl", f"{E4}/bench/adapter", max_steps=3, tag="bench")
    print(f"\nBENCH: {m['sec_per_step']:.1f} s/step on {m['gpu']}, peak {m['peak_mem_gb']:.1f} GB")
    print(f"=> est. W2 (250 steps) {250 * m['sec_per_step'] / 60:.0f} min, each W3 (250 steps) "
          f"{250 * m['sec_per_step'] / 60:.0f} min, all 6 DPO jobs {1000 * m['sec_per_step'] / 3600:.1f} GPU-h")
    shutil.rmtree(f"{E4}/bench", ignore_errors=True)


# ----------------------------------------------------------------------------- evaluation
def eval_pref(path, n=500):
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


# jobs whose output must be kept until these jobs are done (used by --delete_after)
def _dependents(tag):
    if tag.startswith("w1_"):
        return [f"dpo_w3_{tag[3:]}"]
    if tag.startswith("al_"):
        return [f"merge_w2_{m}" for m in METHODS]
    if tag.startswith(("w2_", "w3_")):
        return []
    return None      # never delete (b0, SFT experts, e_pref)


def job_eval(tag, delete_after=False):
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
    deps = _dependents(tag)
    if delete_after and deps is not None and all(is_done(d) for d in deps) and path != MODEL_ID:
        shutil.rmtree(path, ignore_errors=True)
        print(f"deleted {path} (all dependents finished)")


# ----------------------------------------------------------------------------- report
def paired_boot(a, b, n_boot=2000, seed=0):
    """Mean(a)-mean(b) with a 95% paired-bootstrap CI over items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def prr_with_ci(m, parent, n_boot=2000, seed=0):
    """PRR = mean(margin_M)/mean(margin_parent) (S_base = 0 because margins are B0-relative),
    unclipped, with a paired bootstrap over test pairs."""
    m, parent = np.asarray(m, float), np.asarray(parent, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(m), (n_boot, len(m)))
    boots = m[idx].mean(1) / parent[idx].mean(1)
    return float(m.mean() / parent.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def job_report():
    R = {t: json.load(open(f"{RESULTS}/eval_{t}.json")) for t in all_eval_tags()
         if os.path.exists(f"{RESULTS}/eval_{t}.json")}
    TM = {f.split(".")[0]: json.load(open(f"{METRICS}/{f}")) for f in os.listdir(METRICS)} \
        if os.path.exists(METRICS) else {}
    merge_meta = {}
    for st in ("w1", "w2"):
        for m in METHODS:
            p = done_path(f"merge_{st}_{m}")
            if os.path.exists(p):
                merge_meta[f"{st}_{m}"] = json.load(open(p))

    def sc(t, bench):
        r = R[t]
        return {"svamp": r["svamp"]["acc"], "mbpp": r["mbpp"]["acc"], "ifeval": r["ifeval"]["prompt_acc"]}[bench]

    def cr(t, bench, expert):
        if "b0" not in R or expert not in R:
            return float("nan")
        den = sc(expert, bench) - sc("b0", bench)
        return (sc(t, bench) - sc("b0", bench)) / den * 100 if den > 0 else float("nan")

    have_al = all(f"al_{k}" in R for k in EXPERTS)
    parent_pairs = np.mean([R[f"al_{k}"]["pref"]["margins"] for k in EXPERTS], axis=0) if have_al else None
    out = {"rows": {}, "comparisons": {}}
    hdr = ("| Model | Pref margin (500) [Δ vs B0] | Raw win % (500) | Margin (first 200, E2-comparable) | "
           "PRR vs mean aligned parent [95% CI] | PRR vs E_pref | SVAMP | MBPP | IFEval (prompt) | "
           "CR_math | CR_code | CR_instr | DPO pairs | DPO steps | DPO tokens | DPO train s | merge s |")
    lines = ["# Table E4: order of operations at equal DPO budget", "", hdr,
             "|" + " :--- |" * 17]
    order = ["b0", "e_math", "e_code", "e_instr", "e_pref"] + [f"al_{k}" for k in EXPERTS] + \
            [f"{w}_{m}" for m in METHODS for w in ("w1", "w2", "w3")]
    for t in order:
        if t not in R:
            continue
        p = R[t]["pref"]
        prr_s = prr_e = "-"
        prr = None
        if parent_pairs is not None and t not in ("b0",):
            prr = prr_with_ci(p["margins"], parent_pairs)
            prr_s = f"{prr[0] * 100:.1f}% [{prr[1] * 100:.1f}, {prr[2] * 100:.1f}]"
        if "e_pref" in R and t != "b0" and R["e_pref"]["pref"]["margin_500"] != 0:
            prr_e = f"{p['margin_500'] / R['e_pref']['pref']['margin_500'] * 100:.1f}%"
        # budget columns
        if t.startswith("w2_"):
            ms = [TM[f"al_{k}"] for k in EXPERTS if f"al_{k}" in TM]
        elif t.startswith("w3_"):
            ms = [TM[t]] if t in TM else []
        elif t.startswith("al_"):
            ms = [TM[t]] if t in TM else []
        else:
            ms = []
        if ms:
            bud = (sum(x["pairs"] for x in ms), sum(x["optimizer_steps"] for x in ms),
                   sum(x["tokens"] for x in ms), sum(x["train_seconds"] for x in ms))
            bud_s = f"{bud[0]} | {bud[1]} | {bud[2]:,} | {bud[3]:.0f}"
        else:
            bud_s = "- | - | - | -"
        mseconds = merge_meta.get(t, {}).get("merge_seconds")
        crs = " | ".join("-" if t == "b0" else f"{cr(t, b, e):.0f}%" for b, e in
                         (("svamp", "e_math"), ("mbpp", "e_code"), ("ifeval", "e_instr")))
        lines.append(f"| **{t}** | {p['margin_500']:+.4f} [{p['margin_500']:+.4f}] | {p['raw_win_500']:.1f} | "
                     f"{p['margin_200']:+.4f} | {prr_s} | {prr_e} | {sc(t, 'svamp'):.2f} | {sc(t, 'mbpp'):.2f} | "
                     f"{sc(t, 'ifeval'):.2f} | {crs} | {bud_s} | {'-' if mseconds is None else f'{mseconds:.0f}'} |")
        out["rows"][t] = {"pref": {k: v for k, v in p.items() if k not in ("margins", "raw_wins")},
                          "svamp": sc(t, "svamp"), "mbpp": sc(t, "mbpp"), "ifeval": sc(t, "ifeval"),
                          "prr_vs_mean_aligned_parent": prr, "budget": ms and {
                              "pairs": bud[0], "steps": bud[1], "tokens": bud[2], "train_seconds": bud[3]}}
    # per-expert parents (W2 and W3), and paired W2-vs-W3 comparison
    lines += ["", "## PRR with each aligned expert as parent (S_base = B0)", "",
              "| Model | " + " | ".join(f"parent al_{k}" for k in EXPERTS) + " |", "|:---|" + ":---|" * len(EXPERTS)]
    for t in [f"{w}_{m}" for m in METHODS for w in ("w2", "w3")]:
        if t in R and have_al:
            cells = [f"{prr_with_ci(R[t]['pref']['margins'], R[f'al_{k}']['pref']['margins'])[0] * 100:.1f}%"
                     for k in EXPERTS]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparison W2 − W3 (95% paired bootstrap CI; >0 favours align→merge)", "",
              "| Method | Δ pref margin (500) | Δ SVAMP | Δ MBPP | Δ IFEval |", "|:---|:---|:---|:---|:---|"]
    for m in METHODS:
        a, b = R.get(f"w2_{m}"), R.get(f"w3_{m}")
        if not (a and b):
            continue
        d = {"pref": paired_boot(a["pref"]["margins"], b["pref"]["margins"]),
             "svamp": paired_boot(a["svamp"]["correct"], b["svamp"]["correct"]),
             "mbpp": paired_boot(a["mbpp"]["passed"], b["mbpp"]["passed"]),
             "ifeval": paired_boot(a["ifeval"]["prompt_pass"], b["ifeval"]["prompt_pass"])}
        f = lambda x, s=1: f"{x[0] * s:+.3f} [{x[1] * s:+.3f}, {x[2] * s:+.3f}]"
        lines.append(f"| {m} | {f(d['pref'])} | {f(d['svamp'], 100)} | {f(d['mbpp'], 100)} | {f(d['ifeval'], 100)} |")
        out["comparisons"][m] = d
    if "e_pref" in R and "e_pref" in R:
        lines += ["", f"Reproducibility check - E_pref margin on first 200 pairs: "
                      f"{R['e_pref']['pref']['margin_200']:+.5f} (E2 reported +0.31489)."]
    md = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    open(f"{RESULTS}/table_e4.md", "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(f"{RESULTS}/table_e4.json", "w"), indent=1)
    print(md)
    mark_done("report")


def job_pack():
    """Zip all small artefacts (adapters, metrics, eval JSONs, tables, logs) for download."""
    import zipfile
    zp = "e4_artifacts.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (E4,):
            for dp, _, fs in os.walk(root):
                for f in fs:
                    fp = os.path.join(dp, f)
                    if f.endswith("_with_ref.jsonl"):
                        continue
                    z.write(fp)
    print("wrote", zp, f"{os.path.getsize(zp) / 1e6:.0f} MB")


# ----------------------------------------------------------------------------- job registry / CLI
def all_jobs():
    j = ["shards"] + [f"bake_sft_{n}" for n in EXPERTS + ["pref"]] + ["b0_ref500"]
    j += [f"merge_w1_{m}" for m in METHODS]
    j += [f"dpo_exp_{k}" for k in EXPERTS] + [f"dpo_w3_{m}" for m in METHODS]
    j += [f"merge_w2_{m}" for m in METHODS]
    j += [f"eval_{t}" for t in all_eval_tags()]
    return j + ["report"]


def run_job(job, delete_after=False):
    if job == "bench":
        return job_bench()
    if job == "pack":
        return job_pack()
    if is_done(job) and job != "report":
        print(f"[skip] {job} already done")
        return
    print(f"\n===== JOB {job} =====", flush=True)
    if job == "shards":
        job_shards()
    elif job.startswith("bake_sft_"):
        job_bake_sft(job[len("bake_sft_"):])
    elif job == "b0_ref500":
        job_b0_ref500()
    elif job.startswith("merge_"):
        _, st, m = job.split("_", 2)
        job_merge(st, m)
    elif job.startswith("dpo_exp_"):
        job_dpo("exp", job[len("dpo_exp_"):])
    elif job.startswith("dpo_w3_"):
        job_dpo("w3", job[len("dpo_w3_"):])
    elif job.startswith("eval_"):
        job_eval(job[len("eval_"):], delete_after)
    elif job == "report":
        job_report()
    else:
        raise ValueError(job)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list", "status"])
    ap.add_argument("--job")
    ap.add_argument("--delete_after", action="store_true",
                    help="eval jobs: delete the checkpoint once no later job needs it (saves disk)")
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(all_jobs() + ["bench", "pack"]))
    elif a.cmd == "status":
        for j in all_jobs():
            print(("DONE " if is_done(j) else "todo ") + j)
    else:
        run_job(a.job, a.delete_after)
