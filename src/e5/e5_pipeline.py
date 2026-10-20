"""
Experiment 5 (E5): Post-merge repair budget.

Take alignment-degrading two-model merges from E2 (E_math + E_pref; "worst" and "median" PRR cases),
run post-merge DPO with a small budget of preference pairs (nominally 125 / 500 / 2k, here 128 / 512 / 2000 so
every budget is a multiple of the 8-pair gradient accumulation) and measure how much of the parent's
(E_pref) preference gain is recovered.

  pre_<case>        the E2 merge (rebuilt with E2's own merge_state_dict), margin before repair
  rep_<case>_<n>    pre_<case> -> DPO on the first n pairs of one fixed permutation of the pool (nested subsets)
  ctl_b0_<n>        control: DPO from B0 with the same n pairs (is a merged model easier to repair than B0 is to align?)
  parent            E_pref = DPO on B0 with all 2,000 pairs (the PRR denominator)

Reuses the E4 building blocks (bake, DPO loop, evals, bootstrap) so that settings are identical to E4:
beta 0.1, lr 5e-5, cosine, warmup max(5, 5%), batch 1 x accum 8, LoRA r16/a32, seed 42, greedy evals,
SVAMP 300 / (MBPP 257, IFEval 300 optional) / UltraFeedback margin on 500 test pairs.
DPO reference model: the immediate pre-DPO checkpoint (config ref_mode "pre_dpo"), optionally B0 ("b0").

Every step is a resumable job:  python src/e5/e5_pipeline.py run --job <id>   |   list [--stage S]   |   status
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import math
import time
import shutil
import argparse
import numpy as np

from src.e4 import e4_pipeline as E4P

MODEL_ID = E4P.MODEL_ID
E5 = "outputs/e5"
RESULTS, METRICS, DATA = f"{E5}/results", f"{E5}/train_metrics", f"{E5}/data"
E2_RESULTS = ["outputs/results/e2_sweep_results.json", "outputs/results/e2_sweep_progress.json"]
E2_METHODS = ["linear", "ties", "dare_ties"]
ANCHORS = ["b0", "e_math", "e_pref"]

DEFAULT_CFG = {
    "budgets": [128, 512, 2000],          # nominal 125 / 500 / 2k; add 32, 64 (finer) or 8000 (needs a bigger pool)
    "cases": ["worst", "median"],         # or explicit [["linear", 0.1], ["ties", 0.5]]
    "ref_modes": ["pre_dpo"],             # add "b0" for the reference-model ablation
    "controls": True,                     # DPO-from-B0 control for budgets < 2000
    "full_eval": False,                   # False: SVAMP + preference (fast); True: + MBPP + IFEval
    "pool_path": E4P.TRAIN_JSONL,
    "seed": 42,
    "thresholds": [0.90, 0.95],
    "reuse_e4_anchors": True,
}


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    p = f"{E5}/config.json"
    if os.path.exists(p):
        cfg.update(json.load(open(p)))
    return cfg


CFG = load_cfg()
SEED = int(CFG["seed"])


def sfx():
    return "" if SEED == 42 else f"_s{SEED}"


DONE = f"{E5}/done{sfx()}"          # per-seed markers; eval results / merges of seed-independent models are shared


# ----------------------------------------------------------------------------- naming
def tag_pre(c):
    return f"pre_{c}"


def tag_rep(c, n, rm):
    return f"rep_{c}_{n}" + ("" if rm == "pre_dpo" else "_b0ref") + sfx()


def tag_ctl(n):
    return f"ctl_b0_{n}{sfx()}"


def ckpt(tag):
    if tag in ANCHORS:
        return E4P.ckpt_path(tag)
    return f"outputs/checkpoints/e5/{tag}"


def done_path(j):
    return f"{DONE}/{j}.json"


def is_done(j):
    return os.path.exists(done_path(j))


def mark_done(j, info=None):
    os.makedirs(DONE, exist_ok=True)
    json.dump(info or {}, open(done_path(j), "w"), indent=2)


def pool_perm_path():
    return f"{DATA}/pool_perm{sfx()}.jsonl"


def ref_path(name):
    return f"{DATA}/ref_{name}{sfx()}.jsonl"


# ----------------------------------------------------------------------------- case selection
def select_from_e2(e2, which):
    """Pick the 'worst' / 'median' merge by E2 PRR among linear/TIES/DARE-TIES at 0 < alpha < 1."""
    rows = sorted((v["prr"], k) for k, v in e2.items() if v["method"] in E2_METHODS and 0 < v["alpha"] < 1)
    if not rows:
        raise RuntimeError("no merged cases found in the E2 results")
    pick = {"worst": rows[0], "median": rows[len(rows) // 2], "best": rows[-1]}[which]
    return pick[1]


def case_name(method, alpha):
    return f"{method}_{float(alpha):.1f}"


def job_select_cases():
    e2 = None
    for p in E2_RESULTS:
        if os.path.exists(p):
            e2 = json.load(open(p))
            break
    cases = []
    for c in CFG["cases"]:
        if isinstance(c, str):
            if e2 is None:
                raise RuntimeError("cases given as worst/median but no E2 results file found; pass explicit cases")
            cases.append(select_from_e2(e2, c))
        else:
            cases.append(case_name(c[0], c[1]))
    cases = list(dict.fromkeys(cases))
    info = {"cases": cases, "e2": {c: {"prr_e2": e2[c]["prr"], "cr_math_e2": e2[c]["cr_math"]}
                                   for c in cases if e2 and c in e2}}
    os.makedirs(E5, exist_ok=True)
    json.dump(info, open(f"{E5}/cases.json", "w"), indent=1)
    print("cases:", info)
    mark_done("select_cases", info)


def get_cases():
    p = f"{E5}/cases.json"
    return json.load(open(p))["cases"] if os.path.exists(p) else []


# ----------------------------------------------------------------------------- data
def job_pool():
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(CFG["pool_path"], encoding="utf-8") if l.strip()]
    need = max(CFG["budgets"])
    assert len(rows) >= need, f"pool has {len(rows)} pairs but the largest budget is {need}"
    perm = np.random.RandomState(SEED).permutation(len(rows))
    os.makedirs(DATA, exist_ok=True)
    with open(pool_perm_path(), "w", encoding="utf-8") as f:
        for i in perm:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    toks = [E4P.pair_tokens(tok, rows[i]) for i in perm]
    info = {"pool_pairs": len(rows), "seed": SEED, "perm_head": [int(i) for i in perm[:20]],
            "tokens_per_budget": {str(n): int(sum(toks[:n])) for n in CFG["budgets"]}}
    json.dump(info, open(f"{DATA}/pool_info{sfx()}.json", "w"), indent=1)
    print(info["tokens_per_budget"])
    mark_done("pool" + sfx(), info)


# ----------------------------------------------------------------------------- merge (E2-exact)
def job_merge(c):
    from src.merge.run_e2_sweep import merge_state_dict, load_state_dict_cpu   # E2's own merge (bf16, seed 10)
    method, alpha = c.rsplit("_", 1)
    t0 = time.time()
    base = load_state_dict_cpu(MODEL_ID)
    math_sd = load_state_dict_cpu(E4P.ckpt_path("e_math"))
    pref_sd = load_state_dict_cpu(E4P.ckpt_path("e_pref"))
    merged = merge_state_dict(method, float(alpha), base, math_sd, pref_sd)
    E4P.save_sd_as_model(merged, ckpt(tag_pre(c)))
    mark_done(f"merge_{c}", {"method": method, "alpha": float(alpha), "seconds": time.time() - t0})


# ----------------------------------------------------------------------------- references
def count_lines(p):
    return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0


def job_ref(name):
    """Reference log-probs for the first max(budgets) pairs of the permuted pool.
    name = 'b0' (reference B0) or a case id (reference = its pre-repair merge)."""
    n = max(CFG["budgets"])
    out = ref_path(name)
    if count_lines(out) >= n:
        mark_done(f"ref_{name}{sfx()}")
        return
    start = MODEL_ID if name == "b0" else ckpt(tag_pre(name))
    t0 = time.time()
    E4P.compute_ref_file(start, pool_perm_path(), out, limit=n)
    mark_done(f"ref_{name}{sfx()}", {"pairs": n, "seconds": time.time() - t0})


# ----------------------------------------------------------------------------- repair / control DPO
def _first_n(src, dst, n):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)


def _run_dpo(tag, start, ref_name, n):
    E4P.SEED = SEED
    os.makedirs(f"{DATA}/subsets", exist_ok=True)
    data = f"{DATA}/subsets/{tag}.jsonl"
    _first_n(ref_path(ref_name), data, n)
    assert count_lines(data) == n, f"{data} has {count_lines(data)} pairs, expected {n}"
    adapter = f"{E5}/adapters/{tag}"
    t0 = time.time()
    m = E4P.train_dpo(start, data, adapter, tag=tag)
    t1 = time.time()
    E4P.bake(start, adapter, ckpt(tag))
    m.update({"bake_seconds": time.time() - t1, "job_wall_seconds": time.time() - t0, "budget": n,
              "ref_model": "B0" if ref_name == "b0" else "pre-DPO start", "seed": SEED})
    os.makedirs(METRICS, exist_ok=True)
    json.dump(m, open(f"{METRICS}/{tag}.json", "w"), indent=1)
    os.remove(data)
    mark_done(f"run_{tag}", {k: m[k] for k in ("pairs", "optimizer_steps", "tokens", "train_seconds")})


def job_repair(c, n, rm):
    _run_dpo(tag_rep(c, n, rm), ckpt(tag_pre(c)), c if rm == "pre_dpo" else "b0", n)


def job_control(n):
    _run_dpo(tag_ctl(n), MODEL_ID, "b0", n)


# ----------------------------------------------------------------------------- evaluation
def ctl_budgets():
    return [n for n in CFG["budgets"] if n < 2000] if CFG["controls"] else []


def eval_tags():
    tags = list(ANCHORS)
    for c in get_cases():
        tags.append(tag_pre(c))
        for rm in CFG["ref_modes"]:
            tags += [tag_rep(c, n, rm) for n in CFG["budgets"]]
    tags += [tag_ctl(n) for n in ctl_budgets()]
    return tags


def _deletable(tag):
    if tag.startswith("pre_"):
        c = tag[len("pre_"):]
        return all(is_done(f"run_{tag_rep(c, n, rm)}") for rm in CFG["ref_modes"] for n in CFG["budgets"])
    return tag.startswith(("rep_", "ctl_"))


def job_eval(tag, delete_after=False):
    os.makedirs(RESULTS, exist_ok=True)
    out = f"{RESULTS}/eval_{tag}.json"
    e4_res = f"outputs/e4/results/eval_{tag}.json"
    if os.path.exists(out):
        mark_done(f"eval_{tag}", {"reused": out})
        return
    if tag in ANCHORS and CFG["reuse_e4_anchors"] and os.path.exists(e4_res) and os.path.exists(E4P.B0_CACHE_500):
        shutil.copy(e4_res, out)
        mark_done(f"eval_{tag}", {"reused_from": e4_res})
        return
    from src.eval.eval_math import evaluate_svamp
    path = ckpt(tag)
    t0 = time.time()
    res = {"tag": tag, "model_path": path, "gpu": E4P.gpu_name(), "full_eval": bool(CFG["full_eval"])}
    sv = evaluate_svamp(path, 300, None, 16)
    res["svamp"] = {"acc": sv["accuracy"], "correct": [int(d["correct"]) for d in sv["details"]]}
    if CFG["full_eval"]:
        from src.eval.eval_code import evaluate_mbpp
        from src.eval.eval_ifeval import evaluate_ifeval
        cd = evaluate_mbpp(path, 257, None)
        res["mbpp"] = {"acc": cd["pass_at_1"], "passed": [int(d["passed"]) for d in cd["details"]]}
        ie = evaluate_ifeval(path, 300, None, 16)
        res["ifeval"] = {"prompt_acc": ie["prompt_accuracy"], "inst_acc": ie["instruction_accuracy"],
                         "prompt_pass": [int(d["prompt_all_passed"]) for d in ie["details"]]}
    E4P.free_gpu()
    res["pref"] = E4P.eval_pref(path)
    res["eval_seconds"] = time.time() - t0
    json.dump(res, open(out, "w"))
    mark_done(f"eval_{tag}", {"svamp": res["svamp"]["acc"], "margin_500": res["pref"]["margin_500"]})
    if delete_after and _deletable(tag) and path != MODEL_ID and tag not in ANCHORS:
        shutil.rmtree(path, ignore_errors=True)
        print(f"deleted {path}")


def job_b0_ref500():
    if os.path.exists(E4P.B0_CACHE_500):
        print("B0 500-pair reference cache already present")
        mark_done("b0_ref500")
        return
    E4P.job_b0_ref500()
    mark_done("b0_ref500")


def job_bake(name):
    E4P.job_bake_sft(name)
    mark_done(f"bake_{name}")


# ----------------------------------------------------------------------------- analysis helpers
def threshold_budget(points, thr):
    """points: [(budget, prr_point, prr_ci_low)] (fractions).  Returns smallest budget reaching thr by
    point estimate, by lower CI bound, and a log-linear interpolation of the point estimates."""
    pts = sorted(points)
    by_point = next((n for n, p, lo in pts if p >= thr), None)
    by_ci = next((n for n, p, lo in pts if lo >= thr), None)
    interp = None
    if pts and pts[0][1] >= thr:
        interp = f"<= {pts[0][0]}"
    else:
        for (n0, p0, _), (n1, p1, _) in zip(pts, pts[1:]):
            if p0 < thr <= p1:
                x = math.log10(n0) + (thr - p0) / (p1 - p0) * (math.log10(n1) - math.log10(n0))
                interp = f"~{10 ** x:.0f}"
                break
    return by_point, by_ci, interp


def _fmt_budget(x):
    return "not reached" if x is None else str(x)


def job_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = {t: json.load(open(f"{RESULTS}/eval_{t}.json")) for t in eval_tags()
         if os.path.exists(f"{RESULTS}/eval_{t}.json")}
    TM = {f[:-5]: json.load(open(f"{METRICS}/{f}")) for f in os.listdir(METRICS)} if os.path.exists(METRICS) else {}
    cases = get_cases()
    e2info = json.load(open(f"{E5}/cases.json")).get("e2", {})
    assert "b0" in R and "e_pref" in R, "need b0 and e_pref anchors evaluated"
    parent = R["e_pref"]["pref"]["margins"]
    S = lambda t: R[t]["svamp"]["acc"]
    cr_math = lambda t: (S(t) - S("b0")) / (S("e_math") - S("b0")) * 100 if "e_math" in R and S("e_math") > S("b0") else float("nan")
    full = CFG["full_eval"]

    hdr = ["Model", "Budget", "Steps", "Tokens", "DPO train s", "Pref margin (500)", "Raw win % (500)",
           "Margin (first 200)", "PRR vs E_pref [95% CI]", "Lost gain recovered", "SVAMP", "CR_math"]
    if full:
        hdr += ["MBPP", "IFEval"]
    lines = ["# Table E5: post-merge repair budget", "",
             f"Parent = E_pref (margin {R['e_pref']['pref']['margin_500']:+.4f} on 500 pairs). Seed {SEED}. "
             f"Reference model for repair DPO: {', '.join(CFG['ref_modes'])}.", "",
             "| " + " | ".join(hdr) + " |", "|" + " :--- |" * len(hdr)]
    out = {"rows": {}, "thresholds": {}, "config": CFG}
    curves = {}                                      # label -> list of (budget, prr, lo, hi, svamp)

    def row(label, tag, budget):
        if tag not in R:
            return None
        p = R[tag]["pref"]
        prr = E4P.prr_with_ci(p["margins"], parent) if tag != "b0" else (0.0, 0.0, 0.0)
        m = TM.get(tag)
        pre_prr = None
        lost = "-"
        cells = [label, str(budget), str(m["optimizer_steps"]) if m else "-", f"{m['tokens']:,}" if m else "-",
                 f"{m['train_seconds']:.0f}" if m else "-", f"{p['margin_500']:+.4f}", f"{p['raw_win_500']:.1f}",
                 f"{p['margin_200']:+.4f}", f"{prr[0] * 100:.1f}% [{prr[1] * 100:.1f}, {prr[2] * 100:.1f}]", "{LOST}",
                 f"{S(tag):.2f}", f"{cr_math(tag):.0f}%"]
        if full:
            cells += [f"{R[tag]['mbpp']['acc']:.2f}" if "mbpp" in R[tag] else "-",
                      f"{R[tag]['ifeval']['prompt_acc']:.2f}" if "ifeval" in R[tag] else "-"]
        out["rows"][tag] = {"budget": budget, "prr": prr, "margin_500": p["margin_500"], "svamp": S(tag),
                            "cr_math": cr_math(tag), "train": m and {k: m[k] for k in
                                                                      ("pairs", "optimizer_steps", "tokens", "train_seconds")}}
        return cells, prr

    for t, lab in (("b0", "B0"), ("e_math", "E_math"), ("e_pref", "E_pref (parent)")):
        r = row(lab, t, "-")
        if r:
            lines.append("| " + " | ".join(r[0]).replace("{LOST}", "-") + " |")

    for c in cases:
        info = e2info.get(c, {})
        lines.append(f"| **case {c}** (E2: PRR {info.get('prr_e2', float('nan')):+.1f}%, CR_math {info.get('cr_math_e2', float('nan')):+.0f}%) |" + " |" * (len(hdr) - 1))
        pre = row(f"pre-repair {c}", tag_pre(c), 0)
        pre_prr = pre[1][0] if pre else None
        if pre:
            lines.append("| " + " | ".join(pre[0]).replace("{LOST}", "-") + " |")
        for rm in CFG["ref_modes"]:
            label = f"{c}" + ("" if rm == "pre_dpo" else " [ref=B0]")
            pts = []
            if pre:
                curves.setdefault(label, []).append((0, pre[1][0], pre[1][1], pre[1][2], S(tag_pre(c))))
            for n in CFG["budgets"]:
                tg = tag_rep(c, n, rm)
                r = row(f"repair {label}", tg, n)
                if not r:
                    continue
                lost = f"{(r[1][0] - pre_prr) / (1 - pre_prr) * 100:.0f}%" if pre_prr is not None and pre_prr != 1 else "-"
                lines.append("| " + " | ".join(r[0]).replace("{LOST}", lost) + " |")
                pts.append((n, r[1][0], r[1][1]))
                curves.setdefault(label, []).append((n, r[1][0], r[1][1], r[1][2], S(tg)))
            for thr in CFG["thresholds"]:
                out["thresholds"][f"{label}@{thr}"] = threshold_budget(pts, thr)
    ctl = []
    for n in ctl_budgets():
        r = row("control: DPO from B0", tag_ctl(n), n)
        if r:
            lines.append("| " + " | ".join(r[0]).replace("{LOST}", "-") + " |")
            ctl.append((n, r[1][0], r[1][1], r[1][2], S(tag_ctl(n))))
    if ctl:
        curves["control (DPO from B0)"] = ctl

    lines += ["", "## Smallest repair budget reaching the target PRR", "",
              "| Case | Target | By point estimate | By 95% CI lower bound | Log-linear interpolation |",
              "|:---|:---|:---|:---|:---|"]
    for k, (bp, bc, ip) in out["thresholds"].items():
        label, thr = k.rsplit("@", 1)
        lines.append(f"| {label} | {float(thr) * 100:.0f}% | {_fmt_budget(bp)} | {_fmt_budget(bc)} | {ip or 'not reached'} |")

    lines += ["", "## Repaired merge vs DPO-from-B0 control at equal budget (paired bootstrap, 95% CI)", "",
              "| Case | Budget | Δ pref margin (repair − control) | Δ SVAMP (pts) |", "|:---|:---|:---|:---|"]
    for c in cases:
        for n in ctl_budgets():
            a, b = R.get(tag_rep(c, n, "pre_dpo")), R.get(tag_ctl(n))
            if a and b:
                dm = E4P.paired_boot(a["pref"]["margins"], b["pref"]["margins"])
                ds = E4P.paired_boot(a["svamp"]["correct"], b["svamp"]["correct"])
                lines.append(f"| {c} | {n} | {dm[0]:+.3f} [{dm[1]:+.3f}, {dm[2]:+.3f}] | "
                             f"{ds[0] * 100:+.1f} [{ds[1] * 100:+.1f}, {ds[2] * 100:+.1f}] |")
    md = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    open(f"{RESULTS}/table_e5.md", "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(f"{RESULTS}/table_e5.json", "w"), indent=1, default=str)
    print(md)

    # ---- Figure 4: recovery curve (+ capability panel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    xs0 = max(1, min(CFG["budgets"])) / 3.0          # position of the "0 (pre-repair)" point on the log axis
    for label, pts in curves.items():
        pts = sorted(pts)
        x = [xs0 if n == 0 else n for n, *_ in pts]
        y = [p[1] * 100 for p in pts]
        ls = "--" if label.startswith("control") else "-"
        a1.plot(x, y, ls, marker="o", label=label)
        if not label.startswith("control"):
            a1.fill_between(x, [p[2] * 100 for p in pts], [p[3] * 100 for p in pts], alpha=0.12)
        a2.plot(x, [p[4] for p in pts], ls, marker="o", label=label)
    for thr, st in zip(CFG["thresholds"], (":", "-.")):
        a1.axhline(thr * 100, color="gray", linestyle=st, linewidth=1, label=f"{thr * 100:.0f}% of parent")
    a1.axhline(100, color="purple", linestyle=":", linewidth=1.3, label="E_pref (parent)")
    a1.axhline(0, color="black", linestyle="--", linewidth=1, label="B0")
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([xs0] + CFG["budgets"])
        a.set_xticklabels(["0\n(pre)"] + [str(n) for n in CFG["budgets"]])
        a.minorticks_off()
        a.set_xlabel("Post-merge preference examples")
        a.grid(True, linestyle="--", alpha=0.5)
    a1.set_ylabel("PRR vs E_pref (%)")
    a1.set_title("(a) Preference recovery (95% CI over test pairs)")
    a1.legend(fontsize=8)
    if "e_math" in R:
        a2.axhline(S("e_math"), color="purple", linestyle=":", label="E_math")
        a2.axhline(S("b0"), color="black", linestyle="--", label="B0")
    a2.set_ylabel("SVAMP accuracy (%)")
    a2.set_title("(b) Math capability during repair")
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figure4_repair_curve.png", bbox_inches="tight")
    plt.close()
    print(f"saved {RESULTS}/figure4_repair_curve.png")
    mark_done("report")


def job_pack():
    import zipfile
    with zipfile.ZipFile("e5_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(E5):
            for f in fs:
                if f.startswith("ref_") or "subsets" in dp:
                    continue
                z.write(os.path.join(dp, f))
    print("wrote e5_artifacts.zip", f"{os.path.getsize('e5_artifacts.zip') / 1e6:.0f} MB")


# ----------------------------------------------------------------------------- registry / CLI
def jobs_by_stage():
    cases, B = get_cases(), CFG["budgets"]
    st = {
        "setup": ["select_cases", "pool" + sfx(), "bake_math", "bake_pref", "b0_ref500"],
        "anchors": [f"eval_{t}" for t in ANCHORS],
        "merge": [f"merge_{c}" for c in cases],
        "ref": ["ref_b0"] + [f"ref_{c}" for c in cases],
        "repair": [f"rep_{c}_{n}_{rm}" for c in cases for rm in CFG["ref_modes"] for n in B]
                  + [f"ctl_b0_{n}" for n in ctl_budgets()],
        "eval": [f"eval_{t}" for t in eval_tags() if t not in ANCHORS],
        "report": ["report"],
    }
    return st


def run_job(job, delete_after=False):
    if job == "pack":
        return job_pack()
    if is_done(job) and job != "report":
        print(f"[skip] {job} already done")
        return
    print(f"\n===== JOB {job} (seed {SEED}) =====", flush=True)
    if job == "select_cases":
        job_select_cases()
    elif job.startswith("pool"):
        job_pool()
    elif job.startswith("bake_"):
        job_bake(job[len("bake_"):])
    elif job == "b0_ref500":
        job_b0_ref500()
    elif job.startswith("merge_"):
        job_merge(job[len("merge_"):])
    elif job.startswith("ref_"):
        job_ref(job[len("ref_"):])
    elif job.startswith("rep_"):
        m = re.match(r"rep_(.+)_(\d+)_(pre_dpo|b0)$", job)
        job_repair(m.group(1), int(m.group(2)), m.group(3))
    elif job.startswith("ctl_b0_"):
        job_control(int(job[len("ctl_b0_"):]))
    elif job.startswith("eval_"):
        job_eval(job[len("eval_"):], delete_after)
    elif job == "report":
        job_report()
    else:
        raise ValueError(job)
    if not is_done(job):
        mark_done(job)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list", "status"])
    ap.add_argument("--job")
    ap.add_argument("--stage")
    ap.add_argument("--delete_after", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        st = jobs_by_stage()
        print("\n".join(st[a.stage] if a.stage else sum(st.values(), [])))
    elif a.cmd == "status":
        for stg, js in jobs_by_stage().items():
            for j in js:
                print(("DONE " if is_done(j) else "todo ") + f"[{stg}] {j}")
    else:
        run_job(a.job, a.delete_after)
