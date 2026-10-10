"""
Master Evaluation Runner for Experiment 1 (E1).
Evaluates B0, E_math, E_code, E_instr, and E_pref across:
- GSM8K (Exact Match)
- MBPP (Pass@1 with 3-second timeout)
- IFEval (Strict Constraint Accuracy)
- UltraFeedback Held-Out (Raw Win Rate & Implicit DPO Margin vs B0)

Formats results into a publication-ready Table 1 (Markdown and LaTeX).
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import gc
import argparse
import json
import torch
torch.set_num_threads(2)
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import (
    MODEL_ID,
    OUTPUT_CHECKPOINTS_DIR,
    OUTPUT_DATA_DIR,
    OUTPUT_RESULTS_DIR
)
from src.eval.eval_math import evaluate_gsm8k
from src.eval.eval_code import evaluate_mbpp
from src.eval.eval_ifeval import evaluate_ifeval
from src.eval.eval_pref import evaluate_preference


def should_recompute(json_path: str, target_samples: int, force: bool = False) -> bool:
    if force or not os.path.exists(json_path):
        return True
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        count = data.get("total", data.get("total_evaluated", data.get("total_pairs", len(data.get("results", data.get("details", []))))))
        if target_samples and count < target_samples:
            return True
        return False
    except Exception:
        return True


def run_table1(
    models: dict[str, str],
    max_eval_samples: int = None,
    output_dir: str = OUTPUT_RESULTS_DIR,
    force: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    pref_test_path = os.path.join(OUTPUT_DATA_DIR, "e_pref_test.jsonl")
    ref_cache_path = os.path.join(output_dir, "b0_pref_logps_cache.json")

    print("\n" + "="*80)
    print("STEP 1: PRE-COMPUTING / LOADING B0 PREFERENCE REFERENCE LOGPS")
    print("="*80)
    b0_path = models.get("B0", MODEL_ID)
    b0_pref_json = os.path.join(output_dir, "b0_pref_eval.json")
    if should_recompute(b0_pref_json, max_eval_samples, force):
        evaluate_preference(
            model_path=b0_path,
            test_dataset_path=pref_test_path,
            ref_logps_cache_path=ref_cache_path,
            max_samples=max_eval_samples,
            output_json=b0_pref_json
        )

    table_data = []

    for name, path in models.items():
        print("\n" + "#"*80)
        print(f"EVALUATING MODEL: {name} ({path})")
        print("#"*80)

        # 1. Math (GSM8K)
        print(f"\n--- [1/4] GSM8K Evaluation for {name} ---")
        math_json = os.path.join(output_dir, f"{name}_gsm8k.json")
        if not should_recompute(math_json, max_eval_samples, force):
            with open(math_json, "r") as f:
                math_res = json.load(f)
            print(f"Loaded existing results from {math_json}")
        else:
            math_res = evaluate_gsm8k(
                model_path=path,
                max_samples=max_eval_samples,
                output_json=math_json
            )
        gsm8k_acc = math_res.get("exact_match_accuracy", 0.0)

        # 2. Code (MBPP)
        print(f"\n--- [2/4] MBPP Evaluation for {name} ---")
        code_json = os.path.join(output_dir, f"{name}_mbpp.json")
        if not should_recompute(code_json, max_eval_samples, force):
            with open(code_json, "r") as f:
                code_res = json.load(f)
            print(f"Loaded existing results from {code_json}")
        else:
            code_res = evaluate_mbpp(
                model_path=path,
                max_samples=max_eval_samples,
                output_json=code_json
            )
        mbpp_acc = code_res.get("pass_at_1", 0.0)

        # 3. Instruction Following (IFEval)
        print(f"\n--- [3/4] IFEval Evaluation for {name} ---")
        ifeval_json = os.path.join(output_dir, f"{name}_ifeval.json")
        if not should_recompute(ifeval_json, max_eval_samples, force):
            with open(ifeval_json, "r") as f:
                ifeval_res = json.load(f)
            print(f"Loaded existing results from {ifeval_json}")
        else:
            ifeval_res = evaluate_ifeval(
                model_path=path,
                max_samples=max_eval_samples,
                output_json=ifeval_json
            )
        ifeval_acc = ifeval_res.get("prompt_accuracy", 0.0)

        # 4. Preference (Held-Out UltraFeedback)
        print(f"\n--- [4/4] Preference Evaluation for {name} ---")
        pref_json = os.path.join(output_dir, f"{name}_pref.json")
        if not should_recompute(pref_json, max_eval_samples, force):
            with open(pref_json, "r") as f:
                pref_res = json.load(f)
            print(f"Loaded existing results from {pref_json}")
        else:
            pref_res = evaluate_preference(
                model_path=path,
                test_dataset_path=pref_test_path,
                ref_logps_cache_path=ref_cache_path,
                max_samples=max_eval_samples,
                output_json=pref_json
            )
        raw_pref_win = pref_res.get("raw_accuracy", 0.0)
        dpo_margin = pref_res.get("dpo_implicit_margin", 0.0)

        row = {
            "Model": name,
            "GSM8K (EM %)": round(gsm8k_acc, 2),
            "MBPP (Pass@1 %)": round(mbpp_acc, 2),
            "IFEval (Strict %)": round(ifeval_acc, 2),
            "Raw Pref Win %": round(raw_pref_win, 2),
            "DPO Margin (vs B0)": round(dpo_margin, 4)
        }
        table_data.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(table_data)

    print("\n" + "="*80)
    print("EXPERIMENT 1 (E1): SANITY CHECK TABLE 1")
    print("="*80)
    print(df.to_markdown(index=False))

    # Save Markdown
    md_path = os.path.join(output_dir, "table1.md")
    with open(md_path, "w") as f:
        f.write("# Table 1: Specialist Sanity Checks Before Model Merging\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    # Save LaTeX
    tex_path = os.path.join(output_dir, "table1.tex")
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False))

    # Save JSON
    json_path = os.path.join(output_dir, "table1.json")
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nTable 1 artifacts saved to:\n- {md_path}\n- {tex_path}\n- {json_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Table 1 comprehensive evaluation.")
    parser.add_argument("--dev", action="store_true", help="Run in fast dev mode (200 samples per benchmark)")
    parser.add_argument("--max_samples", type=int, default=None, help="Explicit max samples per benchmark")
    parser.add_argument("--eval_b0_only", action="store_true", help="Evaluate only B0 baseline")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of benchmarks")
    args = parser.parse_args()

    max_samples = 200 if args.dev else args.max_samples

    models_to_eval = {
        "B0": MODEL_ID,
    }

    if not args.eval_b0_only:
        models_to_eval.update({
            "E_math": os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_math"),
            "E_code": os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_code"),
            "E_instr": os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_instr"),
            "E_pref": os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_pref"),
        })

    run_table1(models_to_eval, max_eval_samples=max_samples, force=args.force)
