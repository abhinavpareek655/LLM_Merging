"""
Driver script to compute all pairwise conflict diagnostic maps involving E_pref.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import gc
import torch
torch.set_num_threads(2)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, OUTPUT_CHECKPOINTS_DIR
from src.diagnostics.layer_conflicts import analyze_conflicts

def main():
    diag_dir = "outputs/diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    
    e_pref_path = os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_pref")
    experts = [
        ("math", os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_math")),
        ("code", os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_code")),
        ("instr", os.path.join(OUTPUT_CHECKPOINTS_DIR, "e_instr")),
    ]
    
    for name, path in experts:
        out_json = os.path.join(diag_dir, f"pref_vs_{name}.json")
        print(f"\n{'='*70}\nRunning conflict analysis: E_pref vs E_{name}\n{'='*70}")
        analyze_conflicts(
            base_model_path=MODEL_ID,
            expert1_path=e_pref_path,
            expert2_path=path,
            name1="E_pref",
            name2=f"E_{name}",
            output_json=out_json
        )
        gc.collect()

if __name__ == "__main__":
    main()
