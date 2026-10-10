"""
Layer-by-Layer and Global Parameter Conflict Diagnostic Engine.
Analyzes cosine similarity, sign conflict rate (joint top-20%), and norm ratios
between two fine-tuned expert checkpoints relative to a common base model.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import argparse
import json
import re
import torch
torch.set_num_threads(2)
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE, LORA_TARGET_MODULES


def is_adapted_linear(param_name: str) -> bool:
    """Strictly checks if a parameter belongs to an adapted linear layer.
    Excludes layernorms, embeddings, and lm_head to prevent 0/0 NaN division.
    """
    return any(target in param_name for target in LORA_TARGET_MODULES) and "weight" in param_name


def get_submodule_type(param_name: str) -> str:
    """Classifies parameter into attention or mlp subspace."""
    if any(k in param_name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    elif any(k in param_name for k in ["gate_proj", "up_proj", "down_proj"]):
        return "mlp"
    return "other"


def extract_layer_idx(param_name: str) -> int:
    """Extracts transformer block index (e.g. model.layers.5 -> 5)."""
    match = re.search(r"layers\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))
    return -1


def compute_metrics(delta1: torch.Tensor, delta2: torch.Tensor, top_k_ratio: float = 0.20):
    """Computes cosine similarity, sign conflict rate over joint top-k% magnitude, and norm ratio."""
    d1 = delta1.float().view(-1)
    d2 = delta2.float().view(-1)

    norm1 = torch.norm(d1, p=2).item()
    norm2 = torch.norm(d2, p=2).item()

    # Cosine similarity (protected against div-by-zero)
    if norm1 < 1e-12 or norm2 < 1e-12:
        cosine = 0.0
    else:
        cosine = (torch.dot(d1, d2) / (norm1 * norm2)).item()

    # Joint magnitude top-k% sign conflict
    joint_mag = torch.abs(d1) + torch.abs(d2)
    k_elements = max(1, int(len(joint_mag) * top_k_ratio))
    topk_vals, topk_indices = torch.topk(joint_mag, k_elements)

    d1_topk = d1[topk_indices]
    d2_topk = d2[topk_indices]

    sign1 = torch.sign(d1_topk)
    sign2 = torch.sign(d2_topk)
    # A sign conflict occurs when both are non-zero and push in opposite directions
    conflicts = (sign1 * sign2 < 0).float()
    scr = conflicts.mean().item()

    norm_ratio = norm1 / (norm2 + 1e-12)

    return {
        "cosine_similarity": cosine,
        "sign_conflict_rate": scr,
        "norm_ratio": norm_ratio,
        "norm1": norm1,
        "norm2": norm2,
        "num_params": len(d1)
    }


def analyze_conflicts(
    base_model_path: str,
    expert1_path: str,
    expert2_path: str,
    name1: str = "expert1",
    name2: str = "expert2",
    output_json: str = None
):
    print(f"Loading Base model: {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=TORCH_DTYPE, device_map="cpu")
    base_sd = base_model.state_dict()

    print(f"Loading {name1}: {expert1_path}...")
    exp1_model = AutoModelForCausalLM.from_pretrained(expert1_path, torch_dtype=TORCH_DTYPE, device_map="cpu")
    exp1_sd = exp1_model.state_dict()

    print(f"Loading {name2}: {expert2_path}...")
    exp2_model = AutoModelForCausalLM.from_pretrained(expert2_path, torch_dtype=TORCH_DTYPE, device_map="cpu")
    exp2_sd = exp2_model.state_dict()

    # Group deltas
    global_d1, global_d2 = [], []
    layer_deltas = {}  # layer_idx -> { 'attention': (d1, d2), 'mlp': (d1, d2), 'all': (d1, d2) }

    for key in base_sd.keys():
        if not is_adapted_linear(key):
            continue

        d1 = exp1_sd[key] - base_sd[key]
        d2 = exp2_sd[key] - base_sd[key]

        global_d1.append(d1.view(-1))
        global_d2.append(d2.view(-1))

        layer_idx = extract_layer_idx(key)
        if layer_idx == -1:
            continue

        subtype = get_submodule_type(key)

        if layer_idx not in layer_deltas:
            layer_deltas[layer_idx] = {
                "attention": {"d1": [], "d2": []},
                "mlp": {"d1": [], "d2": []},
                "all": {"d1": [], "d2": []}
            }

        layer_deltas[layer_idx][subtype]["d1"].append(d1.view(-1))
        layer_deltas[layer_idx][subtype]["d2"].append(d2.view(-1))
        layer_deltas[layer_idx]["all"]["d1"].append(d1.view(-1))
        layer_deltas[layer_idx]["all"]["d2"].append(d2.view(-1))

    # Compute Global metrics
    print("\n" + "="*80)
    print(f"GLOBAL CONFLICT ANALYSIS: {name1} vs {name2}")
    print("="*80)
    flat_global_d1 = torch.cat(global_d1)
    flat_global_d2 = torch.cat(global_d2)
    global_results = compute_metrics(flat_global_d1, flat_global_d2)
    print(f"Global Cosine Similarity : {global_results['cosine_similarity']:.4f}")
    print(f"Global Sign Conflict Rate: {global_results['sign_conflict_rate']*100:.2f}% (Joint Top-20% Mag)")
    print(f"Global Delta Norm Ratio  : {global_results['norm_ratio']:.4f} (||{name1}|| / ||{name2}||)")
    print(f"Total Analyzed Parameters: {global_results['num_params']:,}")

    # Compute Layer-by-Layer metrics
    layer_results = {}
    print("\n" + "="*80)
    print(f"{'Layer':<6} | {'Subtype':<10} | {'Cosine Sim':<12} | {'Sign Conflict %':<16} | {'Norm Ratio':<12}")
    print("-" * 80)

    for l in sorted(layer_deltas.keys()):
        layer_results[l] = {}
        for sub in ["attention", "mlp", "all"]:
            d1_cat = torch.cat(layer_deltas[l][sub]["d1"])
            d2_cat = torch.cat(layer_deltas[l][sub]["d2"])
            res = compute_metrics(d1_cat, d2_cat)
            layer_results[l][sub] = res
            print(f"L{l:<5} | {sub:<10} | {res['cosine_similarity']:<12.4f} | {res['sign_conflict_rate']*100:<15.2f}% | {res['norm_ratio']:<12.4f}")

    full_output = {
        "model_base": base_model_path,
        "expert1": {"name": name1, "path": expert1_path},
        "expert2": {"name": name2, "path": expert2_path},
        "global": global_results,
        "layer_wise": layer_results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved conflict diagnostic results to: {output_json}")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter interference between experts.")
    parser.add_argument("--base_model", type=str, default=MODEL_ID, help="Base model path")
    parser.add_argument("--expert1", type=str, required=True, help="First expert checkpoint path")
    parser.add_argument("--expert2", type=str, required=True, help="Second expert checkpoint path")
    parser.add_argument("--name1", type=str, default="pref_expert", help="Name of expert 1")
    parser.add_argument("--name2", type=str, default="domain_expert", help="Name of expert 2")
    parser.add_argument("--output_json", type=str, default=None, help="JSON output file path")
    args = parser.parse_args()

    analyze_conflicts(
        base_model_path=args.base_model,
        expert1_path=args.expert1,
        expert2_path=args.expert2,
        name1=args.name1,
        name2=args.name2,
        output_json=args.output_json
    )
