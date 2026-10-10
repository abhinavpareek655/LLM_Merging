"""
Comprehensive Model Merging Engine for Experiment 2 (E2).
Implements:
1. Linear Task Arithmetic (Naive Unscaled Baseline)
2. Linear Task Arithmetic (Norm-Balanced Variant)
3. SLERP (Spherical Linear Interpolation)
4. TIES-Merging (Trim, Elect Signs, Disjoint Merge with coordinate logging)
5. DARE-TIES (Random Drop + Rescale + TIES)
"""
import os
import sys
import copy
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE

def load_state_dict_cpu(model_path: str):
    """Loads model state dict into CPU memory in bfloat16."""
    print(f"Loading state dict on CPU: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    return sd

def slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical Linear Interpolation between two tensors."""
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    n0 = torch.norm(v0_flat)
    n1 = torch.norm(v1_flat)

    if n0 < eps or n1 < eps:
        return ((1.0 - t) * v0_flat + t * v1_flat).view(orig_shape).to(v0.dtype)

    u0 = v0_flat / n0
    u1 = v1_flat / n1

    dot = torch.sum(u0 * u1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # If collinear, standard linear interpolation
    if torch.abs(dot) > (1.0 - 1e-4):
        res = (1.0 - t) * v0_flat + t * v1_flat
        return res.view(orig_shape).to(v0.dtype)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    c0 = torch.sin((1.0 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega

    u_interp = c0 * u0 + c1 * u1
    n_interp = (1.0 - t) * n0 + t * n1

    res = n_interp * (u_interp / torch.norm(u_interp))
    return res.view(orig_shape).to(v0.dtype)

def ties_merge_tensors(
    tau_list: list[torch.Tensor],
    density: float = 0.20,
    alpha: float = 0.50
) -> tuple[torch.Tensor, dict]:
    """
    TIES-Merging on a coordinate list of task vectors [tau_math, tau_pref].
    Step 1: Trim top-density parameters
    Step 2: Elect sign consensus (logs coordinate clashes)
    Step 3: Disjoint merge
    """
    stats = {
        "total_coords": tau_list[0].numel(),
        "both_active": 0,
        "sign_agreed": 0,
        "sign_clashed": 0,
        "math_won_sign": 0,
        "pref_won_sign": 0
    }

    trimmed = []
    masks = []
    for tau in tau_list:
        flat = tau.flatten().float()
        k = int(density * flat.numel())
        if k > 0:
            topk_vals, _ = torch.topk(torch.abs(flat), k)
            thresh = topk_vals[-1]
            mask = torch.abs(flat) >= thresh
        else:
            mask = torch.zeros_like(flat, dtype=torch.bool)
        trimmed.append(flat * mask)
        masks.append(mask)

    t_math, t_pref = trimmed[0], trimmed[1]
    m_math, m_pref = masks[0], masks[1]

    # Sign consensus: sum of trimmed updates
    sum_tau = t_math + t_pref
    consensus_sign = torch.sign(sum_tau)

    # Detailed clash logging
    active_both = m_math & m_pref
    stats["both_active"] = int(active_both.sum().item())

    if stats["both_active"] > 0:
        s_math = torch.sign(t_math[active_both])
        s_pref = torch.sign(t_pref[active_both])
        s_cons = consensus_sign[active_both]

        clash = (s_math != s_pref) & (s_math != 0) & (s_pref != 0)
        agree = (s_math == s_pref) & (s_math != 0)

        stats["sign_agreed"] = int(agree.sum().item())
        stats["sign_clashed"] = int(clash.sum().item())

        math_won = (s_cons[clash] == s_math[clash])
        pref_won = (s_cons[clash] == s_pref[clash])

        stats["math_won_sign"] = int(math_won.sum().item())
        stats["pref_won_sign"] = int(pref_won.sum().item())

    # Disjoint merge: filter by consensus sign
    merged_flat = torch.zeros_like(t_math)
    count_agree = torch.zeros_like(t_math)

    for t in [t_math, t_pref]:
        t_sign = torch.sign(t)
        agree_mask = (t_sign == consensus_sign) & (consensus_sign != 0)
        merged_flat += t * agree_mask
        count_agree += agree_mask.float()

    valid_mask = count_agree > 0
    merged_flat[valid_mask] = merged_flat[valid_mask] / count_agree[valid_mask]

    res = (alpha * merged_flat).view(tau_list[0].shape).to(tau_list[0].dtype)
    return res, stats

def dare_tensors(tau: torch.Tensor, drop_rate: float = 0.50) -> torch.Tensor:
    """DARE: Randomly drops parameters and rescales survivors."""
    if drop_rate <= 0.0:
        return tau
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(tau.float(), keep_prob)).to(torch.bool)
    res = (tau.float() * mask) / keep_prob
    return res.to(tau.dtype)

def merge_checkpoints(
    method: str,
    output_dir: str,
    base_sd: dict,
    math_sd: dict,
    pref_sd: dict,
    alpha_math: float = 0.50,
    alpha_pref: float = 0.50,
    t_slerp: float = 0.50,
    density_ties: float = 0.20,
    drop_dare: float = 0.50
):
    print(f"\n{'='*70}\nEXECUTING MERGE: {method.upper()} -> {output_dir}\n{'='*70}")
    merged_sd = {}
    aggregate_stats = {
        "method": method,
        "total_params": 0,
        "both_active": 0,
        "sign_agreed": 0,
        "sign_clashed": 0,
        "math_won_sign": 0,
        "pref_won_sign": 0
    }

    for k in base_sd.keys():
        b_val = base_sd[k]
        m_val = math_sd[k]
        p_val = pref_sd[k]

        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            merged_sd[k] = b_val.clone()
            continue

        tau_m = m_val - b_val
        tau_p = p_val - b_val
        aggregate_stats["total_params"] += b_val.numel()

        if method == "linear":
            # Pure naive unscaled linear merge
            tau_merged = alpha_math * tau_m + alpha_pref * tau_p
            merged_sd[k] = b_val + tau_merged

        elif method == "linear_balanced":
            # Norm-balanced linear merge
            norm_m = torch.norm(tau_m.float()) + 1e-8
            norm_p = torch.norm(tau_p.float()) + 1e-8
            target_norm = (norm_m + norm_p) / 2.0
            tau_m_bal = (tau_m.float() / norm_m) * target_norm
            tau_p_bal = (tau_p.float() / norm_p) * target_norm
            tau_merged = (alpha_math * tau_m_bal + alpha_pref * tau_p_bal).to(b_val.dtype)
            merged_sd[k] = b_val + tau_merged

        elif method == "slerp":
            # Standard SLERP directly between full model weights (matching MergeKit)
            merged_sd[k] = slerp_tensor(m_val, p_val, t=t_slerp)

        elif method == "ties":
            tau_merged, stats = ties_merge_tensors([tau_m, tau_p], density=density_ties, alpha=alpha_math)
            merged_sd[k] = b_val + tau_merged
            for stat_k in ["both_active", "sign_agreed", "sign_clashed", "math_won_sign", "pref_won_sign"]:
                aggregate_stats[stat_k] += stats[stat_k]

        elif method == "dare_ties":
            tau_m_dare = dare_tensors(tau_m, drop_rate=drop_dare)
            tau_p_dare = dare_tensors(tau_p, drop_rate=drop_dare)
            tau_merged, stats = ties_merge_tensors([tau_m_dare, tau_p_dare], density=density_ties, alpha=alpha_math)
            merged_sd[k] = b_val + tau_merged
            for stat_k in ["both_active", "sign_agreed", "sign_clashed", "math_won_sign", "pref_won_sign"]:
                aggregate_stats[stat_k] += stats[stat_k]

        else:
            raise ValueError(f"Unknown merge method: {method}")

    if method in ["ties", "dare_ties"] and aggregate_stats["sign_clashed"] > 0:
        c = aggregate_stats["sign_clashed"]
        mw = aggregate_stats["math_won_sign"]
        pw = aggregate_stats["pref_won_sign"]
        print("\n--- TIES Sign Election Analysis ---")
        print(f"Total Conflicting Coordinates: {c:,}")
        print(f"Math Won Sign Consensus      : {mw:,} ({(mw/c)*100:.2f}%)")
        print(f"Pref Won Sign Consensus      : {pw:,} ({(pw/c)*100:.2f}%)")
        print("-----------------------------------")

    # Save merged model to disk
    print(f"Saving merged weights to: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model skeleton on CPU to save complete HuggingFace repo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.load_state_dict(merged_sd)
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(output_dir)

    del model
    del merged_sd

    # Save merge metadata
    meta_path = os.path.join(output_dir, "merge_meta.json")
    with open(meta_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Successfully saved {method} model to {output_dir}")
    return aggregate_stats

def run_phase_2a_merges():
    """Generates the 4 canonical merged checkpoints for Phase 2A (Midpoint Sanity Check)."""
    base_sd = load_state_dict_cpu(MODEL_ID)
    math_sd = load_state_dict_cpu("outputs/checkpoints/e_math")
    pref_sd = load_state_dict_cpu("outputs/checkpoints/e_pref")

    merges = [
        ("linear", "outputs/checkpoints/merge_linear_naive"),
        ("linear_balanced", "outputs/checkpoints/merge_linear_balanced"),
        ("slerp", "outputs/checkpoints/merge_slerp"),
        ("ties", "outputs/checkpoints/merge_ties"),
        ("dare_ties", "outputs/checkpoints/merge_dare_ties"),
    ]

    all_stats = {}
    for method, out_path in merges:
        stats = merge_checkpoints(
            method=method,
            output_dir=out_path,
            base_sd=base_sd,
            math_sd=math_sd,
            pref_sd=pref_sd,
            alpha_math=0.50,
            alpha_pref=0.50,
            t_slerp=0.50,
            density_ties=0.20,
            drop_dare=0.50
        )
        all_stats[method] = stats

    stats_out = "outputs/results/phase2a_merge_stats.json"
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    with open(stats_out, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAll Phase 2A merges created successfully! Metadata saved to {stats_out}")

if __name__ == "__main__":
    run_phase_2a_merges()
