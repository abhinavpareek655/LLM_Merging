# Does Preference Alignment Survive the Merge?
## Preference Retention and Low-Cost Repair in Small Language Models (CS769 Project)

This repository investigates the behavior of preference-aligned language models under multi-task model merging. Specifically, we study:
1. **Specialist Domain Alignment**: Fine-tuning domain experts ($E_{\text{math}}$, $E_{\text{code}}$, $E_{\text{instr}}$) and a preference alignment anchor ($E_{\text{pref}}$) on clean base model $B_0$ (`Qwen/Qwen2.5-0.5B-Instruct`).
2. **Preference & Capability Retention**: Mapping continuous **Preference Retention Ratio (PRR)** and **Capability Retention (CR)** curves across multiple merge algorithms (Linear Averaging, Task Arithmetic, TIES, SLERP, DARE-TIES) across $\alpha_{\text{pref}} \in [0.0, 1.0]$.
3. **Parameter-Space Geometric Conflicts**: Quantifying cosine similarity, norm ratios, and sign conflict rates between preference task vectors and capability task vectors.

---

## 📊 Domain Specialists Performance vs. Baseline ($B_0$)

All SFT specialists were trained under a strictly uniform hyperparameter regimen (3.0 epochs, LoRA rank $r=16, \alpha=32$, learning rate $2.0\times 10^{-4}$, effective batch size 16) with native Qwen ChatML templates. Every domain specialist demonstrates a statistically verified advantage over the base model $B_0$:

| Specialist Model | Domain | Training Dataset | Epochs & Steps | Train Samples ($N_{\text{train}}$) | Evaluation Benchmark & Split | Test Samples ($N_{\text{test}}$) | Baseline $B_0$ Score (%) | Expert Score (%) | Advantage ($\Delta$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_{\text{math}}$** | **Mathematics** | `meta-math/MetaMathQA` | **3.0 Epochs** (939 steps) | 5,000 | **SVAMP** (Held-Out Test Split) | **300** | **53.33%** (160/300) | **58.67%** (176/300) | **+5.34%** (+16 solved) | **Specialist Advantage** |
| **$E_{\text{code}}$** | **Code Generation** | Python Instructions + MBPP Train ($\ge 601$) | **3.0 Epochs** (564 steps) | 3,000 | **MBPP Sanitized Test** (Tasks 11–510) | **257** | **31.13%** (80/257) | **38.91%** (100/257) | **+7.78%** (+20 solved) | **Specialist Advantage** |
| **$E_{\text{instr}}$** | **Instruction Following** | SmolTalk (`smol-constraints`) | **3.0 Epochs** (564 steps) | 3,000 | **IFEval** (Strict Prompt-Level) | **300** | **29.67%** (89/300) | **38.00%** (114/300) | **+8.33%** (+25 satisfied) | **Specialist Advantage** |
| | | | | | *IFEval (Instruction-Level)* | *456* | *36.62%* (167/456) | *45.18%* (206/456) | *+8.56%* (+39 satisfied) | *Specialist Advantage* |
| **$E_{\text{pref}}$** | **Safety & Preference** | `ultrafeedback_binarized` | **1.0 Epoch** (250 steps, DPO) | 2,000 | **UltraFeedback Test** (Held-Out Pairs) | **500** | **53.20%** | **64.20%** | **+11.00%** | **Specialist Advantage** |

> [!NOTE]
> - **Split Integrity**: Zero train-test leakage across all benchmarks. MBPP training strictly uses tasks $\ge 601$ while evaluation uses tasks 11–510. IFEval training data (`smol-constraints`) has 0 prompt overlap with the official IFEval test set ($0 / 541$).
> - **Retention Denominators**: Because all specialists beat baseline $B_0$, the Capability Retention denominator ($\text{Score}(E_k) - \text{Score}(B_0)$) remains strictly positive and well-defined across all merge evaluations.

---

## 📁 Repository Structure

```text
CS769/
├── configs/
│   ├── base_config.py          # Central model constants (Qwen2.5-0.5B-Instruct, LoRA r=16, α=32, target modules)
│   ├── sft_config.yaml         # SFT training hyperparameters (lr=2e-4, 3 epochs, batch_size=16)
│   └── dpo_config.yaml         # DPO training hyperparameters (lr=5e-5, β=0.1, 1 epoch)
├── src/
│   ├── data/
│   │   ├── prepare_data.py            # Streamlined dataset prep (MetaMathQA, Python Code, SmolTalk, UltraFeedback)
│   │   └── precompute_ref_logps.py    # Precomputes offline reference log-probabilities for DPO
│   ├── train/
│   │   ├── train_sft.py               # SFT LoRA fine-tuning for domain specialists
│   │   └── train_dpo_stable.py        # Stable DPO training loop with precomputed reference logps
│   ├── eval/
│   │   ├── eval_math.py               # SVAMP & GSM8K exact-match extraction and accuracy
│   │   ├── eval_code.py               # MBPP Pass@1 evaluation with batched inference & execution sandbox
│   │   ├── eval_ifeval.py             # IFEval constraint satisfaction checker (batched GPU inference)
│   │   ├── eval_pref.py               # UltraFeedback win rate and sequence log-probability evaluator
│   │   └── run_table1.py              # Master evaluation runner for Table 1 benchmarks
│   ├── diagnostics/
│   │   ├── layer_conflicts.py         # Norm ratios, cosine similarity, and top-20% sign conflict calculator
│   │   └── run_all_diagnostics.py     # Batch runner across all pairwise expert combinations
│   └── merge/
│       ├── merge_methods.py           # Core merging algorithms: Linear, Task Arithmetic, TIES, SLERP, DARE
│       ├── run_e2_sweep.py            # Automated grid sweeps across α_pref ∈ [0.0, 1.0] and densities
│       ├── eval_merged.py             # Evaluation harness for merged checkpoints
│       ├── bake_adapters.py           # Bakes LoRA adapters into standalone base model checkpoints
│       └── plot_figure2.py            # Publication-grade figures (Figures 2 & 3: Retention curves)
├── outputs/
│   ├── adapters/                      # Lightweight LoRA adapters (~34MB each, tracked in Git)
│   │   ├── e_math/                    # Trained on MetaMathQA (5k samples, 3 epochs)
│   │   ├── e_code/                    # Trained on Python Instructions (3k samples, 3 epochs)
│   │   ├── e_instr/                   # Trained on SmolTalk constraints (3k samples, 3 epochs)
│   │   └── e_pref/                    # Trained on UltraFeedback (2k pairs, 1 epoch, DPO)
│   ├── datasets/                      # Filtered training and evaluation JSONL files
│   ├── diagnostics/                   # Parameter conflict JSON reports (pref_vs_math.json, etc.)
│   └── results/                       # Authoritative benchmark JSON logs, tables, and figures
├── requirements.txt                   # Environment dependencies
├── run_e2_colab.ipynb                 # Google Colab notebook for running sweeps on cloud T4/A100 GPUs
└── README.md                          # Project documentation and quickstart guide
```

---

## 🚀 Quickstart for Team Members

### 1. Environment Setup
```bash
git clone <repo-url>
cd CS769
pip install -r requirements.txt
```

### 2. Working with the 4 Pre-Trained Adapters
All 4 specialized adapters are pre-trained and saved in `outputs/adapters/`:
- `outputs/adapters/e_math` (Math Reasoning)
- `outputs/adapters/e_code` (Python Code Generation)
- `outputs/adapters/e_instr` (Complex Instruction Compliance)
- `outputs/adapters/e_pref` (Helpful/Harmless Preference Anchor)

To bake any adapter into a standalone full checkpoint (`merge_and_unload` in `bfloat16`):
```bash
python src/merge/bake_adapters.py --adapter_path outputs/adapters/e_math --output_path outputs/checkpoints/e_math
```

### 3. Merging Experts (Experiment 2 & 3)
Import and run any merge algorithm directly in Python:
```python
from src.merge.merge_methods import merge_linear, merge_slerp, merge_ties, merge_dare_ties

# Example: TIES-merge between e_pref, e_math, e_code, e_instr
merged_model = merge_ties(
    base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_paths=[
        "outputs/adapters/e_pref",
        "outputs/adapters/e_math",
        "outputs/adapters/e_code",
        "outputs/adapters/e_instr"
    ],
    weights=[0.4, 0.2, 0.2, 0.2],  # α_pref = 0.4
    density=0.5
)
```
Or execute the automated retention sweep across $\alpha \in [0.0, 1.0]$:
```bash
python src/merge/run_e2_sweep.py
```

### 4. Running Benchmarks Locally
All evaluations run with native PyTorch SDPA on GPU:
- **Math (SVAMP)**:
  ```bash
  python src/eval/eval_math.py --model_path outputs/checkpoints/e_math --dataset svamp --max_samples 300
  ```
- **Code (MBPP Pass@1 on 257 sanitized problems)**:
  ```bash
  python src/eval/eval_code.py --model_path outputs/checkpoints/e_code --max_samples 257
  ```
- **Instruction Following (IFEval on 300 prompts)**:
  ```bash
  python src/eval/eval_ifeval.py --model_path outputs/checkpoints/e_instr --max_samples 300 --batch_size 16
  ```
- **Preference Win Rate (UltraFeedback on 500 pairs)**:
  ```bash
  python src/eval/eval_pref.py --model_path outputs/checkpoints/e_pref --max_samples 500
  ```
- **Parameter Conflict Diagnostics**:
  ```bash
  python src/diagnostics/layer_conflicts.py --adapter_a outputs/adapters/e_pref --adapter_b outputs/adapters/e_math
  ```

---

## 📈 Key Findings & Diagnostic Insights

- **Parameter Magnitude Discrepancy**:
  - $\|\Delta\theta_{\text{pref}}\|_2 = 1.2365$ vs. $\|\Delta\theta_{\text{math}}\|_2 = 9.2217$ (Norm ratio: **$0.1341$**).
  - SFT updates are **$7.46\times$ larger** than DPO updates, explaining why naive linear averaging rapidly erodes preference alignment unless protected by TIES sign agreement or density filtering.
- **Subspace Orthogonality**:
  - Cosine similarity across global task vectors is **$-5.45 \times 10^{-7} \approx 0.0$** (near-perfect orthogonality).
- **Localized Interference**:
  - Despite global orthogonality, the sign conflict rate across the top-20% high-magnitude parameters reaches **$32.07\%$**, causing destructive localized cancellation during arithmetic addition.
