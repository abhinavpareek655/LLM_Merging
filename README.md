# Does Preference Alignment Survive the Merge?
## Preference Retention and Low-Cost Repair in Small Language Models (CS769 Project)

This repository benchmarks how preference alignment (DPO) behaves when merged into multi-task language model consortiums, maps out continuous **Preference Retention Ratio (PRR)** and **Capability Retention (CR)** curves across multiple merge algorithms, and evaluates low-cost post-merge repair strategies.

---

## 📁 Repository Structure

```text
CS769/
├── configs/
│   ├── base_config.py          # Central model constants (Qwen2.5-0.5B-Instruct, LoRA r=16, α=32, target modules)
│   ├── sft_config.yaml         # SFT training hyperparameters (lr=2e-4, 3 epochs)
│   └── dpo_config.yaml         # DPO training hyperparameters (lr=5e-5, β=0.1, 1 epoch)
├── src/
│   ├── data/
│   │   ├── prepare_data.py            # Streamlined dataset prep (MetaMath, Code, NoRobots, UltraFeedback)
│   │   └── precompute_ref_logps.py    # Offline reference log-probabilities for DPO stability
│   ├── train/
│   │   ├── train_sft.py               # SFT LoRA fine-tuning for domain specialists
│   │   └── train_dpo_stable.py        # Rock-solid DPO training loop with precomputed reference logps
│   ├── eval/
│   │   ├── eval_math.py               # SVAMP & GSM8K exact-match extraction and accuracy
│   │   ├── eval_code.py               # MBPP Pass@1 evaluation with execution sandbox timeout
│   │   ├── eval_ifeval.py             # IFEval constraint satisfaction checker
│   │   ├── eval_pref.py               # Sequence log-probability margin and DPO win rate evaluator
│   │   └── run_table1.py              # Specialist sanity check runner (Table 1 generation)
│   ├── diagnostics/
│   │   └── layer_conflicts.py         # Angle, norm ratio, cosine, and top-20% sign conflict calculator
│   └── merge/
│       ├── merge_methods.py           # Core merging implementations: Linear, SLERP, TIES, DARE-TIES
│       ├── run_e2_sweep.py            # Automated 44-point E2 sweep across α ∈ [0.0, 1.0]
│       ├── eval_merged.py             # Dual-benchmark evaluation harness for merged checkpoints
│       ├── bake_adapters.py           # Unloads LoRA adapters into standalone base model checkpoints
│       └── plot_figure2.py            # Matplotlib renderer for publication Figure 2 and Figure 3
├── outputs/
│   ├── adapters/                      # Trained LoRA adapters (<35MB each, trackable)
│   │   ├── e_math/                    # Trained on MetaMathQA (5k samples, 3 epochs)
│   │   ├── e_code/                    # Trained on CodeFeedback (5k samples, 3 epochs)
│   │   ├── e_instr/                   # Trained on NoRobots (5k samples, 3 epochs)
│   │   └── e_pref/                    # Trained on UltraFeedback (2k pairs, 1 epoch, β=0.1)
│   ├── datasets/                      # Filtered training and evaluation JSONL files
│   ├── diagnostics/                   # Parameter conflict JSON reports (pref_vs_math.json)
│   └── results/                       # Sweep JSON results, tables (MD/LaTeX), and publication figures
├── requirements.txt                   # Environment dependencies
├── run_e2_colab.ipynb                 # Google Colab notebook for running sweeps on cloud T4/A100 GPUs
└── README.md                          # Project overview and workflow instructions
```

---

## 🚀 Quickstart for Team Members

### 1. Environment Setup
```bash
git clone <repo-url>
cd CS769
pip install -r requirements.txt
```

### 2. Trained LoRA Adapters
All four specialized LoRA adapters are saved in `outputs/adapters/`:
- `outputs/adapters/e_math` (Math Reasoning)
- `outputs/adapters/e_code` (Python Code Generation)
- `outputs/adapters/e_instr` (Complex Instruction Following)
- `outputs/adapters/e_pref` (Helpful/Harmless Preference Anchor)

To merge any adapter with the base model into a standalone checkpoint:
```bash
python src/merge/bake_adapters.py --adapter_path outputs/adapters/e_math --output_path outputs/checkpoints/e_math
```

### 3. Merging Experts (Experiment 2 & Experiment 3)
To merge experts using Linear Arithmetic, SLERP, TIES, or DARE-TIES:
```python
from src.merge.merge_methods import merge_linear, merge_slerp, merge_ties, merge_dare_ties
```
Or run the full automated sweep:
```bash
python src/merge/run_e2_sweep.py
```

### 4. Running Benchmarks
- **Math (SVAMP / GSM8K)**: `python src/eval/eval_math.py`
- **Code (MBPP Pass@1)**: `python src/eval/eval_code.py`
- **Instruction Following (IFEval)**: `python src/eval/eval_ifeval.py`
- **Preference Retention (UltraFeedback)**: `python src/eval/eval_pref.py`
- **Parameter Conflict Diagnostics**: `python src/diagnostics/layer_conflicts.py`

---

## 📊 Summary of Key Results

- **Figure 2**: Capability Retention ($\text{CR}_{\text{math}}$) and Preference Retention Ratio ($\text{PRR}$) curves across $\alpha \in [0.0, 1.0]$.
- **Figure 3**: Raw SVAMP accuracy (%) and DPO margin against single-task baselines.
- **Diagnostics**:
  - $\|\Delta\theta_{\text{pref}}\|_2 = 1.2365$, $\|\Delta\theta_{\text{math}}\|_2 = 9.2217$ (Norm ratio: **$0.1341$**, math updates are $7.46\times$ larger).
  - Cosine similarity: **$-5.45 \times 10^{-7} \approx 0.0$** (Orthogonal updates).
  - Sign conflict rate: **$32.07\%$** over the joint top-20% magnitude entries.
