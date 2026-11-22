# Does Alignment Survive the Merge?
## A Controlled Study of Preference Retention and Low-Cost Post-Merge Repair in Small Language Models

*Detailed project plan for a paper submission on consortium model merging with DPO/GRPO*

### One-Sentence Paper Idea
Train several compatible small-model experts from the same base, preference-tune them before and/or after weight merging, and measure a preference-retention curve that reveals when alignment survives, when it is erased by parameter interference, and how few preference examples are needed to repair it.

> **Recommended scope:** DPO as the primary clean experiment; GRPO as a secondary replication using a compact reward/verifier.

---

## 1. Executive Summary and Novelty Position

* **Core research question:** If members of a consortium independently fine-tune compatible copies of the same small language model, does preference alignment learned by DPO or GRPO survive when their weights are merged, or is preference tuning most effective after the merge?

The simple "DPO before versus after merging" comparison is no longer novel by itself. Aakanksha et al. (2024) already tested continual DPO after merging in a multilingual safety/general-purpose setup and reported stronger safety gains for merge-then-DPO than DPO-then-merge. Newer work also explicitly targets alignment-preserving or preference-guided merging. The defensible gap is therefore a controlled characterization of preference retention: its dose-response behavior, dependence on merge method and number of experts, predictors of failure, and minimum repair budget.

### Recommended Paper Claim
Weight merging should be treated as an alignment intervention, not a neutral deployment operation. Preference behavior is retained non-uniformly across merge ratios and methods; simple parameter-conflict statistics can predict retention loss; and a very small post-merge preference "repair" set can recover most lost alignment without retraining each expert.

### Novel Contributions to Target
* Define and benchmark **Preference Retention Ratio (PRR)**: how much of a preference-tuned parent model's gain remains after merging, normalized against the shared base.
* Produce retention curves over merge coefficient, number of merged experts, merge algorithm, preference-set size, and expert heterogeneity instead of reporting only one merge point.
* Compare three realistic consortium workflows: `align-each-then-merge`, `merge-then-align`, and `align-one-anchor-plus-domain-experts`.
* Measure whether lightweight diagnostics such as delta cosine similarity, sign conflict, and alignment-delta norm predict preference loss before expensive generation evaluation.
* Quantify the smallest post-merge DPO/GRPO repair budget (for example 125, 500, 2,000 examples/prompts) needed to recover 90–95% of the pre-merge preference gain.
* Release a compact, reproducible benchmark recipe using sub-2B models so other labs can reproduce the result without frontier-scale compute.

---

## 2. Candidate Paper Titles

| Use | Title |
| :--- | :--- |
| **Primary** | Does Alignment Survive the Merge? Preference Retention and Low-Cost Repair in Small Language Models |
| **More technical** | Preference Retention Under Weight Merging: When DPO/GRPO Alignment Survives and When It Needs Repair |
| **Consortium framing** | Merge First or Align First? Preference Retention in Consortium Language-Model Merging |
| **Punchier** | Aligned Apart, Misaligned Together? Measuring Preference Drift After Model Merging |

---

## 3. Research Questions and Falsifiable Hypotheses

| ID | Question | Testable Hypothesis |
| :--- | :--- | :--- |
| **RQ1** | Does preference tuning survive merging? | **H1:** PRR declines as the aligned model receives less effective merge weight and as parameter conflict increases. It will not be binary; there will be a retention curve. |
| **RQ2** | Does workflow order matter? | **H2:** $\text{Merge} \rightarrow \text{DPO}$ will yield the strongest final preference score at equal preference-data budget, while $\text{DPO} \rightarrow \text{merge}$ will retain more non-preference expert capability in some settings. |
| **RQ3** | Do better merge algorithms preserve alignment? | **H3:** TIES/DARE-TIES will preserve preference gains better than naive linear averaging when aligned and task deltas contain sign conflict. |
| **RQ4** | Does alignment method matter? | **H4:** DPO and GRPO will show different retention patterns because they induce different parameter updates; the difference should be visible in delta geometry and repair sample efficiency. |
| **RQ5** | Can we predict failure cheaply? | **H5:** Sign-conflict rate and cosine similarity between the preference delta and other expert deltas will correlate with PRR and can flag risky merges without running a full judge evaluation. |
| **RQ6** | How much repair is enough? | **H6:** A small post-merge repair set will recover most preference loss, with diminishing returns beyond a few thousand examples for a $0.5\text{B}\text{–}1.5\text{B}$ model. |

---

## 4. Experimental Design

### 4.1 Model Families
* **Primary recommendation:** Use one $0.5\text{B}\text{–}0.6\text{B}$ instruct model for the full factorial study, then replicate the central finding on one $1.5\text{B}\text{–}1.7\text{B}$ instruct model. Good candidates include `Qwen2.5-0.5B/1.5B-Instruct` or `SmolLM2-360M/1.7B-Instruct`. All experts in a merge must start from the exact same base checkpoint and tokenizer.
* **Primary scale:** $0.5\text{B}$-ish model, three random seeds for the core conditions.
* **Replication scale:** $1.5\text{B}$-ish model, one or two seeds for the most important conditions.
* **Training:** LoRA/QLoRA for cost control, then bake adapters into full weights before applying weight-space merging.
* Keep chat template, sequence length, optimizer family, quantization policy, and decoding settings fixed across conditions.

### 4.2 Consortium Experts to Train
Create three domain experts plus one preference expert from the same base. The exact datasets may be swapped for license/compute reasons, but keep the domains behaviorally distinct.

| Expert | Training | Suggested Data | Purpose |
| :--- | :--- | :--- | :--- |
| $E_{\text{math}}$ | SFT | MetaMath/GSM8K-style subset | Reasoning specialist |
| $E_{\text{code}}$ | SFT | CodeAlpaca/MBPP-style subset | Code specialist |
| $E_{\text{instr}}$ | SFT | Small instruction-following subset | General instruction specialist |
| $E_{\text{pref}}$ | DPO | 2k preference pairs from UltraFeedback or HH-RLHF | Preference/alignment specialist |

> **Important control:** Also construct "aligned domain experts" by applying the same DPO preference set separately to $E_{\text{math}}$, $E_{\text{code}}$, and $E_{\text{instr}}$. This allows a genuine consortium test: several independently aligned experts are merged together, rather than one aligned checkpoint being diluted by unaligned experts.

### 4.3 Preference Data Regimes
* **Tiny:** $500$ chosen/rejected pairs. Designed to test the "small preference set" premise.
* **Small:** $2,000$ pairs. Recommended main setting.
* **Medium:** $8,000$ pairs. Used only for a scaling ablation.
* Keep $500\text{–}1,000$ held-out preference pairs completely out of training for pairwise log-probability evaluation.
* For **GRPO**, use the same underlying preference domain. Convert it to prompt-only training and score sampled responses with a fixed compact reward model trained on the training split or a deterministic verifier where possible. Do not let the evaluation judge be the GRPO reward source.

### 4.4 The Essential Workflow Comparison

| ID | Pipeline | Interpretation | Must-have? |
| :--- | :--- | :--- | :--- |
| **B0** | Base | Untuned reference | Yes |
| **B1** | Domain SFT experts individually | Upper bounds for specialist capabilities | Yes |
| **B2** | DPO-only preference expert | Preference upper bound before merge | Yes |
| **W1** | $\text{SFT experts} \rightarrow \text{merge}$ | Unaligned merged baseline | Yes |
| **W2** | $\text{Each expert DPO} \rightarrow \text{merge}$ | Align first, then merge | Yes |
| **W3** | $\text{SFT experts} \rightarrow \text{merge} \rightarrow \text{DPO}$ | Merge first, then align | Yes |
| **W4** | $\text{Preference expert} + \text{domain experts merge}$ | Does a single alignment anchor survive dilution? | Yes |
| **W5** | $\text{SFT experts} \rightarrow \text{merge} \rightarrow \text{GRPO}$ | GRPO post-merge replication | Secondary |
| **W6** | $\text{Each expert GRPO} \rightarrow \text{merge}$ | GRPO pre-merge replication | Secondary |

### 4.5 Merge Methods and Coefficient Sweeps
* **Linear interpolation / weighted averaging:** The interpretable baseline. For two-model merges sweep preference-expert weight $\alpha \in \{0.0, 0.1, \dots, 1.0\}$.
* **Task arithmetic:** Merge expert deltas relative to the common base, with a small scaling sweep.
* **TIES-Merging:** Tests whether explicit resolution of sign interference helps retain alignment.
* **DARE-TIES:** Tests whether sparsifying redundant deltas before conflict resolution improves retention.
* **Optional SLERP:** Include if implementation is stable, mainly to connect with prior multilingual merging results.
* For $N$-way consortium merges, use $N \in \{2, 3, 4\}$. Keep total merge mass normalized and report both equal-weight and alignment-anchor-weighted variants.

---

## 5. Metrics: What Makes the Answer Clear

### 5.1 Primary Metric: Preference Retention Ratio (PRR)
Define:
$$\text{PRR} = \frac{S_{\text{merge}} - S_{\text{base}}}{S_{\text{pref}} - S_{\text{base}}}$$
where $S$ is a held-out preference score, $S_{\text{pref}}$ is the aligned parent before merging, and $S_{\text{merge}}$ is the merged checkpoint.
* $\text{PRR} \approx 1$ means the learned preference gain survives.
* $\text{PRR} \approx 0$ means it is erased.
* $\text{PRR} < 0$ means merging reverses the gain.

Report unclipped PRR and bootstrap confidence intervals. Use at least two $S$ definitions:
1. Pairwise preference accuracy or mean DPO margin on held-out chosen/rejected pairs (cheap, deterministic, and directly tied to the preference dataset).
2. Generation-level win rate/reward on unseen prompts using an evaluator that was not used for training. For human-facing claims, manually inspect a small stratified sample.

### 5.2 Capability-Retention Metrics
* **Math:** GSM8K exact match or another small reasoning benchmark.
* **Code:** MBPP pass@1 on a manageable subset.
* **Instruction following:** IFEval-style constraint success or an equivalent deterministic subset.
* **General language ability:** One compact multiple-choice benchmark such as ARC-Challenge or HellaSwag.

Report a normalized capability-retention score so the paper exposes the preference/capability Pareto trade-off instead of optimizing alignment in isolation.

### 5.3 Merge-Conflict Diagnostics
* Cosine similarity between flattened task deltas $\Delta_{\text{pref}}$ and $\Delta_{\text{domain}}$, globally and layer-wise.
* Fraction of overlapping parameters with opposite signs among top-$k$ magnitude delta entries.
* Delta norm ratio $\|\Delta_{\text{pref}}\| / \|\Delta_{\text{domain}}\|$ and layer-wise concentration of preference updates.
* **Optional:** Fisher-weighted or gradient-based sensitivity on the held-out preference set for a mechanistic add-on.

---

## 6. Required Experiments

| Experiment | Question | Protocol | Paper Artifact |
| :--- | :--- | :--- | :--- |
| **E1** | Sanity check | Verify each specialist improves its intended task and the preference expert improves held-out preference metrics versus base. | Table 1 |
| **E2** | Two-model retention curve | Merge $E_{\text{pref}}$ with one domain expert across $\alpha = 0 \dots 1$ for linear, TIES, DARE-TIES. | Figure 2: PRR vs $\alpha$ |
| **E3** | Consortium size | Merge 2, 3, and 4 experts; compare one alignment anchor versus all-experts-aligned. | Figure 3 |
| **E4** | Order-of-operations | Equal-budget comparison of $\text{DPO} \rightarrow \text{merge}$ vs $\text{merge} \rightarrow \text{DPO}$ using the same preference examples and training tokens. | Table 2 |
| **E5** | Repair budget | After an alignment-degrading merge, run DPO with $125/500/2\text{k}$ examples; measure recovery to parent PRR. | Figure 4: recovery curve |
| **E6** | Preference data size | Repeat core workflow at $500/2\text{k}/8\text{k}$ pairs. | Table/Figure 5 |
| **E7** | Merge algorithm ablation | Linear vs task arithmetic vs TIES vs DARE-TIES; matched coefficients. | Table 3 |
| **E8** | Conflict predictor | Correlate delta conflict metrics with PRR across all merges; simple regression or rank correlation. | Figure 6 |
| **E9** | GRPO replication | Repeat the central pre/post-merge result and one repair-budget experiment using GRPO. | Table 4 |
| **E10** | Scale replication | Repeat E2, E4, and E5 on the $1.5\text{B}\text{–}1.7\text{B}$ model. | Table 5 / Appendix |
| **E11** | Seed robustness | Three seeds for primary $0.5\text{B}$ conditions; confidence intervals and paired tests. | Error bars |
| **E12** | Evaluator robustness | Compare held-out preference margin, reward model, and independent judge on a subset. | Appendix |

> **Minimum viable paper:** If compute is tight, $\text{E1–E8} + \text{E11}$ on one $0.5\text{B}$ model can already tell a coherent story. Add E9 (GRPO) and E10 (larger model) as strengthening experiments rather than letting them block the core result.

---

## 7. Ablations Reviewers Are Likely to Ask For

* **Same data, same compute:** Control total DPO examples/tokens between pre-merge and post-merge workflows.
* **Same starting checkpoint:** No cross-family merges in the main experiment.
* **LoRA rank sensitivity:** At least $r=16$ versus $r=64$ for a small subset, or justify one rank with update-norm matching.
* **Merge-weight sweep** rather than one handpicked coefficient.
* **One preference dataset shift:** Train on one source and evaluate on another small preference set to test whether retention is merely memorization.
* **Number-of-experts ablation:** Two versus three versus four experts.
* **Alignment-anchor ablation:** Only one aligned expert versus all aligned experts before merge.
* **Reference-model choice for DPO:** Fixed original instruct base versus immediate pre-DPO checkpoint, documented consistently.
* **GRPO reward-source ablation** (if GRPO is a main claim): Compact reward model versus deterministic/verifiable reward where feasible.

---

## 8. Statistical Analysis

* Report $\text{mean} \pm 95\%$ bootstrap CI for win rates and PRR. Use prompt-level paired bootstrap whenever the same prompts are scored across models.
* For benchmark accuracy, report exact or bootstrap confidence intervals and paired tests where outputs share items.
* For conflict-to-retention analysis, report Spearman $\rho$ with confidence interval; optionally fit a simple linear/ridge model with merge method and $\alpha$ as covariates.
* Predefine the main comparison: **W2** ($\text{align} \rightarrow \text{merge}$) versus **W3** ($\text{merge} \rightarrow \text{align}$) at $2\text{k}$ preference examples and the same domain experts.
* Do not claim universal superiority from one dataset. Phrase the result as a controlled empirical characterization over the tested model families, tasks, and merge methods.

---

## 9. What Outcomes Would Make a Strong Paper?

The paper remains interesting under several possible outcomes:
* **If alignment mostly survives:** Show the boundary conditions, identify safe merge regimes, and demonstrate that re-DPO is often unnecessary.
* **If alignment often collapses:** Quantify the collapse curve and show that parameter conflict predicts it.
* **If $\text{merge} \rightarrow \text{DPO}$ consistently wins:** The contribution becomes the controlled sample-efficiency result and the minimal post-merge repair recipe.
* **If DPO and GRPO differ sharply:** The mechanism-specific retention result becomes a central contribution.
* **If TIES/DARE-TIES preserve alignment substantially better:** Frame the result around alignment-aware merge selection and diagnostics.

---

## 10. Figures and Tables to Plan Before Running Experiments

| Artifact | Name | What It Proves |
| :--- | :--- | :--- |
| **Fig. 1** | Pipeline schematic | Base $\rightarrow$ independent experts $\rightarrow$ preference tuning before/after merge $\rightarrow$ evaluation. Show the three consortium workflows. |
| **Fig. 2** | Preference retention curves | PRR versus preference-expert merge weight $\alpha$, one curve per merge method. |
| **Fig. 3** | Consortium scaling | PRR and capability retention versus number of experts $N$. |
| **Fig. 4** | Repair efficiency | PRR recovered versus post-merge preference examples ($0, 125, 500, 2\text{k}, 8\text{k}$). |
| **Fig. 5** | Preference/capability Pareto | Scatterplot: preference score vs normalized domain capability for each workflow. |
| **Fig. 6** | Conflict predicts forgetting | Sign-conflict/cosine-similarity diagnostic versus PRR, with regression trend. |
| **Table 1** | Expert sanity checks | Base and each specialist on preference + task metrics. |
| **Table 2** | Main workflow comparison | $\text{Align} \rightarrow \text{merge}$ vs $\text{merge} \rightarrow \text{align}$ vs alignment-anchor merge, matched budget. |
| **Table 3** | Merge algorithm ablation | Linear, task arithmetic, TIES, DARE-TIES. |
| **Table 4** | DPO vs GRPO | Central retention and repair numbers. |
| **Table 5** | Scale replication | Primary results on the larger small model. |

---

## 11. Recommended Paper Headings

| Section | Content |
| :--- | :--- |
| **Abstract** | One problem sentence, one gap sentence, method/benchmark sentence, 2–3 numerical findings, implication. |
| **1. Introduction** | Motivate consortium/federated model development; explain why merging may perturb alignment; state the exact research questions and contributions. |
| **2. Related Work** | 2.1 Model merging; 2.2 Preference optimization (DPO/GRPO); 2.3 Alignment and safety under model composition; 2.4 Post-merge adaptation. |
| **3. Problem Setup** | Common base, expert deltas, merge operator, preference objective, definitions of PRR and capability retention. |
| **4. Experimental Protocol** | Models, datasets, expert training, merge methods, DPO/GRPO setup, evaluation, statistics, compute. |
| **5. Does Preference Tuning Survive Merging?** | Core retention curves and consortium-size results. |
| **6. Align Before or After Merging?** | Matched-budget workflow comparison. |
| **7. How Much Post-Merge Repair Is Needed?** | Sample-efficiency / repair-budget experiments. |
| **8. What Predicts Preference Loss?** | Parameter-delta conflict and layer-wise analysis. |
| **9. DPO vs GRPO and Scale Replication** | Secondary algorithm comparison and larger-model confirmation. |
| **10. Discussion and Limitations** | Scope, judge limitations, model-size limits, preference-data representativeness, implications for real consortium deployments. |
| **11. Conclusion** | One concise answer to the title question plus practical recommendation conditioned on measured diagnostics. |
| **Appendix** | Hyperparameters, data filtering, prompts, extra coefficients, all seeds, evaluator agreement, compute, licenses, examples. |

---

## 12. Execution Plan

| Milestone | Deliverable |
| :--- | :--- |
| **Lock scope and infrastructure** | Freeze base models, datasets, evaluation scripts, merge library, seeds, metric definitions. Run 50-example smoke tests. |
| **Train experts** | Train three domain SFT experts and DPO preference expert at primary model scale. Validate gains. |
| **Core two-model merges** | Run $\alpha$ sweeps for linear + TIES + DARE-TIES. Generate first PRR curves. |
| **Consortium and workflow order** | Run $N$-way merges and W2/W3/W4 matched-budget comparison. |
| **Repair-budget study** | Run post-merge DPO at $125/500/2\text{k}/8\text{k}$ examples. Lock main paper story. |
| **Diagnostics and ablations** | Compute delta conflict metrics, data-size ablation, key hyperparameter checks. |
| **GRPO + larger-model replication** | Run only the central conditions, not the full factorial grid. |
| **Statistics and figure freeze** | Bootstrap CIs, significance checks, evaluator agreement, final plots/tables. |
| **Write full draft** | Methods first, then results, introduction, related work, limitations. Prepare appendix and reproducibility checklist. |
| **Internal review and submission** | Run leakage/license checks, rerun any fragile result, polish abstract/title, format for target venue, archive code/configs. |

---

## 13. Compute-Conscious Implementation Plan

* Use PEFT/LoRA for expert training and DPO; save adapters and bake them only when producing merge checkpoints.
* Cache evaluation prompts, reference log-probabilities, and generated candidates when valid so coefficient sweeps do not repeat unnecessary work.
* Run deterministic pairwise log-prob evaluation on every checkpoint; reserve generation-based judging for the reduced set of representative checkpoints.
* Start with $0.5\text{B}$ factorial experiments. Do not start the larger-model replication until the main effect and plots are stable.
* Track every checkpoint with a manifest: base hash, dataset hash, seed, LoRA config, merge method/coefficients, preference method, and evaluation commit.

---

## 14. Main Risks and Mitigation

| Risk | Why It Matters | Mitigation |
| :--- | :--- | :--- |
| **Novelty collision** | Recent model-merging/alignment papers are appearing quickly. | Keep the paper centered on PRR, controlled sweeps, repair sample efficiency, and predictors. Re-run literature search before submission. |
| **Weak DPO signal at $0.5\text{B}$** | Small model may not learn preferences reliably. | Use $1.5\text{B}$ model earlier for the DPO sanity check, or choose a cleaner preference dataset; verify pairwise margin before merging. |
| **GRPO instability** | Reward variance may obscure the merge effect. | Treat GRPO as replication, use a fixed compact reward/verifier, conservative KL, and fewer central conditions. |
| **Judge contamination** | Training reward and evaluator overlap can inflate results. | Separate training reward from final judge; include deterministic held-out preference margin and evaluator agreement. |
| **Too many runs** | Full grid can explode combinatorially. | Use staged gating: linear merge first, then only add TIES/DARE where the effect is clear; run larger scale only on central conditions. |
| **Capability trade-off** | Post-merge DPO may recover alignment but damage expert skills. | Always report capability retention and Pareto plots, not preference score alone. |

---

## 15. Reproducible Run Checklist

1. Freeze the base checkpoint and tokenizer hash.
2. Create train/validation/test splits for each domain and the preference data before training any model.
3. Train each domain expert independently from the base using identical PEFT structure where possible.
4. Train DPO preference expert from the same base; verify held-out preference improvement.
5. Bake adapters and compute expert deltas relative to the base.
6. Generate merge checkpoints for planned coefficients and algorithms.
7. Evaluate deterministic held-out preference margin and domain benchmarks on every checkpoint.
8. Select representative checkpoints for generation-level evaluation.
9. Run matched-budget pre-merge/post-merge alignment conditions.
10. Run repair-budget sweep on the worst and median merge cases.
11. Compute parameter-conflict diagnostics and correlate them with PRR.
12. Repeat central conditions with GRPO and then with the larger model.
13. Freeze figures/tables only after seed and evaluator robustness checks.

---

## 16. Literature Positioning Checked for This Plan

The following papers are particularly relevant. They should be cited in the eventual manuscript, and their exact relationship to the proposed contribution should be described explicitly.

| Reference | Paper | Relevance | Link |
| :--- | :--- | :--- | :--- |
| **Rafailov et al., 2023** | *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* | Foundation for DPO. | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| **Yadav et al., 2023** | *TIES-Merging: Resolving Interference When Merging Models.* | Merge algorithm designed to resolve parameter interference. | [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) |
| **Yu et al., 2024** | *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch.* | DARE sparsification and merging of homologous fine-tuned models. | [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) |
| **Aakanksha et al., 2024** | *Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning.* | Directly relevant prior evidence: compares DPO before vs after merge in multilingual safety/general-purpose training. | [arXiv:2410.10801](https://arxiv.org/abs/2410.10801) |
| **Hammoud et al., 2024** | *Model Merging and Safety Alignment: One Bad Model Spoils the Bunch.* | Shows merging can propagate misalignment and studies alignment-aware merging. | [arXiv:2406.14563](https://arxiv.org/abs/2406.14563) |
| **Lu et al., 2024** | *Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignment.* | Connects merging operations with RLHF/DPO optimization dynamics. | [arXiv:2405.17931](https://arxiv.org/abs/2405.17931) |
| **Corrado et al., 2025** | *AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs.* | Preference optimization across multiple tasks; compares against model-merging baselines. | [ACL 2025](https://aclanthology.org/2025.acl-long.990/) |
| **Roy et al., 2025** | *AlignMerge: Alignment-Preserving Large Language Model Merging via Fisher-Guided Geometric Constraints.* | Explicitly frames alignment preservation as a merging objective. | [arXiv:2512.16245](https://arxiv.org/abs/2512.16245) |
| **Fu & Xu, 2026** | *TPMM-DPO: Trajectory-aware Preference-guided Model Merging for Iterative Direct Preference Optimization.* | Uses model merging inside iterative DPO to stabilize reference-model trajectories. | [arXiv:2605.23398](https://arxiv.org/abs/2605.23398) |
| **Zheng et al., 2026** | *CoMerge: Conflict-Driven Preference Optimization for Multi-Task Model Merging.* | Uses preference optimization to learn/refine merge coefficients from conflict-induced preference pairs. | [arXiv:2609.02273](https://arxiv.org/abs/2609.02273) |