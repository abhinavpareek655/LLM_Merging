# Table 1: Specialist Sanity Checks Before Model Merging

| Model   |   GSM8K (EM %) |   MBPP (Pass@1 %) |   IFEval (Strict %) |   Raw Pref Win % |   DPO Margin (vs B0) |
|:--------|---------------:|------------------:|--------------------:|-----------------:|---------------------:|
| B0      |             30 |                45 |                  30 |               65 |               0      |
| E_math  |             25 |                45 |                  35 |               65 |              -0.0557 |
| E_code  |             30 |                35 |                  20 |               70 |              -0.0037 |
| E_instr |             15 |                 0 |                  25 |               60 |              -1.1456 |
| E_pref  |             40 |                45 |                  30 |               70 |               0.4274 |
