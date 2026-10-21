# Table E5: post-merge repair budget

Parent = E_pref (margin +0.2761 on 500 pairs). Seed 42. Reference model for repair DPO: pre_dpo.

| Model | Budget | Steps | Tokens | DPO train s | Pref margin (500) | Raw win % (500) | Margin (first 200) | PRR vs E_pref [95% CI] | Lost gain recovered | SVAMP | CR_math |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 | - | - | - | - | +0.0000 | 53.8 | +0.0000 | 0.0% [0.0, 0.0] | - | 53.67 | 0% |
| E_math | - | - | - | - | -0.0963 | 55.6 | -0.1436 | -34.9% [-76.4, 1.7] | - | 60.00 | 100% |
| E_pref (parent) | - | - | - | - | +0.2761 | 56.2 | +0.3021 | 100.0% [100.0, 100.0] | - | 53.67 | 0% |
| **case linear_0.1** (E2: PRR -26.4%, CR_math +162%) | | | | | | | | | | | |
| pre-repair linear_0.1 | 0 | - | - | - | -0.0737 | 55.6 | -0.1176 | -26.7% [-66.6, 7.4] | - | 60.33 | 105% |
| repair linear_0.1 | 128 | 16 | 102,735 | 139 | -0.0232 | 55.4 | -0.0473 | -8.4% [-48.2, 28.1] | 14% | 59.33 | 89% |
| repair linear_0.1 | 512 | 64 | 398,559 | 523 | +0.1111 | 55.2 | +0.1121 | 40.3% [1.6, 76.8] | 53% | 58.33 | 74% |
| repair linear_0.1 | 2000 | 250 | 1,586,561 | 2107 | +0.2476 | 55.8 | +0.2736 | 89.7% [49.9, 131.2] | 92% | 61.00 | 116% |
| **case linear_0.5** (E2: PRR +29.8%, CR_math +81%) | | | | | | | | | | | |
| pre-repair linear_0.5 | 0 | - | - | - | +0.0664 | 54.2 | +0.1013 | 24.0% [-1.0, 49.9] | - | 57.33 | 58% |
| repair linear_0.5 | 128 | 16 | 102,735 | 138 | +0.1145 | 54.8 | +0.1477 | 41.5% [16.0, 69.4] | 23% | 58.00 | 68% |
| repair linear_0.5 | 512 | 64 | 398,559 | 536 | +0.2511 | 55.8 | +0.2847 | 90.9% [60.7, 122.1] | 88% | 56.00 | 37% |
| repair linear_0.5 | 2000 | 250 | 1,586,561 | 2138 | +0.3406 | 56.4 | +0.3581 | 123.4% [91.2, 157.2] | 131% | 58.00 | 68% |
| control: DPO from B0 | 128 | 16 | 102,735 | 133 | +0.0664 | 55.0 | +0.0681 | 24.1% [5.1, 39.3] | - | 51.67 | -32% |
| control: DPO from B0 | 512 | 64 | 398,559 | 522 | +0.1311 | 55.0 | +0.1435 | 47.5% [34.8, 60.6] | - | 52.67 | -16% |

## Smallest repair budget reaching the target PRR

| Case | Target | By point estimate | By 95% CI lower bound | Log-linear interpolation |
|:---|:---|:---|:---|:---|
| linear_0.1 | 90% | not reached | not reached | not reached |
| linear_0.1 | 95% | not reached | not reached | not reached |
| linear_0.5 | 90% | 512 | 2000 | ~499 |
| linear_0.5 | 95% | 2000 | not reached | ~607 |

## Repaired merge vs DPO-from-B0 control at equal budget (paired bootstrap, 95% CI)

| Case | Budget | Δ pref margin (repair − control) | Δ SVAMP (pts) |
|:---|:---|:---|:---|
| linear_0.1 | 128 | -0.090 [-0.199, +0.017] | +7.7 [+1.7, +13.7] |
| linear_0.1 | 512 | -0.020 [-0.116, +0.073] | +5.7 [-0.3, +12.0] |
| linear_0.5 | 128 | +0.048 [-0.033, +0.132] | +6.3 [+0.7, +12.3] |
| linear_0.5 | 512 | +0.120 [+0.036, +0.203] | +3.3 [-2.0, +9.0] |
