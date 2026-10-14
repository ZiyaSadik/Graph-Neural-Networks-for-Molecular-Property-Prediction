# Phase 5 — BACE Calibration Assessment

**Claim calibrated probabilities:** **False**

Prototype distances are not generative logits. Prior gap analysis showed only modest discrimination (AUROC≈0.589 on episodic gaps). Validation is severely imbalanced (≈21 inactive / 130 active), so ECE/reliability bins are underpowered. Platt on (d0−d1) is reported as a diagnostic comparison only; outputs are NOT claimed as calibrated probabilities.

- Fit: train∖support (n=1200)
- Eval: validation (n=151; counts {'0': 21, '1': 130})
- Locked test used: False

| method | Brier | ECE | AUROC | log_loss |
|---|---:|---:|---:|---:|
| sigmoid(score) diagnostic | 0.5495 | 0.5644 | 0.6384615384615385 | 5.1050 |
| Platt(score) | 0.3412 | 0.4737 | 0.6384615384615385 | 0.8854 |

Figures: `reliability_uncalibrated_sigmoid.png`, `reliability_platt.png`.
