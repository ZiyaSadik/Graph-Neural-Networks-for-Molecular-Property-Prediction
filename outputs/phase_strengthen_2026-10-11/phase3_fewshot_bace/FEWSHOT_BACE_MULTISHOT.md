# Phase 3 — Rigorous Few-Shot BACE Evaluation (Protocol B family)

**Created (UTC):** 2026-10-11T09:01:05.902079+00:00

## Protocol boundaries

- Support from **train only**; queries on **validation** for multi-K/multi-seed stats.
- Locked **test** used only to reproduce the original K=5 seed=42 report.
- Distances / (d0−d1) are **not** calibrated probabilities.
- **Not interchangeable** with Protocol A (episodic ProtoNet on the test pool).
- No-GraphCL checkpoint was **not available** on disk for Protocol B comparison; episodic Protocol A ablation remains the recorded GraphCL vs No-GraphCL comparison.

## Canonical reproduction (K=5, seed=42)

- Checkpoint SHA match: **True**
- Overall consistent with saved report: **True**

### validation
- support indices match: True
- metrics match: True

| metric | original | reproduced | |Δ| |
|---|---:|---:|---:|
| accuracy | 0.4503311258278146 | 0.4503311258278146 | 0.0 |
| balanced_accuracy | 0.6408424908424908 | 0.6408424908424908 | 0.0 |
| sensitivity_class_1 | 0.3769230769230769 | 0.3769230769230769 | 0.0 |
| specificity_class_0 | 0.9047619047619048 | 0.9047619047619048 | 0.0 |
| auroc | 0.6384615384615384 | 0.6384615384615384 | 0.0 |

### test
- support indices match: True
- metrics match: True

| metric | original | reproduced | |Δ| |
|---|---:|---:|---:|
| accuracy | 0.6447368421052632 | 0.6447368421052632 | 0.0 |
| balanced_accuracy | 0.6597113545470353 | 0.6597113545470353 | 0.0 |
| sensitivity_class_1 | 0.43209876543209874 | 0.43209876543209874 | 0.0 |
| specificity_class_0 | 0.8873239436619719 | 0.8873239436619719 | 0.0 |
| auroc | 0.7278734133194227 | 0.7278734133194227 | 0.0 |

## Multi-shot validation summary (predeclared seeds)

- K shots: [1, 3, 5, 10]
- Seeds: [42, 7, 13, 21, 99]
- Query split: `valid` (n=151)
- Episodes planned: 20

For each class c in {0,1}, prototype_c = mean(GIN graph_embedding of support examples with original label c). Query prediction = argmin_c ||emb(q) - prototype_c||_2. Score for AUROC = d0 - d1 (higher => more class-1-like). Scores are not calibrated probabilities.

| K | n_eps | bal_acc mean±std | sens mean±std | spec mean±std | AUROC mean±std | AUROC undef |
|---:|---:|---|---|---|---|---:|
| 1 | 5 | 0.5450±0.0496 | 0.3662±0.2481 | 0.7238±0.2031 | 0.5766±0.0640 | 0 |
| 3 | 5 | 0.5340±0.0601 | 0.4108±0.2104 | 0.6571±0.1594 | 0.5437±0.0853 | 0 |
| 5 | 5 | 0.5290±0.0950 | 0.3246±0.1290 | 0.7333±0.2065 | 0.5783±0.0765 | 0 |
| 10 | 5 | 0.5839±0.0178 | 0.4631±0.0840 | 0.7048±0.1032 | 0.6080±0.0310 | 0 |

## Figures

- `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/plot_valid_balanced_accuracy.png`
- `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/plot_valid_auroc.png`
- `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/plot_valid_sensitivity_class_1.png`
- `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/plot_valid_specificity_class_0.png`

## Limitations

- Validation is class-imbalanced (more actives); specificity estimates have high variance at small n_neg.
- Multi-K results are **not** tuned on the locked test set.
- GraphCL vs No-GraphCL under this Protocol B family was not re-run (no No-GraphCL weights).
