# Phase 4E.1 — ESOL Train-Mean Baseline (Validation Only)

## Summary

Constant baseline: predict the **training-set mean** measured log-solubility for every **validation** molecule.

| Item | Value |
|---|---|
| Train mean (measured logS) | -3.1296666667 |
| Validation n | 113 |
| Validation MAE | 1.8863687316 |
| Validation RMSE | 2.3202618188 |
| Validation R² (supplementary) | -0.0013085822 |
| Test metrics | **Not computed** |

## Procedure

1. Ran `python data/moleculenet/esol/verify_esol_split_frozen_v1.py` (PASS).
2. Loaded frozen assignments from `esol_split_assignments_v1.csv` (not regenerated).
3. Confirmed source SHA-256 `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` and mapping fingerprint `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac`.
4. Target column: `measured log solubility in mols per litre` (never `ESOL predicted log solubility in mols per litre`).
5. `train_mean = mean(y_train)` over 903 train rows.
6. For each of 113 validation rows: `y_pred = train_mean`.
7. MAE / RMSE on original measured scale; R² vs validation mean (supplementary).

## Reproducibility

Internal recompute matched. External check:

```text
python outputs/esol/phase4e1_mean_baseline/recompute_mean_baseline.py
```

## Scope

- No descriptor model, GIN head, toxicity, CLI changes, or dependency installs.
- No test-label use for fitting or metrics.
- Split artifacts and research/BACE code untouched.
