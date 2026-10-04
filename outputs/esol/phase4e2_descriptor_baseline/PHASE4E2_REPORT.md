# Phase 4E.2 — ESOL Descriptor Regression Baseline

## Summary

| Item | Value |
|---|---|
| Model | StandardScaler + Ridge(α=1.0) |
| Descriptors | 24 RDKit 2D features |
| Train / valid n | 903 / 113 |
| Validation MAE | 0.965157 |
| Validation RMSE | 1.286634 |
| Validation R² (supplementary) | 0.692104 |
| Phase 4E.1 mean MAE / RMSE | 1.886369 / 2.320262 |
| MAE improvement vs mean | 0.921211 |
| RMSE improvement vs mean | 1.033627 |
| Test metrics | Not computed |

## Justification

Ridge on standardized RDKit descriptors is a simple, mostly deterministic conventional baseline for ~900 training molecules. No hyperparameter search was performed (`alpha=1.0`, `random_state=42`). Random forests were not used to avoid unnecessary variance and tuning on this small scaffold-split set.

## Descriptors

MolWt, ExactMolWt, HeavyAtomCount, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, NumAliphaticRings, NumSaturatedRings, NumHeteroatoms, MolLogP, TPSA, LabuteASA, FractionCSP3, BertzCT, Chi0v, Chi1v, Chi0n, Chi1n, Kappa1, Kappa2, HallKierAlpha, NumValenceElectrons

Non-finite descriptor values: fail loudly (none observed). No rows dropped.

## Comparison to Phase 4E.1

Same frozen validation rows and measured logS target. Descriptor model MAE/RMSE are lower than the train-mean constant baseline (see `results.json`).

## Reproducibility

```text
python outputs/esol/phase4e2_descriptor_baseline/recompute_descriptor_baseline.py
```

Internal refit check: matches=True

## Provenance

- Source SHA-256: `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
- Mapping fingerprint: `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac`
- RDKit 2025.09.3, scikit-learn 1.8.0, NumPy 2.3.5

## Scope

No GIN regression head, toxicity, CLI changes, dependency installs, or split modifications. Phase 4E.1 artifacts untouched.
