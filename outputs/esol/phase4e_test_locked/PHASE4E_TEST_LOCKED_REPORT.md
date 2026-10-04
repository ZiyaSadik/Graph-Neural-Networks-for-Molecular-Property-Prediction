# ESOL locked TEST evaluation

**Status:** One-shot test scoring complete. No retraining. No test-based selection.
**Created UTC:** 2026-10-04T09:39:28+00:00

## Pre-flight

- `verify_esol_split_frozen_v1.py`: PASS
- Mapping fingerprint: `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac`
- Source SHA-256: `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
- Encoder SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Regression head SHA-256: `368b49751ca56e14b87778af9203eaf4f487c2ea0f503bd994757ef3a7a2cdcb` (best validation epoch 103)
- Validation repro before test: 4E.1 / 4E.2 / 4E.3 MAE matched

## Test metrics (n=112; exclusions=0)

| Model | MAE | RMSE | R² (suppl.) | Valid MAE (ref) |
|---|---:|---:|---:|---:|
| Train-mean | 1.838101 | 2.229061 | -0.115299 | 1.8864 |
| RDKit descriptors + Ridge(α=1) | 0.786672 | 1.023883 | 0.764686 | 0.9652 |
| Frozen GIN + head | 0.724545 | 1.063461 | 0.746142 | 0.8777 |

Target: `measured log solubility in mols per litre` (forbidden predicted column unused).

## Governance

- Test data not used for model selection, tuning, or early stopping
- Validation artifacts and frozen split files not modified
- Head/checkpoint bytes not modified
