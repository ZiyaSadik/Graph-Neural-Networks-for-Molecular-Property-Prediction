# Phase 4 — BACE Explainability Evaluation

**Created (UTC):** 2026-10-11T09:02:54.912613+00:00

## Caveats

- High atom/bond importance is not chemical causality.
- Prototype distance gaps are not calibrated probabilities.
- GNNExplainer masks on PrototypeDistanceLogits have historically shown poor soft fidelity in this project; treat new fidelity numbers cautiously.
- Feature-zeroing perturbations are not chemically valid ablations.
- Tox21 explainability remains limited to the existing 2-molecule smoke test under outputs/tox21/phase4f6_explainability/; not extended here.

## Setup

- Checkpoint SHA: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Support manifest: `apps/mol_cli/support_manifests/bace_train_k5_v1.json`
- Predeclared valid indices: `[241, 244, 254, 255, 797, 801, 802, 806]`
- Predictions: active=1, inactive=7
- Index mapping OK: **True**

## Prototype-distance explanations

| idx | mol_id | true | pred | d0 | d1 | gap | correct |
|---:|---|---:|---:|---:|---:|---:|---|
| 241 | BACE_268 | 1 | 0 | 36.338 | 47.142 | 10.803 | False |
| 244 | BACE_271 | 1 | 0 | 28.224 | 43.669 | 15.445 | False |
| 254 | BACE_282 | 1 | 0 | 70.244 | 78.597 | 8.354 | False |
| 255 | BACE_283 | 1 | 1 | 34.828 | 31.492 | 3.336 | True |
| 797 | BACE_829 | 0 | 0 | 45.380 | 56.750 | 11.370 | True |
| 801 | BACE_833 | 0 | 0 | 27.543 | 32.212 | 4.669 | True |
| 802 | BACE_834 | 0 | 0 | 32.347 | 34.772 | 2.425 | True |
| 806 | BACE_839 | 0 | 0 | 34.999 | 42.155 | 7.156 | True |

## GNNExplainer

- Attempted: True
- Succeeded / failed: 8 / 0

- `BACE_268` target=0 pearson=0.5779241423859393 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_241_atom_importance.png
- `BACE_271` target=0 pearson=0.5492700291810652 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_244_atom_importance.png
- `BACE_282` target=0 pearson=0.43412803720594534 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_254_atom_importance.png
- `BACE_283` target=1 pearson=0.5460143929968868 mask_topk_flip=True random_flip=True fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_255_atom_importance.png
- `BACE_829` target=0 pearson=-0.008891352571223083 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_797_atom_importance.png
- `BACE_833` target=0 pearson=0.4514330502081462 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_801_atom_importance.png
- `BACE_834` target=0 pearson=0.5026985447158957 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_802_atom_importance.png
- `BACE_839` target=0 pearson=0.3731587597215656 mask_topk_flip=False random_flip=False fig=outputs/phase_strengthen_2026-10-11/phase4_explainability/figures/bace_806_atom_importance.png

## Tox21

Not extended beyond the existing two-molecule smoke under `outputs/tox21/phase4f6_explainability/`. A broader reproducible Tox21 explanation study would require a predeclared multi-molecule protocol and fidelity design that this phase did not add.
