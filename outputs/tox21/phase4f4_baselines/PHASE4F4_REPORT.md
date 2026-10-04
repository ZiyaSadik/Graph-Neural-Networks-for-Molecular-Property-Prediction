# Phase 4F.4 — Tox21 Classification Baselines

**Status:** Validation-only baselines complete. Encoder not fine-tuned. No test metrics.

## Verifier

Pre- and post-run: `python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py`
Mapping fingerprint: `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd` (matches audit).

## Configuration (approved)

| Item | Value |
|---|---|
| Encoder | Frozen GraphCL GIN, 128-d, eval, requires_grad=False |
| Head | Linear(128→64)→ReLU→Dropout(0.1)→Linear(64→12) |
| Loss | Masked BCEWithLogits (no pos_weight) |
| Optimizer | Adam lr=1e-3, wd=1e-4, batch=64 |
| Seed / device | 42 / CPU |
| Max epochs / patience | 100 / 15 on valid macro ROC-AUC |
| Best epoch | 74 |
| Graph API | `smiles_to_pyg_data` (node dim 9; edge_attr unused by GIN) |

## Graph eligibility

Policy: `retain_all_rows_mark_graph_ineligible_v1`
Eligible train/valid: **6259 / 783**
Ineligible train/valid/test (accounted): **5 / 0 / 3**

## Majority-class baseline (validation)

Macro accuracy: **0.9361**
Constant-score ROC-AUC: **null** for all tasks.

| Task | Maj | Valid +/− | Prevalence | Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---|
| NR-AR | 0 | 17/714 | 0.0233 | 0.9767 | null |
| NR-AR-LBD | 0 | 23/651 | 0.0341 | 0.9659 | null |
| NR-AhR | 0 | 44/607 | 0.0676 | 0.9324 | null |
| NR-Aromatase | 0 | 26/554 | 0.0448 | 0.9552 | null |
| NR-ER | 0 | 59/547 | 0.0974 | 0.9026 | null |
| NR-ER-LBD | 0 | 19/658 | 0.0281 | 0.9719 | null |
| NR-PPAR-gamma | 0 | 19/630 | 0.0293 | 0.9707 | null |
| SR-ARE | 0 | 90/489 | 0.1554 | 0.8446 | null |
| SR-ATAD5 | 0 | 32/668 | 0.0457 | 0.9543 | null |
| SR-HSE | 0 | 44/593 | 0.0691 | 0.9309 | null |
| SR-MMP | 0 | 58/496 | 0.1047 | 0.8953 | null |
| SR-p53 | 0 | 45/627 | 0.0670 | 0.9330 | null |

## Frozen GIN + multi-task head (validation)

| Aggregate | Value |
|---|---|
| Macro ROC-AUC (12 tasks) | **0.7687** |
| Macro AP (12 tasks) | **0.2984** |

| Task | Valid +/− | ROC-AUC | AP | Acc@0 |
|---|---:|---:|---:|---:|
| NR-AR | 17/714 | 0.6246 | 0.1816 | 0.9740 |
| NR-AR-LBD | 23/651 | 0.8093 | 0.3400 | 0.9674 |
| NR-AhR | 44/607 | 0.8798 | 0.3696 | 0.9217 |
| NR-Aromatase | 26/554 | 0.7814 | 0.2271 | 0.9431 |
| NR-ER | 59/547 | 0.6449 | 0.1851 | 0.8878 |
| NR-ER-LBD | 19/658 | 0.7771 | 0.1007 | 0.9645 |
| NR-PPAR-gamma | 19/630 | 0.8671 | 0.4964 | 0.9753 |
| SR-ARE | 90/489 | 0.7653 | 0.4398 | 0.8549 |
| SR-ATAD5 | 32/668 | 0.7300 | 0.1893 | 0.9557 |
| SR-HSE | 44/593 | 0.7187 | 0.2856 | 0.9294 |
| SR-MMP | 58/496 | 0.8564 | 0.5344 | 0.9206 |
| SR-p53 | 45/627 | 0.7705 | 0.2313 | 0.9286 |

## Provenance

- Source SHA-256: `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761`
- Mapping fingerprint: `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd`
- Pretrained checkpoint SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Embedding recompute max abs diff: 0.000e+00

## Artifacts

| File | Role |
|---|---|
| `results.json` | Machine-readable results |
| `PHASE4F4_REPORT.md` | This report |
| `training_history.json` | Epoch history |
| `classification_head.pt` | Best validation-selected head |
| `evaluate_tox21_baselines.py` | Recompute validation metrics |

## Limitations

- Encoder frozen; not fine-tuned.
- Bond `edge_attr` unused by GIN forward.
- Validation positives are sparse on several tasks → high metric variance.
- No test evaluation in this phase.

## Scope

No modifications to `main.py`, `src/`, CLI, BACE, ESOL, frozen Tox21 split/source, or pretrained checkpoint bytes.
