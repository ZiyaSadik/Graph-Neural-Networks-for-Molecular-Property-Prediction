# Tox21 locked TEST evaluation

**Status:** One-shot test scoring complete. No retraining. No test-based selection.
**Created UTC:** 2026-10-04T09:39:28+00:00

## Pre-flight

- `verify_tox21_split_frozen_v1.py`: PASS
- Mapping fingerprint: `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd`
- Source SHA-256: `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761`
- Encoder SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Classification head SHA-256: `7a201bcd2bc50c44d79c6e3fe719354b002e5c16234921c0213a6623badff42d` (best validation epoch 74)
- Validation macro ROC-AUC repro before test: **PASS** (0.768744)

## Exclusions / census

| Item | n |
|---|---:|
| Test rows (split census) | 784 |
| Graph-eligible scored | 781 |
| Graph-ineligible excluded from graph scoring | 3 |

Policy: `retain_all_rows_mark_graph_ineligible_v1`.

## Aggregates (graph-eligible test)

| Model / metric | Value |
|---|---|
| Majority macro accuracy | 0.9367 |
| Majority ROC-AUC | null (constant) |
| Frozen GIN macro ROC-AUC (12 tasks) | 0.7282848816559224 |
| Frozen GIN macro AP (12 tasks) | 0.2844318916053813 |
| Reference valid macro ROC-AUC | 0.7687 |

Macro = unweighted mean over tasks with defined ROC-AUC/AP (both classes among labeled eligible examples); else null.

## Per-task test

| Task | Maj acc | GIN ROC-AUC | GIN AP | +/− labeled |
|---|---:|---:|---:|---:|
| NR-AR | 0.9779 | 0.6799 | 0.2551 | 16/708 |
| NR-AR-LBD | 0.9832 | 0.5659 | 0.1612 | 11/645 |
| NR-AhR | 0.9287 | 0.8232 | 0.3817 | 48/625 |
| NR-Aromatase | 0.9648 | 0.8199 | 0.3362 | 21/575 |
| NR-ER | 0.8901 | 0.6308 | 0.2536 | 68/551 |
| NR-ER-LBD | 0.9626 | 0.6606 | 0.2691 | 26/670 |
| NR-PPAR-gamma | 0.9651 | 0.7013 | 0.1728 | 22/609 |
| SR-ARE | 0.8396 | 0.7610 | 0.3977 | 89/466 |
| SR-ATAD5 | 0.9690 | 0.7040 | 0.1304 | 22/687 |
| SR-HSE | 0.9421 | 0.7126 | 0.2268 | 36/586 |
| SR-MMP | 0.8832 | 0.8534 | 0.5069 | 68/514 |
| SR-p53 | 0.9342 | 0.8268 | 0.3217 | 45/639 |

## Governance

- Test data not used for model selection, tuning, or early stopping
- Missing labels masked
- Validation artifacts and frozen split files not modified
