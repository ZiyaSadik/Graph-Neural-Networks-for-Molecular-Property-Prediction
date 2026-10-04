# Phase 4F.6 — Tox21 GNNExplainer Smoke Test

**Status:** All requested explanations produced node and edge masks.
**Created UTC:** 2026-10-04T09:30:45+00:00

## Scope

- New artifacts only under `outputs/tox21/phase4f6_explainability/`
- No training / fine-tuning; encoder frozen + eval
- Validation graph-eligible molecules only; **test split not accessed**
- Shared `src/`, splits, datasets, checkpoints, and Phase 4F.4 artifacts unmodified

## Checks

| Check | Result |
|---|---|
| Source SHA-256 | `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761` |
| Mapping fingerprint | `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd` |
| Encoder load (weights_only safe path) | OK (`d6951b12412cbab9…`) |
| Head load (Phase 4F.4 state_dict) | OK (`strict=True`, keys `net.0` / `net.3`) |
| Adapter returns shape `[1, 12]` | OK |
| Logit match vs 4F.4 path (n=783, atol=1e-05, rtol=1e-05) | **PASS** (max abs diff `1.144e-05`) |
| TaskLogitWrapper column `2` (`NR-AhR`) | PASS |
| GNNExplainer smokes succeeded | **2/2** |

## Task targeting (verified against installed PyG 2.7.0)

Installed `ModelMode` values: `binary_classification`, `multiclass_classification`, `regression` (no multilabel).

`Explainer(..., index=…)` selects the **first dimension** of the model output (batch/graph index), confirmed in `Explainer.__call__` docs and `GNNExplainer._train` (`y_hat[index]`). It **cannot** select a Tox21 task among 12 logits.

**Method used:** `Tox21MultiTaskAdapter` returns all 12 logits; `TaskLogitWrapper` exposes `logits[:, 2]` (`NR-AhR`); Explainer uses `mode="binary_classification"`, `return_type="raw"`, `explanation_type="model"`.

**Rejected:** multiclass over 12 task logits (would imply mutually exclusive tasks).

## Molecules explained

Task: **NR-AhR** (index **2**)

| row_index | mol_id | task logit | pred active | GT label (unused) | node_mask shape | edge_mask shape | success |
|---|---|---:|---|---|---|---|---|
| 3254 | `TOX918` | 4.5232 | True | 1 | `[28, 1]` | `[64]` | yes |
| 4051 | `TOX5065` | -38.0078 | False | None | `[85, 1]` | `[190]` | yes |

Ground-truth labels are recorded when present but **not** used for explanation. Missing labels remain masked / null.

## Limitations

- Masks are explainer attributions for the binary task-logit objective, **not** chemical causality.
- Encoder **ignores `edge_attr`**; edge masks are connectivity-level only, not bond-feature attribution.
- Smoke test only (2 molecules, 40 explainer epochs).
- No test-set evaluation.

## Artifacts

| File | Role |
|---|---|
| `run_tox21_gnnexplainer_smoke.py` | Reproducible smoke script |
| `smoke_metadata.json` | Machine-readable results |
| `PHASE4F6_REPORT.md` | This report |

## Warnings captured

[
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.",
  "Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15."
]
