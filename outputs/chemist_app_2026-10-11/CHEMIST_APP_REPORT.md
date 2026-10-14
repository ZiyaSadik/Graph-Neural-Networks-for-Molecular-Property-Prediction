# Chemist Application & Research Integration Report

**Date:** 2026-10-11
**Output directory:** `outputs/chemist_app_2026-10-11/`
**GitHub:** not modified / not committed

---

## 1. What changed

### Added
| Path | Purpose |
|---|---|
| `apps/chemist_app/app.py` | Streamlit UI |
| `apps/chemist_app/service.py` | Orchestrates existing mol_cli models |
| `apps/chemist_app/explain.py` | GNNExplainer wrapper (fail-safe) |
| `apps/chemist_app/drawing.py` | 2D structure + importance drawings |
| `apps/chemist_app/README.md` | App usage |
| `apps/mol_cli/tests/test_chemist_app_service.py` | Service/drawing/mapping tests |
| `outputs/chemist_app_2026-10-11/*` | Architecture + research summaries |
| `requirements.txt` | optional `streamlit>=1.28.0` |
| Root `README.md` | Chemist app section |

### Reused (not rewritten)
- `BacePrototypeModel`, `EsolSolubilityModel`, `Tox21MultiTaskModel`
- GraphCL checkpoint + ESOL/Tox21 heads + BACE support manifest
- Prior few-shot multi-K results from `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/`
- Prior explainability / calibration diagnostics under `phase_strengthen_2026-10-11/`

### Preserved
- All locked-test JSON/reports/checkpoints
- Existing CLI entry points

---

## 2. Experiments that ran (this session)

| Item | Result |
|---|---|
| Architecture audit | Written (`ARCHITECTURE_REPORT.md`) |
| Service smoke (`c1ccccc1`, no XAI) | BACE/ESOL/Tox21 available; 12 Tox21 tasks |
| Unit tests (`discover apps/mol_cli/tests`) | **96 tests OK** (1 skipped) |
| Few-shot multi-K | **Reused** prior executed Protocol B study (canonical consistent=True); summary copied to `fewshot_comparison_summary.json` |
| GraphCL vs No-GraphCL Protocol B | **Not run** — No-GraphCL `.pt` missing |
| Streamlit | Installed **1.65.0**; `apps.chemist_app.app` imports OK |

---

## 3. Verified few-shot comparison (Protocol B, validation)

From reused `fewshot_comparison_summary.json` (seeds 42,7,13,21,99):

| K | bal_acc mean±std | AUROC mean±std |
|---:|---|---|
| 1 | 0.5450±0.0496 | 0.5766±0.0640 |
| 3 | 0.5340±0.0601 | 0.5437±0.0853 |
| 5 | ~0.53±0.09 | ~0.58±0.08 |
| 10 | 0.5839±0.0178 | 0.6080±0.0310 |

Canonical K=5 seed=42 locked report **exactly reproduced** earlier (valid/test metrics match `bace_fixed_support_eval_v1.json`).
Plots: `plot_valid_*.png` in this folder.

---

## 4. Explainability status

- App integrates real `PrototypeDistanceLogits` + PyG GNNExplainer.
- Failures surface as errors; **no invented masks**.
- Marked **exploratory** (historical fidelity limitations documented).
- Index mapping verified in tests (`CCO` → 3 atoms/nodes).
- Tox21 XAI remains the existing 2-molecule smoke only.

---

## 5. How to run the application locally

```text
# From repo root, with checkpoints present
pip install -r requirements.txt
streamlit run apps/chemist_app/app.py
```

CLI (unchanged):

```text
python -m apps.mol_cli.predict
python -m apps.mol_cli.predict --bace
python -m apps.mol_cli.esol_infer
python -m apps.mol_cli.tox21_infer
```

---

## 6. Remaining limitations

1. No formal applicability-domain certificate for arbitrary SMILES.
2. BACE distances ≠ probabilities; Tox21 sigmoid ≠ calibrated probabilities.
3. Protocol A vs B must stay separate in any paper claim.
4. No-GraphCL Protocol B comparison blocked by missing weights.
5. GNNExplainer not fidelity-validated for routine chemist decisions.
6. `edge_attr` unused in GIN forward.

---

## 7. Proposed paper edits

See `PROPOSED_MANUSCRIPT_CHANGES.md` — manuscript **not** auto-rewritten.
