# Phase Strengthen Report (2026-10-11)

**Project:** Graph Neural Networks with Few-Shot Learning: An Explainable Framework for Molecular Property Prediction
**New experiment root:** `outputs/phase_strengthen_2026-10-11/`
**GitHub / commits:** none (explicitly not modified)

---

## 1. What was already implemented

- GraphCL–GIN encoder (5×128, sum pool, 184,197 params) and checkpoint `outputs/core_pipeline/graphcl_checkpoint.pt`
- Protocol A: episodic ProtoNet on BACE test pool (`fewshot_summary.json`, mean acc 0.5910±0.1698)
- Protocol B: fixed train K=5 support CLI eval (`bace_fixed_support_eval_v1.json`)
- ESOL / Tox21 locked-test tracks under `outputs/esol/`, `outputs/tox21/`
- Prototype-distance XAI + GNNExplainer smoke / limitation docs under `outputs/core_pipeline/xai/`
- Tox21 2-molecule GNNExplainer smoke under `outputs/tox21/phase4f6_explainability/`
- Unit tests under `apps/mol_cli/tests/`

---

## 2. What was changed

**Added (no overwrite of locked artifacts):**

| Path | Role |
|---|---|
| `diagnostics/gin_embedding_validation.py` | Phase 2 embedding diagnostics |
| `diagnostics/fewshot_bace_multishot.py` | Phase 3 multi-K Protocol B evaluation |
| `diagnostics/bace_explainability_eval.py` | Phase 4 BACE explanation eval |
| `diagnostics/bace_calibration_assessment.py` | Phase 5 calibration assessment |
| `diagnostics/__init__.py` | Package marker |
| `apps/mol_cli/tests/test_gin_embedding_validation.py` | New tests |
| `apps/mol_cli/tests/test_fewshot_bace_multishot.py` | New tests |
| `apps/mol_cli/tests/test_bace_explainability_eval.py` | New tests |
| `outputs/phase_strengthen_2026-10-11/**` | All new reports, JSON, figures |

**Unchanged (verified still present; not overwritten by these scripts):**

- `outputs/core_pipeline/graphcl_checkpoint.pt` (SHA `d6951b12…`)
- `apps/mol_cli/support_manifests/bace_train_k5_v1.json`
- `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json`
- `outputs/core_pipeline/fewshot_summary.json`
- Manuscript DOCX/PDF outside the repo (not modified)

---

## 3. Experiments that actually ran

| Phase | Status | Key output |
|---|---|---|
| 1 Audit | Done | `PHASE1_AUDIT.md` |
| 2 GIN embedding validation | Done (`overall_pass=True`) | `phase2_gin_embedding/` |
| 3 Few-shot BACE (Protocol B family) | Done (`canonical_consistent=True`) | `phase3_fewshot_bace/` |
| 4 BACE explainability | Done (8/8 GNNExplainer runs; mapping OK) | `phase4_explainability/` |
| 5 Calibration assessment | Done (`claim_calibrated_probabilities=False`) | `phase5_calibration/` |
| 6 Regression tests | Done | See §6 |

---

## 4. Exact verified results

### Phase 2 — embeddings
- Checkpoint SHA: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Node emb `[N,128]`, graph emb `[1,128]` / batch `[B,128]`, finite, eval-repeatable
- Min pairwise L2 across 6 diagnostic molecules: **6.53** (not constant)
- **Interpretation:** implementation checks only — not chemical proof

### Phase 3 — few-shot (Protocol B)
**Canonical K=5 seed=42 reproduction (exact match to locked report):**

| Split | bal_acc | sens | spec | AUROC |
|---|---:|---:|---:|---:|
| valid | 0.6408 | 0.3769 | 0.9048 | 0.6385 |
| test | 0.6597 | 0.4321 | 0.8873 | 0.7279 |

**Multi-K validation (seeds 42,7,13,21,99; n_query=151; mean±std):**

| K | bal_acc | AUROC |
|---:|---|---|
| 1 | 0.5450±0.0496 | 0.5766±0.0640 |
| 3 | 0.5340±0.0601 | 0.5437±0.0853 |
| 5 | ~0.53±0.09 | ~0.58±0.08 |
| 10 | 0.5839±0.0178 | 0.6080±0.0310 |

Prototypes = mean GIN graph embeddings per original label; pred = argmin Euclidean distance; score = d0−d1 (**not** a probability).
No-GraphCL Protocol B comparison: **not run** (no No-GraphCL checkpoint on disk).

### Phase 4 — explainability
- Predeclared valid indices (stratified): first 4 inactive + 4 active by sorted index
- Index mapping RDKit↔OGB: **OK** for all 8
- GNNExplainer via `PrototypeDistanceLogits`: 8/8 executed; figures under `phase4_explainability/figures/`
- Fidelity uses feature-zeroing (explicitly **not** causal / not chemically valid ablation)
- Tox21: **not extended** beyond existing 2-molecule smoke

### Phase 5 — calibration
- Platt fit on train∖support; eval on validation only; test unused
- **Do not claim calibrated probabilities** (imbalance, non-generative scores, weak prior gap AUROC)

---

## 5. New tables and figures

- Phase 3 plots: `phase3_fewshot_bace/plot_valid_*.png`
- Phase 4 atom-importance figures: `phase4_explainability/figures/bace_*_atom_importance.png`
- Phase 5 reliability diagrams: `phase5_calibration/reliability_*.png`
- Machine-readable JSON beside each phase MD report

---

## 6. Tests run and outcomes

```
python -m unittest discover -s apps/mol_cli/tests -v
→ Ran 89 tests in ~1.6s — OK (skipped=1)
```

New modules covered: embedding validation, multishot helpers, explainability mapping.
CLI / checkpoint load / BACE infer smoke: OK after Phase 6 checks.

---

## 7. Limitations and unresolved issues

1. Dual BACE protocols remain non-interchangeable (A episodic vs B fixed-support).
2. GraphCL underperforms No-GraphCL on recorded Protocol A; Protocol B No-GraphCL weights absent.
3. `edge_attr` unused in GIN forward.
4. Validation imbalance (21 vs 130) limits multi-K specificity/ECE reliability.
5. GNNExplainer on prototype-distance logits remains a weak fidelity setting historically and here.
6. Calibration not scientifically claimable for this score type.
7. No new GraphCL training / no ADMET expansion / no clinical claims.

---

## 8. Suggested paper updates (not applied)

These are **suggestions only**; the manuscript was not modified.

1. Add a short “Encoder implementation checks” paragraph citing Phase 2 (shapes, finite, non-constant; not chemical validation).
2. Clarify Protocol A vs Protocol B tables; add Protocol B multi-K **validation** table with seeds, without mixing episodic accuracy.
3. Strengthen explainability section with the predeclared validation set + fidelity/random baseline caveats; keep GNNExplainer as limitation.
4. State explicitly that prototype scores are **not** calibrated probabilities (Phase 5).
5. Do not claim GraphCL superiority on BACE few-shot without a compatible No-GraphCL Protocol B checkpoint comparison.

---

## How to re-run

```bash
python diagnostics/gin_embedding_validation.py
python diagnostics/fewshot_bace_multishot.py
python diagnostics/bace_explainability_eval.py
python diagnostics/bace_calibration_assessment.py
python -m unittest discover -s apps/mol_cli/tests -v
```
