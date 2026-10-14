# Architecture Report — Chemist-Facing Application

**Date:** 2026-10-11
**Scope:** Read-only audit of existing models/inference before building the app. Locked evaluation artifacts were not modified.

---

## 1. Model used per task

| Task | Model | Head / predictor | Checkpoint(s) |
|---|---|---|---|
| Shared encoder | `GINEncoder` 5×128, sum pool, dropout 0.1 | GraphCL projection unused at eval | `outputs/core_pipeline/graphcl_checkpoint.pt` |
| BACE few-shot | Frozen GIN + Euclidean ProtoNet | Prototypes = mean support embeddings; `pred = argmin ‖z−p_c‖₂` | Encoder + `apps/mol_cli/support_manifests/bace_train_k5_v1.json` |
| ESOL | Frozen GIN + MLP regression | `Linear(128→64)→ReLU→Linear(64→1)`, denormalized | Encoder + `outputs/esol/phase4e3_frozen_gin_regression/regression_head.pt` |
| Tox21 | Frozen GIN + multi-task MLP | `Linear(128→64)→ReLU→Dropout→Linear(64→12)` + sigmoid for display | Encoder + `outputs/tox21/phase4f4_baselines/classification_head.pt` |

**On this machine:** all three checkpoints **exist**.
**Missing:** No-GraphCL encoder `.pt` (ablation CSVs only) — Protocol B GraphCL vs No-GraphCL cannot be re-run without weights.

---

## 2. SMILES → graph

`apps/mol_cli/ogb_graph.py` → `ogb.utils.mol.smiles2graph` → PyG `Data`:
- `x`: `[N, 9]` (OGB atom features)
- `edge_index`: `[2, E]` bidirectional bonds
- `edge_attr`: `[E, 3]` stored but **unused** in `GINEncoder.forward`

---

## 3. Embeddings and pooling

- Node states after GIN: `[N, 128]`
- Graph embedding (sum pool): `[B, 128]` — used by BACE/ESOL/Tox21
- Projection head: `[B, 128]` — GraphCL training only

---

## 4. Support manifests and evaluation protocols

### Protocol A (paper episodic)
- Pool: OGB BACE scaffold **test** (152)
- 2-way 5-shot 5-query, 100 episodes, seed 42
- Result: GraphCL **0.5910±0.1698**; No-GraphCL **0.6590** (same episode list)
- Artifacts: `outputs/core_pipeline/fewshot_summary.json`

### Protocol B (CLI / chemist app)
- Fixed K=5/class support from **train only**, seed 42 (`bace_train_k5_v1`)
- Valid/test scored separately; distances ≠ probabilities
- Locked report: `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json`
  - Valid bal_acc 0.6408, AUROC 0.6385
  - Test bal_acc 0.6597, AUROC 0.7279

**Not interchangeable.** The app uses Protocol B only.

### Multi-K Protocol B (new research folder)
Already executed under `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/` (1/3/5/10-shot × 5 seeds on validation). Reused; not re-tuned on locked test.

---

## 5. Explanation methods

| Method | Status |
|---|---|
| Prototype distances / gaps | **Works** — primary BACE explanation in app |
| GNNExplainer + `PrototypeDistanceLogits` | **Runs** — historically weak fidelity; mark exploratory |
| Tox21 GNNExplainer | Existing **2-molecule smoke** only; not a full study |

Index mapping: RDKit atom `i` ↔ OGB node `i` for `smiles2graph` (validated in Phase 4 tests).

---

## 6. Incompatibilities / dependencies

- Streamlit **not** previously in `requirements.txt` (added as optional app dependency).
- `edge_attr` unused → bond-mask explanations are connectivity-only.
- Dual BACE protocols must stay labeled separately in UI and papers.
- Random/untrained models are **never** substituted if a checkpoint is missing.

---

## 7. CLI preserved

```
python -m apps.mol_cli.predict
python -m apps.mol_cli.predict --bace
python -m apps.mol_cli.esol_infer
python -m apps.mol_cli.tox21_infer
```
