# Phase 1 — Repository Audit

**Date:** 2026-10-11
**Repo:** `gnn_fewshot`
**Scope:** Read-only inspection before new experiments. No training started. Existing locked-test artifacts were not modified.

---

## 1. GIN forward pass and tensor dimensions

**Implementation:** `src/models.py` — `GINEncoder`

| Config key | Recorded / live value |
|---|---|
| `in_dim` | 9 (OGB atom features) |
| `hidden_dim` | 128 |
| `num_layers` | 5 |
| `dropout` | 0.1 |
| `pool_type` | `sum` → `global_add_pool` |
| Parameter count | **184,197** |

**Forward (simplified):**
1. `x = data.x` (cast to float if integer); `edge_index = data.edge_index`.
2. **`edge_attr` is never read.**
3. Five GINConv+BN+ReLU+Dropout layers → node states `[N, 128]`.
4. Sum-pool → `graph_embedding` `[B, 128]`.
5. Projection head → `projection` `[B, 128]` (GraphCL contrastive path).
6. Optional `return_node_embeddings=True` returns final node tensor `[N, 128]`.

**Live probe (ethanol `CCO`, GraphCL checkpoint, eval mode):**
- nodes: `(3, 128)`, graph emb: `(1, 128)`, projection: `(1, 128)`, all finite.

**Downstream usage:** BACE ProtoNet / fixed-support inference uses **`graph_embedding` (index 1)**, not the projection head.

---

## 2. Checkpoint ↔ architecture compatibility

| Item | Value |
|---|---|
| Path | `outputs/core_pipeline/graphcl_checkpoint.pt` |
| SHA-256 | `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f` |
| Keys | `encoder_config`, `encoder_state_dict`, pretrain metadata |
| Load API | `GINEncoder.from_checkpoint` / `apps/mol_cli/checkpoint_smoke.py` |

**Verdict:** Checkpoint loads into the intended 5-layer GIN. Missing `encoder_config` is refused (protects against old 3-layer weights).

---

## 3. Few-shot support/query evaluation protocols

Two **non-interchangeable** BACE protocols exist:

### Protocol A — episodic ProtoNet (paper / `main.py`)
- Pool: OGB BACE scaffold **test** only (n=152).
- Episode: 2-way, K=5, Q=5, Euclidean, temperature 1.0.
- Episodes: 100, seed 42.
- Recorded: mean accuracy **0.5910 ± 0.1698** (`outputs/core_pipeline/fewshot_summary.json`).
- No-GraphCL same episodes: **0.6590** (GraphCL underperforms on this protocol).

### Protocol B — fixed train support (CLI eval)
- Manifest: `apps/mol_cli/support_manifests/bace_train_k5_v1.json` (K=5/class from **train only**, seed 42).
- Prototypes = mean GIN embeddings per label; pred = argmin Euclidean distance.
- Valid: bal_acc 0.6408, AUROC 0.6385; Test: acc 0.6447, AUROC 0.7279.
- Report: `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json`.

**Rule for new work:** Do not mix Protocol A and B numbers as if they were the same experiment.

---

## 4. Explanation methods that execute

| Method | Runs? | Scientific status |
|---|---|---|
| Prototype distances (`PrototypeExplainer`, `prototype_explain.py`) | Yes | Primary qualitative XAI; gaps ≠ probabilities |
| Confidence / gap analysis | Yes | Modest discrimination (AUROC ≈ 0.589); not calibration |
| GNNExplainer BACE wrappers | API smoke succeeds | Soft fidelity often uninformative (fid±≈0); limitation |
| Tox21 GNNExplainer smoke | Yes (2 molecules) | Runnable masks; not a comprehensive study |
| `ExplainableGNN` MLP wrapper | Unused for BACE XAI | Do not cite as the BACE explanation path |

---

## 5. Data and dependencies available (this machine)

- **Python / stack:** See `environment.txt` / `requirements.txt` (torch, PyG, ogb, rdkit, sklearn, …).
- **Checkpoint + BACE/ESOL/Tox21 caches:** Present locally; `*.pt` is gitignored for clones.
- **Locked-test JSON:** `outputs/esol/phase4e_test_locked/`, `outputs/tox21/phase4f_test_locked/` — treat as frozen.
- **Tests:** `apps/mol_cli/tests/` (checkpoint smoke, ogb graph, BACE support/eval/infer/CLI).

---

## 6. Implementation problems for rigorous evaluation

1. Dual BACE protocols — must keep separate.
2. GraphCL underperforms No-GraphCL on recorded episodic Protocol A.
3. `edge_attr` stored but unused in GIN forward (edge-mask XAI is connectivity-only).
4. Small episodic pool (152) → high episode variance.
5. GNNExplainer fidelity not scientifically successful for ProtoNet logits.
6. Quarantined `experiment.py` / historical MSE-noise weights must not be used.
7. Checkpoint distribution gap for fresh clones.
8. No end-to-end multi-property product validation — tracks share encoder only.

---

## Phase gate

Phase 1 complete. Prerequisites for Phase 2 (checkpoint load, graph construction, embedding shapes) are verified. New experiments will write only under `outputs/phase_strengthen_2026-10-11/`.
