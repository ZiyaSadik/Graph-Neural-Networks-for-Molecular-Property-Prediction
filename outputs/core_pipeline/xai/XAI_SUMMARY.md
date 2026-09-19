# Explainability status (canonical GraphCL GINEncoder)

Checkpoint: `outputs/core_pipeline/graphcl_checkpoint.pt`  
Encoder: 5-layer GIN, hidden 128, dropout 0.1, sum pool (`in_dim=9`).  
Predictor: Euclidean ProtoNet, original BACE labels, `pred = argmin_c d_c`.  
This is **not** the old 3-layer / `final_model.pt` / 9-of-10 diagnostic.

GraphCL was not retrained for any XAI run. Projection head is not used.

---

## 1. Prototype distances (primary XAI)

One 2-way 5-shot episode, **seed 7**, OGB BACE scaffold test. Support ∩ query = empty.

- Support indices: `[162, 131, 173, 107, 112, 46, 8, 20, 93, 89]`
- Query indices: `[116, 169, 109, 186, 150, 16, 51, 237, 226, 92]`
- Episode query accuracy: **0.300** (10 queries)
- Files: `prototype_explanations.csv`, `prototype_explanations.json`, `figures/query_*.png`
- Script: `prototype_explain.py`

Method: class prototypes are means of five support embeddings. Confidence reported as `|d0−d1|` is an embedding-space gap, **not** a probability.

| id | true | pred | d0 | d1 | gap | nearest support | cosine |
|---|---:|---:|---:|---:|---:|---|---:|
| BACE_118 | 0 | 1 | 77.7875 | 63.0840 | 14.7035 | BACE_90 | 0.7684 |
| BACE_171 | 0 | 1 | 58.9095 | 48.3306 | 10.5789 | BACE_9 | 0.7344 |
| BACE_111 | 0 | 1 | 59.3708 | 50.0401 | 9.3307 | BACE_9 | 0.7290 |
| BACE_188 | 0 | 1 | 55.9438 | 50.1474 | 5.7964 | BACE_109 | 0.7618 |
| BACE_152 | 0 | 1 | 66.6492 | 49.0474 | 17.6018 | BACE_47 | 0.7866 |
| BACE_17 | 1 | 0 | 44.5818 | 47.9467 | 3.3648 | BACE_21 | 0.9173 |
| BACE_52 | 1 | 1 | 59.4481 | 52.5275 | 6.9206 | BACE_164 | 0.7334 |
| BACE_264 | 1 | 0 | 56.2456 | 57.5435 | 1.2979 | BACE_109 | 0.6409 |
| BACE_253 | 1 | 1 | 75.6625 | 61.4335 | 14.2289 | BACE_90 | 0.7127 |
| BACE_93 | 1 | 1 | 45.9786 | 41.8587 | 4.1199 | BACE_109 | 0.8972 |

Figures (first five queries): `figures/query_0_116.png` … `query_4_150.png`.

n = 10 queries from one episode. Not a substitute for the 100-episode accuracy.

---

## 2. Distance-gap analysis (not calibrated confidence)

`CONFIDENCE_ANALYSIS.md`, `confidence_summary.json`. Seed **42**, 100 episodes, **1000** query predictions (episode list is a **resample**, not the core `fewshot_results.csv` list; overall acc **0.596**, not 0.5910).

- Mean gap correct **13.848** vs incorrect **11.152**
- Point-biserial **0.128**; AUROC **0.589**
- Quantile-bin accuracies are **not** monotone
- Do **not** compute ECE on the raw gap

---

## 3. GNNExplainer (limitation / negative result)

PyG 2.7.0 `torch_geometric.explain.Explainer` + `GNNExplainer`. Wrapper: `PrototypeDistanceLogits` (`logit_c = -d_c`). Same five queries from the seed-7 episode: **BACE_118, BACE_171, BACE_111, BACE_52, BACE_253**. `ExplainableGNN` (MLP) was **not** used.

| Run | Directory | Role |
|---|---|---|
| API smoke (BACE_118, 40 epochs) | `gnnexplainer_smoke_test/` | API check only |
| Unscaled `-d` (80 epochs) | `gnnexplainer/` | Diffuse masks; soft GraphFramEx **fid+ = fid− = 0** on all 5 |
| CE saturation diagnostic | `gnnexplainer_diagnostics/` | Raw −d softmax ~1; CE ~0; regularization dominates |
| T=10 explainer logits `-d/10` | `gnnexplainer_temperature/` | Predictions unchanged; node-mask std 0.053→0.106; atoms ≥0.5 0.014→0.294; **fid+ = fid− still 0** |
| Top-k keep/mask | `gnnexplainer_diagnostics_v2/` | Keep top 10% flips 5/5 (tiny subgraph); mask top 10% and 25% flip **0/5** |

**Paper use:** qualitative limitation of GNNExplainer on this prototype-distance classifier. Do not report unscaled or T=10 maps as quantitative atom-level BACE explanations. Soft GraphFramEx 0/0 is not a successful fidelity result.

---

## 4. Not this XAI track

- `outputs/final_results.txt` **0.5180 ± 0.1590** and `outputs/final_model.pt`: MSE-to-noise diagnostic.
- Legacy 9/10 episode: diagnostic only (`archive/README.md`).
