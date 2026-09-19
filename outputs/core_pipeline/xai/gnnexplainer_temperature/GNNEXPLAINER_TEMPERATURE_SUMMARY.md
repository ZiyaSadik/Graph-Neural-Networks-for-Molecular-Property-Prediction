# GNNExplainer temperature scaling (T=10)

Canonical frozen GraphCL GIN from `outputs/core_pipeline/graphcl_checkpoint.pt`. Classifier unchanged: `pred = argmin(d_c) = argmax(-d_c)`.
GNNExplainer sees `logit_c = -d_c / 10`. Previous `outputs/core_pipeline/xai/gnnexplainer/` files were **not** overwritten.

Same episode seed **7**, same five molecules: BACE_118, BACE_171, BACE_111, BACE_52, BACE_253.
All five T=10 argmax values matched the unscaled ProtoNet predictions.

This is **n = 5**. Not causal. Does not generalize.

## Did T=10 concentrate the masks?

- Mean node-mask **std**: 0.0531 (T=1) → 0.1055 (T=10). Higher std: **True**.
- Mean fraction of atoms ≥ 0.5: 0.014 → 0.294. Increase: **True**.
- Mean node-mask mean: 0.332 → 0.422.

## Did fidelity improve?

Soft GraphFramEx on the explainer model (argmax identical to the classifier):

- Mean fid+: 0.000 → 0.000 (higher is better). Improved: **False**.
- Mean fid−: 0.000 → 0.000 (lower is better). Improved: **False**.

Predicted-class softmax at T=10 is the explainer scale, not a calibrated probability. Classifier T=1 softmax remains saturated (~1).

## Per-molecule T=10

| ID | true | pred | node mean | node std | atoms≥0.5 | edges≥0.5 | node soft sp. | fid+ | fid− | softmax T=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BACE_118 | 0 | 1 | 0.419 | 0.100 | 0.294 | 0.263 | 0.581 | 0.000 | 0.000 | 0.813 |
| BACE_171 | 0 | 1 | 0.418 | 0.106 | 0.280 | 0.339 | 0.582 | 0.000 | 0.000 | 0.742 |
| BACE_111 | 0 | 1 | 0.424 | 0.103 | 0.333 | 0.189 | 0.576 | 0.000 | 0.000 | 0.718 |
| BACE_52 | 1 | 1 | 0.429 | 0.107 | 0.250 | 0.161 | 0.571 | 0.000 | 0.000 | 0.666 |
| BACE_253 | 1 | 1 | 0.422 | 0.112 | 0.312 | 0.264 | 0.578 | 0.000 | 0.000 | 0.806 |

## T=1 previous (same molecules)

| ID | node mean | node std | atoms≥0.5 | edges≥0.5 | node soft sp. | fid+ | fid− | softmax T=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BACE_118 | 0.326 | 0.057 | 0.000 | 0.053 | 0.674 | 0.000 | 0.000 | 1.000 |
| BACE_171 | 0.329 | 0.052 | 0.040 | 0.214 | 0.671 | 0.000 | 0.000 | 1.000 |
| BACE_111 | 0.329 | 0.051 | 0.030 | 0.041 | 0.671 | 0.000 | 0.000 | 1.000 |
| BACE_52 | 0.348 | 0.059 | 0.000 | 0.081 | 0.652 | 0.000 | 0.000 | 0.999 |
| BACE_253 | 0.328 | 0.046 | 0.000 | 0.097 | 0.672 | 0.000 | 0.000 | 1.000 |

GNNExplainer epochs: 80. PyG 2.7.0.
