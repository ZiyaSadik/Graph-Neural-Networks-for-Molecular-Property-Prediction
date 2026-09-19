# GNNExplainer diagnostic (1 molecule, not a paper result)

Molecule `BACE_118` from episode seed 7, query position 0.
Canonical frozen GraphCL GIN. No GraphCL retrain. Existing `gnnexplainer/` files were not modified.

## 1. Likely cause of diffuse masks

Raw prototype logits are `-distance` with distances ~77.8 / 63.1.
Predicted-class softmax = **1**, cross-entropy = **3.58e-07**.

GNNExplainer's objective is CE on these raw logits plus size/entropy regularizers
(`node_feat_size=1.0`, `edge_size=0.005`). With CE already ~0, the prediction term has
almost no gradient, so regularization pushes every mask logit in the same direction.
That matches the 5-molecule run (node means ~0.33, std ~0.03, almost nothing ≥ 0.5).

- Primary: raw -distance logits saturate softmax (CE ~ 0), so GNNExplainer optimization is dominated by size/entropy regularization and drives masks uniformly down.
- Edge masks do change GIN logits; missing edge_weight in GINConv is not the main failure.
- Uniform soft feature scaling (~0.33) does not flip the class, matching zero soft fidelity; empty features do flip, matching hard fid-=1.

## 2. Is the current explanation objective appropriate?

The **wrapper** is appropriate: `PrototypeDistanceLogits` matches ProtoNet (`argmax -d_c`).
Using GNNExplainer **cross-entropy on unscaled -d** is not appropriate here. Distances of
order 50–80 make softmax a step function, so CE cannot ask the mask to preserve a decision
that is already trivially preserved under almost any soft mask.

## 3. Smallest scientifically defensible fix

Keep the classifier. Change only the **explainer** input to `logit_c = -d_c / T` with T > 0.
Argmax is unchanged. T is an explainer hyperparameter, not a new model.

Current T=1 node mask std = 0.0248, frac ≥ 0.5 = 0.000.
T=10 node mask std = 0.0484, frac ≥ 0.5 = 0.294.
Temperature helped: **True**.

Do not treat 0.5 hard GraphFramEx as the main metric until masks actually concentrate.
Soft GraphFramEx will stay 0 whenever uniform down-scaling of features does not flip class.

## 4. Keep or discard the current 5-molecule figures?

**Do not use them as positive chemical explanations.** They are a valid record that
unscaled -distance GNNExplainer failed to sparsify. For the paper, either (a) label them
as a failed baseline / diagnostic, or (b) replace after a temperature-scaled rerun on
the same 5 molecules. Do not add more molecules until the objective is fixed.

## Edge masking

Zeros vs ones max |Δ logit| = 2.878.
GIN `_edge_mask` is used.

## Feature masking vs original class 1

| mask | kept | pred | flipped | softmax(orig class) |
|---|---:|---:|:---:|---:|
| all ones | 1.00 | 1 | False | 1.0000 |
| all 0.33 | 0.33 | 1 | False | 0.9991 |
| random half | 0.38 | 1 | False | 1.0000 |
| first quarter | 0.24 | 0 | True | 0.4116 |
| all zeros | 0.00 | 0 | True | 0.0433 |

## Optimization traces (25 epochs)

### Current T=1
| epoch | CE | node_reg | node_mean | node_std | softmax | CE-grad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.338e-06 | 0.5654 | 0.494 | 0.0243 | 0.999997 | 2.93e-06 |
| 5 | 9.179e-06 | 0.5529 | 0.481 | 0.0242 | 0.999991 | 1.003e-05 |
| 10 | 5.138e-05 | 0.5404 | 0.469 | 0.0242 | 0.999949 | 6.518e-05 |
| 15 | 0.0005913 | 0.5278 | 0.457 | 0.0241 | 0.999409 | 0.001025 |
| 20 | 0.002573 | 0.5158 | 0.445 | 0.0242 | 0.997430 | 0.002033 |
| 24 | 0.003736 | 0.5071 | 0.437 | 0.0248 | 0.996271 | 0.004003 |

### T=10
| epoch | CE | node_reg | node_mean | node_std | softmax | CE-grad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2497 | 0.5654 | 0.495 | 0.0244 | 0.779060 | 0.01914 |
| 5 | 0.2125 | 0.5592 | 0.489 | 0.0265 | 0.808544 | 0.01362 |
| 10 | 0.1953 | 0.553 | 0.483 | 0.0312 | 0.822606 | 0.01073 |
| 15 | 0.1826 | 0.5475 | 0.478 | 0.0376 | 0.833097 | 0.00733 |
| 20 | 0.1751 | 0.5418 | 0.472 | 0.0439 | 0.839342 | 0.004512 |
| 24 | 0.1714 | 0.5368 | 0.467 | 0.0484 | 0.842494 | 0.004498 |

### T = mean distance (70.44)
| epoch | CE | node_reg | node_mean | node_std | softmax | CE-grad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6077 | 0.5654 | 0.494 | 0.0243 | 0.544610 | 0.0056 |
| 5 | 0.5984 | 0.557 | 0.486 | 0.0262 | 0.549673 | 0.005105 |
| 10 | 0.5935 | 0.5496 | 0.479 | 0.0306 | 0.552383 | 0.005341 |
| 15 | 0.5893 | 0.5427 | 0.472 | 0.0361 | 0.554709 | 0.003494 |
| 20 | 0.5873 | 0.5351 | 0.465 | 0.0415 | 0.555816 | 0.003029 |
| 24 | 0.587 | 0.5288 | 0.459 | 0.0458 | 0.555976 | 0.004062 |
