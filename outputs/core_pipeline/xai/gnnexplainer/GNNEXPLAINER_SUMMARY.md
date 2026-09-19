# GNNExplainer analysis (5 molecules)

Canonical frozen GraphCL `GINEncoder` from `outputs/core_pipeline/graphcl_checkpoint.pt`.
Objective: `PrototypeDistanceLogits`, `logit_c = -||z_query - p_c||_2`, prototypes from the **same** 2-way 5-shot support set (OGB BACE scaffold test, episode seed **7**). Not the projection head. Not a supervised classifier.

This is **n = 5**. It is not causal and does not generalize.

## Selection

Queries come from one deterministic episode (seed 7). Positions [0, 1, 2, 6, 8]: first three incorrect then first two correct in episode order (3 incorrect, 2 correct).

Support indices: `[162, 131, 173, 107, 112, 46, 8, 20, 93, 89]`

## Sparsity

- Soft node/edge sparsity = `1 - mean(mask)` with GNNExplainer masks in (0, 1). Larger means less average mass.
- Hard sparsity = fraction of atoms/bonds with mask `< 0.5`.

## Fidelity (GraphFramEx, PyG `fidelity`, model explanations)

PyG applies **soft** masks: `x <- node_mask * x` and MessagePassing `_edge_mask`. Then:

- **fid+**: `1 - 1[pred(graph with explanation removed) == original pred]`. 1 means the complement flips the prototype-distance class.
- **fid-**: `1 - 1[pred(explanation subgraph only) == original pred]`. 0 means the subgraph alone keeps the original class.

**Hard 0.5 companion:** same GraphFramEx 0/1 formula after `mask >= 0.5`. If no atom exceeds 0.5, the explanation subgraph is empty (all node features zeroed).

**Softmax companion:** change in softmax mass of the original predicted class after soft masking (`dp expl` = original minus explanation; `dp comp` = original minus complement). This is not a probability calibration; it only measures how the prototype-distance logits shift.

A useful explanation typically has high fid+ and low fid-. These are prediction-change checks, not causal effects.

## Results

| ID | true | pred | correct | d0 | d1 | node sp. | edge sp. | fid+ | fid- | fid+ hard | fid- hard | dp expl | dp comp |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BACE_118 | 0 | 1 | no | 77.79 | 63.08 | 0.674 | 0.685 | 0.000 | 0.000 | 0.000 | 1.000 | 0.003 | 0.000 |
| BACE_171 | 0 | 1 | no | 58.91 | 48.33 | 0.671 | 0.617 | 0.000 | 0.000 | 0.000 | 1.000 | 0.002 | -0.000 |
| BACE_111 | 0 | 1 | no | 59.37 | 50.04 | 0.671 | 0.699 | 0.000 | 0.000 | 0.000 | 1.000 | 0.003 | -0.000 |
| BACE_52 | 1 | 1 | yes | 59.45 | 52.53 | 0.652 | 0.691 | 0.000 | 0.000 | 0.000 | 1.000 | 0.001 | -0.000 |
| BACE_253 | 1 | 1 | yes | 75.66 | 61.43 | 0.672 | 0.675 | 0.000 | 0.000 | 0.000 | 1.000 | 0.003 | 0.000 |

Mean soft GraphFramEx fid+: 0.000
Mean soft GraphFramEx fid-: 0.000
Mean hard-0.5 fid+: 0.000
Mean hard-0.5 fid-: 1.000
Mean dp on explanation (soft): 0.003
Mean dp on complement (soft): -0.000
Mean node soft sparsity: 0.668
Mean edge soft sparsity: 0.674

GNNExplainer epochs: 80. PyG 2.7.0.
