# Phase 2 — GIN Embedding Validation

**Report ID:** `gin_embedding_validation_v1`
**Created (UTC):** 2026-10-11T08:56:19.475137+00:00
**Overall pass:** True

## Interpretation limits

- These are **implementation diagnostics**, not evidence that embeddings encode specific chemistry.
- A nonzero distance between different molecules only shows the encoder is not constant.
- Individual latent dimensions are **not** interpreted as chemical properties here.
- GIN forward uses `x` and `edge_index` only; `edge_attr` is unused.

## Checkpoint

- Path: `outputs/core_pipeline/graphcl_checkpoint.pt`
- SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Config: `{'in_dim': 9, 'hidden_dim': 128, 'num_layers': 5, 'dropout': 0.1, 'pool_type': 'sum'}`
- Parameters: 184197 (expected 184197)

## Checks

- `checkpoint_loads`: **True**
- `architecture_compatible`: **True**
- `node_emb_shape_matches_hidden`: **True**
- `graph_emb_shape_single`: **True**
- `graph_emb_shape_batch`: **True**
- `all_finite`: **True**
- `eval_repeatable`: **True**
- `not_constant_across_molecules`: **True**
- `not_numerically_degenerate`: **True**

## Per-molecule graph embeddings

| ID | SMILES | N | ‖emb‖₂ | mean | std | finite | repeat Δ |
|---|---|---:|---:|---:|---:|---|---:|
| ethanol | `CCO` | 3 | 9.7102 | 0.3397 | 0.7882 | True | 0.00e+00 |
| benzene | `c1ccccc1` | 6 | 9.2852 | 0.2763 | 0.7728 | True | 0.00e+00 |
| acetic_acid | `CC(=O)O` | 4 | 10.5209 | 0.4105 | 0.8344 | True | 0.00e+00 |
| aniline | `Nc1ccccc1` | 7 | 9.6769 | 0.4136 | 0.7487 | True | 0.00e+00 |
| glucose_open | `OCC(O)C(O)C(O)C(O)C=O` | 12 | 41.6559 | 1.5001 | 3.3625 | True | 0.00e+00 |
| caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | 14 | 37.6633 | 1.1184 | 3.1355 | True | 0.00e+00 |

## Cross-molecule L2 distances

- min / mean / max pairwise L2: 6.529521 / 28.913618 / 54.807750
- near-duplicates (threshold 0.0001): 0
- embedding-matrix std: 2.037140
- fraction of dims with near-zero std: 0.2500

## Batch check

- batch graph emb shape: `[6, 128]`
- max |batch − single|: 1.91e-06
