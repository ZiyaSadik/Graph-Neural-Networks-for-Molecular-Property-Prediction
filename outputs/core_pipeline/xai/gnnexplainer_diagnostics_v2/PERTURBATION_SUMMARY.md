# T=10 GNNExplainer top-k perturbation diagnostic

Canonical frozen GraphCL GIN. Classifier: `pred = argmin(d_c)`. Masks from `outputs/core_pipeline/xai/gnnexplainer_temperature/` (T=10). Existing GNNExplainer folders were not modified.

For each molecule, atoms and directed edges are ranked by the saved GNNExplainer scores. Top *k%* uses `ceil(k * n)` (at least 1). **Keep** zeros everything outside that set. **Mask** zeros the top set and keeps the rest. Node features are multiplied by 0/1; edges use PyG `_edge_mask`.

n = 5. Not causal.

## Verdict

The T=10 masks contain **some** decision-relevant information: at least one keep/remove setting flipped `argmin(d_c)`. GraphFramEx fid+/fid− = 0 therefore understates sensitivity. Still n=5 and not causal.

**Paper use:** Do **not** treat GNNExplainer as a finished quantitative result. Soft GraphFramEx remains uninformative. Top-k perturbations can be reported as a limited sensitivity check, not as faithful chemical explanations.

Flips in 30 keep/mask tests: **10**. Keep-top flips: 8. Mask-top flips: 2.

- **BACE_118**: original pred 1; flips: keep_top_10pct, keep_top_25pct
- **BACE_171**: original pred 1; flips: keep_top_10pct, keep_top_25pct, mask_top_50pct
- **BACE_111**: original pred 1; flips: keep_top_10pct, mask_top_50pct
- **BACE_52**: original pred 1; flips: keep_top_10pct
- **BACE_253**: original pred 1; flips: keep_top_10pct, keep_top_25pct

## Full graph

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 77.79 | 63.08 | 1 | no | 34 | 76 |
| BACE_171 | 1 | 58.91 | 48.33 | 1 | no | 25 | 56 |
| BACE_111 | 1 | 59.37 | 50.04 | 1 | no | 33 | 74 |
| BACE_52 | 1 | 59.45 | 52.53 | 1 | no | 28 | 62 |
| BACE_253 | 1 | 75.66 | 61.43 | 1 | no | 32 | 72 |

## Keep top 10%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 63.35 | 75.38 | 0 | yes | 4 | 8 |
| BACE_171 | 1 | 67.21 | 74.06 | 0 | yes | 3 | 6 |
| BACE_111 | 1 | 67.01 | 76.04 | 0 | yes | 4 | 8 |
| BACE_52 | 1 | 75.31 | 79.24 | 0 | yes | 3 | 7 |
| BACE_253 | 1 | 69.59 | 77.31 | 0 | yes | 4 | 8 |

## Keep top 25%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 65.30 | 68.30 | 0 | yes | 9 | 19 |
| BACE_171 | 1 | 49.87 | 53.25 | 0 | yes | 7 | 14 |
| BACE_111 | 1 | 64.49 | 63.06 | 1 | no | 9 | 19 |
| BACE_52 | 1 | 60.10 | 59.36 | 1 | no | 7 | 16 |
| BACE_253 | 1 | 62.59 | 65.95 | 0 | yes | 8 | 18 |

## Keep top 50%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 85.25 | 76.38 | 1 | no | 17 | 38 |
| BACE_171 | 1 | 52.21 | 51.71 | 1 | no | 13 | 28 |
| BACE_111 | 1 | 52.52 | 50.31 | 1 | no | 17 | 37 |
| BACE_52 | 1 | 56.13 | 54.15 | 1 | no | 14 | 31 |
| BACE_253 | 1 | 64.27 | 58.35 | 1 | no | 16 | 36 |

## Mask top 10%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 78.94 | 65.12 | 1 | no | 30 | 68 |
| BACE_171 | 1 | 59.56 | 49.47 | 1 | no | 22 | 50 |
| BACE_111 | 1 | 57.60 | 51.03 | 1 | no | 29 | 66 |
| BACE_52 | 1 | 56.97 | 52.18 | 1 | no | 25 | 55 |
| BACE_253 | 1 | 76.12 | 63.38 | 1 | no | 28 | 64 |

## Mask top 25%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 66.69 | 55.95 | 1 | no | 25 | 57 |
| BACE_171 | 1 | 72.96 | 67.91 | 1 | no | 18 | 42 |
| BACE_111 | 1 | 66.24 | 63.13 | 1 | no | 24 | 55 |
| BACE_52 | 1 | 66.44 | 59.61 | 1 | no | 21 | 46 |
| BACE_253 | 1 | 71.27 | 65.47 | 1 | no | 24 | 54 |

## Mask top 50%

| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |
|---|---:|---:|---:|---:|:---:|---:|---:|
| BACE_118 | 1 | 69.27 | 68.55 | 1 | no | 17 | 38 |
| BACE_171 | 1 | 66.56 | 68.35 | 0 | yes | 12 | 28 |
| BACE_111 | 1 | 67.41 | 68.65 | 0 | yes | 16 | 37 |
| BACE_52 | 1 | 68.31 | 67.49 | 1 | no | 14 | 31 |
| BACE_253 | 1 | 70.15 | 68.60 | 1 | no | 16 | 36 |
