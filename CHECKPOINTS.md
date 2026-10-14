# Checkpoint distribution (weights not in Git)

`.gitignore` excludes `*.pt`. A normal `git clone` therefore **does not** include model weights. Research summaries, CSV/JSON metrics, and the BACE support manifest **are** in the repository.

## Required for learned inference / chemist app

Place these files at the exact repository-relative paths (create parent folders if needed):

| Path | Approx. size | Role | SHA-256 |
|---|---:|---|---|
| `outputs/core_pipeline/graphcl_checkpoint.pt` | ~744 KB | Shared GraphCL GIN encoder | `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f` |
| `outputs/esol/phase4e3_frozen_gin_regression/regression_head.pt` | ~36 KB | ESOL regression head | `368b49751ca56e14b87778af9203eaf4f487c2ea0f503bd994757ef3a7a2cdcb` |
| `outputs/tox21/phase4f4_baselines/classification_head.pt` | ~39 KB | Tox21 12-task head | `7a201bcd2bc50c44d79c6e3fe719354b002e5c16234921c0213a6623badff42d` |

## Included in Git (no separate download)

| Path | Role |
|---|---|
| `apps/mol_cli/support_manifests/bace_train_k5_v1.json` | Protocol B fixed train support (K=5, seed 42) |
| `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` | Locked Protocol B evaluation report |
| `outputs/**/*.json`, `*.csv`, `*.md` (tracked) | Recorded metrics and write-ups |

## How teammates obtain the `.pt` files

1. **Preferred:** project lead publishes a **GitHub Release** (or shared drive) containing the three files above, after confirming redistribution rights.
2. Copy/download each file into the matching path under a local clone of this repo.
3. Verify the GraphCL encoder hash if needed:
   ```text
   python -c "import hashlib; from pathlib import Path; p=Path('outputs/core_pipeline/graphcl_checkpoint.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
   ```

Until the weights are placed locally, descriptor-only CLI still works; BACE / ESOL / Tox21 / chemist-app learned panels will report missing checkpoints instead of substituting untrained models.

## Not required for single-SMILES interactive use

- Full OGB `dataset/` caches
- MoleculeNet CSVs under `data/`
- No-GraphCL encoder weights (not distributed; Protocol A ablation tables are already in `outputs/core_pipeline/ablation/`)
