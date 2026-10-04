# Molecular CLI (RDKit descriptors + optional OGB graph conversion)

Interactive terminal tool for SMILES validation, calculated molecular descriptors, and optional conversion of SMILES into **official OGB molecular graphs**.

This package is **separate** from the GraphCL / ProtoNet research pipeline. By default it does **not** load checkpoints or produce learned BACE, solubility, toxicity, or ADMET predictions. An **explicit** `--bace` flag enables experimental BACE few-shot prototype scoring only.

## Launch

From the repository root, using a Python interpreter that already has RDKit (and, for graph conversion / BACE mode, torch / torch-geometric / ogb):

```text
python -m apps.mol_cli.predict
python -m apps.mol_cli.predict --bace
```

Type a SMILES string at the `SMILES>` prompt. Type `quit`, `q`, or `exit` to leave.

Without `--bace`, startup and reporting match the descriptor/graph-only CLI: BACE remains unavailable and the GraphCL checkpoint is not loaded.

With `--bace`, the CLI loads the GraphCL checkpoint once at startup (weights-only), builds prototypes from the fixed train-only support manifest, and scores each valid SMILES with nearest Euclidean prototype. Initialization failure exits with a non-zero status before the REPL starts.

Interactive BACE validation uses `dataset_inventory="auto"`: if the local `dataset/ogbg_molbace` cache is present, the manifest is checked against the official OGB inventory; if the cache is absent, a self-contained **inference-only** check validates schema, K=5-per-class balance, labels, and SMILES/canonical consistency from the committed manifest. Inference-only mode does **not** re-verify training-split membership. Offline evaluation (`python -m apps.mol_cli.evaluate_bace_support`) still **requires** the local inventory and never downloads it.

## What is shown

- Original and canonical SMILES
- Atom and bond counts
- RDKit descriptors: molecular weight, cLogP, TPSA, H-bond donors/acceptors, rotatable bonds, ring count
- Optional OGB graph metadata (nodes, directed edges, feature dims/dtypes) when torch, PyG, and ogb are available

Descriptors are **calculated chemical features**, not machine-learning outputs.

Successful OGB graph conversion means the SMILES was mapped with `ogb.utils.mol.smiles2graph` into the same feature layout used by ogbg-mol\* datasets (`x`: 9 int64 atom features; bidirectional `edge_index`; `edge_attr`: 3 int64 bond features). **It does not validate GIN or ProtoNet predictions.**

Solubility, toxicity, and ADMET remain **not available**. Atom/bond explanations remain disabled.

### Experimental BACE (`--bace`)

When enabled, the report adds:

- Predicted class: inactive (0) or active (1)
- Euclidean distances to both class prototypes (and gap)
- `calibrated_probability: false` — distances are **not** probabilities or confidence percentages
- Checkpoint SHA-256 prefix and support manifest id (`bace_train_k5_v1`, train-only, K=5 per class, seed 42)
- Pinned summary of evaluation report `bace_fixed_support_eval_v1` (OGB BACE scaffold **fixed-support** protocol): test balanced accuracy approx. 0.66; sensitivity approx. 0.43 (many true actives were classified inactive)
- Explicit note that this is **not** the research paper's episodic test-pool ProtoNet protocol
- Domain / non-clinical caveats

These pinned metrics refer only to that evaluation-report version and protocol; they are not live or universally representative.

## Graph conversion API

```text
from apps.mol_cli.ogb_graph import smiles_to_ogb_dict, smiles_to_pyg_data
```

Official functions used: `ogb.utils.mol.smiles2graph` (and feature helpers it calls), with PyG `Data` fields aligned to `ogb.io.read_graph_pyg`.

## Tests

From the repository root:

```text
python -m unittest discover -s apps/mol_cli/tests -v
```

Graph tests require torch, torch_geometric, and ogb. Descriptor tests require only RDKit.

Checkpoint smoke tests (`test_checkpoint_smoke.py`) load `outputs/core_pipeline/graphcl_checkpoint.pt` with `torch.load(..., weights_only=True)`, then build the research `GINEncoder` from that dict. A successful embedding check is **software compatibility only** — not predictive accuracy.

## BACE support manifest (train only)

```text
apps/mol_cli/support_manifests/bace_train_k5_v1.json
```

Built with:

```text
python -m apps.mol_cli.build_bace_support_manifest
```

Selection uses the official OGB scaffold **training** split only (seed 42, K=5 per class). Validation/test molecules are never selected. The builder refuses to overwrite an existing manifest.

## Inference adapter

`apps/mol_cli/bace_infer.py` loads the GraphCL checkpoint (weights-only), embeds the fixed train support, builds class prototypes, and scores one query by Euclidean distance. The user-facing CLI calls it **only** when `--bace` is passed. Distance-derived scores are **not** calibrated probabilities. See `validate_manifest_for_inference` vs `validate_manifest` in `bace_support.py` for the interactive vs inventory validation split.

## Fixed-support evaluation (offline)

```text
python -m apps.mol_cli.evaluate_bace_support
```

Writes a versioned report under `apps/mol_cli/evaluation_reports/` (no overwrite). Uses train-only support prototypes; scores official validation and test molecules separately. This is **not** the research paper's test-episode protocol. The CLI disclaimer pins summary numbers to `bace_fixed_support_eval_v1`.
