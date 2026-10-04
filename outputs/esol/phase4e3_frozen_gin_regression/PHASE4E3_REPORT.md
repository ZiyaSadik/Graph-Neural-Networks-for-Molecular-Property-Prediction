# Phase 4E.3 — Frozen GIN + ESOL Regression Head

## Summary

| Item | Value |
|---|---|
| Encoder | Frozen GraphCL GIN (`graphcl_checkpoint.pt`) |
| Embedding | 128-d graph embedding (not projection) |
| Head | Linear(128→64)→ReLU→Linear(64→1), **trained only** |
| Best epoch | 103 |
| Validation MAE | 0.877680 |
| Validation RMSE | 1.223935 |
| Validation R² (supplementary) | 0.721381 |
| Test metrics | Not computed |

## Training configuration (predeclared)

- Seed: 42
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Batch size: 64
- Max epochs: 200
- Loss: MSE on **train-standardized** targets (mean/std fit on train only)
- Early stopping: minimize **validation MAE on original measured scale**, patience=25
- Device: CPU
- Encoder: `eval()` + `requires_grad=False`; embeddings under `torch.no_grad()`
- Embedding recompute max abs diff: 0.000e+00

## Comparison (same frozen validation rows)

| Model | MAE | RMSE | R² (suppl.) |
|---|---:|---:|---:|
| 4E.1 train-mean | 1.8864 | 2.3203 | -0.0013 |
| 4E.2 RDKit descriptors + Ridge | 0.9652 | 1.2866 | 0.6921 |
| **4E.3 frozen GIN + head** | **0.8777** | **1.2239** | **0.7214** |

Better than mean baseline (MAE): True
Better than descriptor baseline (MAE): True

## Limitations

- `GINEncoder.forward` uses node features `x` and `edge_index` only; OGB `edge_attr` bond features are converted but **unused**.
- Encoder is **not** fine-tuned; only the new head was trained.
- Results apply to this ESOL frozen-split validation protocol only.

## Reproducibility

```text
python outputs/esol/phase4e3_frozen_gin_regression/evaluate_frozen_gin_regression.py
```

## Provenance

- Source SHA-256: `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
- Mapping fingerprint: `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac`
- Pretrained checkpoint SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Graph API: `apps.mol_cli.ogb_graph.smiles_to_pyg_data`
- Load API: `load_encoder_from_research_checkpoint` (weights_only=True)

## Scope

No modifications to `main.py`, `src/`, CLI, BACE artifacts, frozen ESOL split files, or the pretrained checkpoint file bytes.
