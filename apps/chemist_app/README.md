# Chemist application (`apps/chemist_app`)

Local Streamlit UI for molecular analysis using the research GraphCL–GIN stack.

## Requirements

- Same environment as the core repo (`requirements.txt`), including `streamlit`
- Repository-relative weight files (not stored in Git; see root README §2):
  - `outputs/core_pipeline/graphcl_checkpoint.pt`
  - `outputs/esol/phase4e3_frozen_gin_regression/regression_head.pt`
  - `outputs/tox21/phase4f4_baselines/classification_head.pt`
- Committed support manifest:
  - `apps/mol_cli/support_manifests/bace_train_k5_v1.json`

Missing checkpoints are reported in the UI; **no untrained substitute** is used.

Paths are resolved from the repository root via `Path(__file__).resolve()`, not machine-specific absolute paths.

## Run (Windows / any OS)

From the **repository root**, with your virtual environment activated:

```text
python -m streamlit run apps/chemist_app/app.py
```

## Design

| Panel | Implementation |
|---|---|
| Descriptors | `apps.mol_cli.descriptors.compute_descriptors` |
| Structure | RDKit draw with atom indices |
| BACE | `BacePrototypeModel` (Protocol B) |
| ESOL | `EsolSolubilityModel` |
| Tox21 | `Tox21MultiTaskModel` |
| Explain | `PrototypeDistanceLogits` + PyG `GNNExplainer` (exploratory) |

Service layer: `service.py`. Do not invent explanations on failure.

## Limitations (shown in the UI)

- BACE Euclidean distances are **not** probabilities.
- Tox21 sigmoid scores are **not** established calibrated probabilities.
- GNNExplainer results are **exploratory** and do not establish chemical causality.
- Not clinical advice.
