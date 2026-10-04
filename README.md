# Graph Neural Networks with Few-Shot Learning: An Explainable Framework for Molecular Property Prediction

This repository records a **completed multi-track research codebase** around a shared GraphCL-pretrained GIN encoder (`outputs/core_pipeline/graphcl_checkpoint.pt`):

1. **BACE** few-shot Prototypical Network evaluation (canonical paper track)
2. **ESOL** aqueous solubility regression (frozen-split baselines + locked test)
3. **Tox21** multi-task toxicity classification (frozen-split baselines + locked test)
4. **Explainability** experiments (BACE ProtoNet XAI; Tox21 GNNExplainer smoke)

Individual pipelines have saved artifacts. That does **not** mean the full multi-property “explainable framework” has been validated end-to-end as a single product. See [outputs/FINAL_RESULTS_REPORT.md](outputs/FINAL_RESULTS_REPORT.md) for the consolidated table and cautious conclusions. Historical MSE-to-noise numbers are **diagnostic only** — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Shared encoder

```
ogbg-molpcba graphs (labels unused)
        -> GraphCL NT-Xent on GINEncoder (5 layers, 128-d, sum pool)
        -> outputs/core_pipeline/graphcl_checkpoint.pt
```

- **Pretraining:** MolPCBA, 437929 graphs, 10 epochs, batch 128, NT-Xent temperature 0.2, Adam lr 0.001, seed 42.
- **Architecture:** `in_dim=9`, hidden 128, 5 GIN layers, dropout 0.1, sum pool; **184,197** parameters.
- **Checkpoint SHA-256:** `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- **Known limitation:** `GINEncoder.forward` uses node features `x` and `edge_index` only; OGB **`edge_attr` bond features are unused**.

Downstream ESOL/Tox21 heads freeze this encoder and train only new task heads (validation-selected), then apply a **one-shot locked test** evaluation without re-tuning.

---

## 1. BACE — episodic ProtoNet (canonical)

```
checkpoint -> reload GINEncoder (projection head not used at eval)
          -> 2-way 5-shot 5-query Euclidean ProtoNet
          -> ogbg-molbace OGB scaffold TEST only
```

- BACE train (1210) and valid (151) reserved; few-shot pool = **152 test** molecules.
- Predictor: `pred = argmin_c ||z - p_c||_2` (temperature 1). Not a supervised classifier head.

**Main result (core run):** GraphCL few-shot accuracy **0.5910 ± 0.1698** (100 episodes, seed 42).
Exact: mean `0.5910000041872263`, population std `0.16976159420898532`.
Sources: `outputs/core_pipeline/fewshot_summary.json`, `fewshot_results.csv`, `RESULTS_SUMMARY.md`.

### No-GraphCL ablation

Same architecture, **zero** encoder training steps, frozen at eval.

| Comparison | GraphCL | No-GraphCL | Note |
|---|---:|---:|---|
| Replayed core episodes (seed 42 list) | 0.5910 | **0.6590** | No-GraphCL **outperformed** GraphCL on this recorded protocol |
| Mean of 5 episode seeds (42–46) | 0.5884 | **0.6540** | GraphCL weights not retrained; No-GraphCL higher 5/5 |

Do **not** claim GraphCL improves few-shot BACE accuracy on this protocol. Do not treat **0.591**, **0.597**, and **0.596** as the same result ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

### Separate protocol: BACE fixed-support CLI

`apps/mol_cli` with `--bace` uses a **different** protocol: fixed K=5-per-class support from OGB scaffold **train only** (`bace_train_k5_v1`), then scores validation and test separately. Report: `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` (test balanced accuracy approx. 0.66).

**Not interchangeable** with the episodic test-pool ProtoNet result (0.5910).

---

## 2. ESOL — solubility regression

- Data: MoleculeNet/DeepChem `delaney-processed.csv` under `data/moleculenet/esol/` (provenance + integrity in that folder).
- Target: **`measured log solubility in mols per litre`** only — never the ESOL predicted column.
- Split: frozen scaffold/leakage-aware assignments (`esol_split_*_v1`); mapping fingerprint `90f827da…`. Seed-only regeneration is **not** reliable.
- Verifier: `python data/moleculenet/esol/verify_esol_split_frozen_v1.py`

| Model | Valid MAE | Valid RMSE | Valid R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Train-mean | 1.8864 | 2.3203 | −0.0013 | 1.8381 | 2.2291 | −0.115 |
| RDKit descriptors + Ridge (α=1) | 0.9652 | 1.2866 | 0.6921 | 0.7867 | 1.0239 | 0.765 |
| Frozen GIN + regression head | 0.8777 | 1.2239 | 0.7214 | 0.7245 | 1.0635 | 0.746 |

Test: **n=112**, no exclusions. Locked test artifacts: `outputs/esol/phase4e_test_locked/`. Validation phases: `outputs/esol/phase4e{1,2,3}_*/`.

License/redistribution status of the DeepChem-hosted CSV remains **unresolved** from download headers (`data/moleculenet/esol/PROVENANCE.md`).

---

## 3. Tox21 — 12-task classification

- Data: MoleculeNet/DeepChem `tox21.csv` under `data/moleculenet/tox21/`.
- Labels: 0 / 1 / **missing**; missing labels stay **masked** (never treated as negatives).
- Split: frozen scaffold/leakage-aware assignments; mapping fingerprint `bffa1dbb…`.
- Verifier: `python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py`
- Graph-ineligible rows (invalid SMILES) remain in the census but are **excluded from graph-model scoring**.

| Aggregate | Validation | Test |
|---|---:|---:|
| Majority-class **accuracy** (imbalanced; not comparable to ROC-AUC) | 0.9361 | **0.9367** |
| Majority ROC-AUC | undefined (null) | undefined (null) |
| Frozen GIN macro ROC-AUC (12/12 defined tasks) | **0.7687** | **0.7283** |
| Frozen GIN macro AP (12/12) | **0.2984** | **0.2844** |

Test census **784**; graph model scored **781**; **3** graph-ineligible excluded. Macro = unweighted mean over tasks with both classes among labeled eligible examples.

Artifacts: validation `outputs/tox21/phase4f4_baselines/`; locked test `outputs/tox21/phase4f_test_locked/`. License status unresolved (`data/moleculenet/tox21/PROVENANCE.md`).

---

## 4. Explainability

### BACE (ProtoNet)

1. **Prototype distances** (primary): `|d0−d1|` is **not** a calibrated probability. Example episode seed 7: query accuracy 0.300 (10 queries).
2. **Distance-gap analysis:** association only (AUROC ~0.589); no ECE.
3. **GNNExplainer** on `PrototypeDistanceLogits`: treat as a **qualitative limitation**, not a positive chemical-XAI result (`outputs/core_pipeline/xai/XAI_SUMMARY.md`).

### Tox21

Phase 4F.6: **two-molecule** GNNExplainer smoke on validation graph-eligible molecules for task **NR-AhR**, after verifying adapter logits match the Phase 4F.4 head. **Not** a comprehensive explainability study. Masks do **not** establish chemical causality; edge masks are connectivity-level because the encoder ignores `edge_attr`.
Artifacts: `outputs/tox21/phase4f6_explainability/`.

---

## Do not report as paper numbers

| Item | Location | Number |
|---|---|---|
| MSE-to-noise BACE “pretrain” | `outputs/final_results.txt` | 0.5180 ± 0.1590 |
| Legacy 9/10 explainer episode | `archive/` / `outputs/final_model.pt` | 9/10 |
| Pipeline smoke few-shot | `outputs/core_pipeline/smoke/` | 0.70 on 1 episode |

---

## Environment (recorded run stack)

Results were produced with **CPU** Python, not the repo `.venv` (3.14, no `ogb`).

- Python 3.13.9 · PyTorch **2.9.1+cpu** · PyTorch Geometric **2.7.0** · ogb 1.3.6 · RDKit 2025.09.3 · scikit-learn 1.8.0

Pins: `requirements.txt`, `environment.txt`. Details: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Windows setup for teammates (interactive CLIs)

A fresh GitHub clone is **not fully runnable** for learned BACE / ESOL / Tox21 / visualization until the three model checkpoints below are obtained separately. That limit is operational (weights are excluded from the source-code repo by `.gitignore` `*.pt`), not a change to the recorded scientific results. Locked research metrics in the docs remain valid without re-running anything.

Interactive BACE does **not** require a local OGB BACE dataset cache: when the cache is absent, startup uses inference-only validation of the committed support manifest (`dataset_inventory="auto"`). Full offline BACE evaluation and support-manifest construction still need the dataset inventory (see §4 below).

### 1. Python environment (Windows)

From the repository root in PowerShell or the VS Code terminal:

1. Install **Python 3.13.x** (recorded: **3.13.9**). Avoid relying on the repo `.venv` if it is Python 3.14 without `ogb`.
2. Create and activate a virtual environment, for example:
   ```text
   py -3.13 -m venv .venv313
   .\.venv313\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```
3. Install **PyTorch CPU** matching the recorded stack (see [pytorch.org](https://pytorch.org) if the `+cpu` wheel is not on the default index), then the remaining pins from `requirements.txt` / `environment.txt`:
   ```text
   pip install torch==2.9.1 torchvision==0.24.1
   pip install torch-geometric==2.7.0 ogb==1.3.6 rdkit==2025.9.3
   pip install numpy==2.3.5 pandas==2.3.3 scikit-learn==1.8.0
   pip install matplotlib==3.10.8 seaborn==0.13.2 pillow==12.0.0 tqdm==4.67.1
   ```
4. Confirm imports:
   ```text
   python -c "import torch, torch_geometric, ogb, rdkit, matplotlib; print(torch.__version__)"
   ```

### 2. Checkpoint distribution and placement (required for learned models)

These three weight files are **excluded from a normal Git clone** (`.gitignore` matches `*.pt`). They are **intended to be distributed separately**, with a **GitHub Release** as the planned download location, **subject to confirming redistribution rights and actually publishing the files**. No release URL is provided here because a public Release may not exist yet.

After download, place each file at the **exact repository-relative path** below (create parent directories if needed):

| Destination path (repo root) | Role |
|---|---|
| `outputs/core_pipeline/graphcl_checkpoint.pt` | Frozen GraphCL GIN encoder (all learned CLIs) |
| `outputs/esol/phase4e3_frozen_gin_regression/regression_head.pt` | ESOL regression head |
| `outputs/tox21/phase4f4_baselines/classification_head.pt` | Tox21 12-task head |

Expected encoder SHA-256 (when present): `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`.

**Checkpoint-dependent commands** (`predict --bace`, `esol_infer`, `tox21_infer`, `visualize_predictions`) require the relevant `.pt` file(s) to be present at those paths. Without them, those entry points fail at startup; descriptor-only `predict` (no `--bace`) does not need weights.

MoleculeNet ESOL/Tox21 CSVs under `data/` are **not** required for single-SMILES interactive inference. They are required only for frozen-split recompute / verifiers.

### 3. Interactive CLI commands

Run from the **repository root** after activating the environment (and after placing any required checkpoints):

```text
python -m apps.mol_cli.predict
python -m apps.mol_cli.predict --bace
python -m apps.mol_cli.esol_infer
python -m apps.mol_cli.tox21_infer
python -m apps.mol_cli.visualize_predictions
```

| Command | Needs GraphCL `.pt` | Needs ESOL head `.pt` | Needs Tox21 head `.pt` | Needs local `dataset/ogbg_molbace` cache |
|---|---|---|---|---|
| `python -m apps.mol_cli.predict` | No | No | No | No |
| `python -m apps.mol_cli.predict --bace` | Yes | No | No | **No** for interactive scoring (`auto`: inference-only if cache absent) |
| `python -m apps.mol_cli.esol_infer` | Yes | Yes | No | No |
| `python -m apps.mol_cli.tox21_infer` | Yes | No | Yes | No |
| `python -m apps.mol_cli.visualize_predictions` | Yes | Yes | Yes | **No** for interactive BACE panel (same `auto` behavior) |

Notes:

- `predict` without `--bace` is RDKit descriptors (+ optional OGB graph metadata). It does **not** load research weights.
- BACE distances are **not** probabilities. ESOL outputs are on the measured log-solubility scale. Tox21 scores are sigmoid(logits), **not** calibrated probabilities or clinical determinations.
- Visualization shows structure + graph + BACE/ESOL/Tox21 panels; it does not establish causal explanations.
- More BACE CLI detail: [apps/mol_cli/README.md](apps/mol_cli/README.md).

### 4. BACE interactive validation vs offline inventory

Interactive BACE (`--bace` and the visualizer BACE panel) uses `dataset_inventory="auto"`:

- If local `dataset/ogbg_molbace` raw/split/mapping files **are** present → full dataset-inventory validation of the committed support manifest.
- If that cache is **absent** → **inference-only** validation of `apps/mol_cli/support_manifests/bace_train_k5_v1.json` (schema, K=5 per class, labels, SMILES parsing, canonical-SMILES consistency). Inference-only mode does **not** independently re-verify that support indices belong to the official OGB training split.

**Still require** the appropriate local OGB BACE dataset inventory/cache:

- Offline evaluation: `python -m apps.mol_cli.evaluate_bace_support` (`dataset_inventory="required"`)
- Support-manifest construction: `python -m apps.mol_cli.build_bace_support_manifest`

Neither path downloads the dataset automatically.

### 5. What remains before “clone and run”

1. Confirm redistribution rights, then publish the three checkpoints (planned: GitHub Release) and place them at the paths in §2.
2. Optional later: LICENSE and dataset redistribution decisions (not changed here).

Until the checkpoints are published and placed locally, treat this section as setup guidance, **not** a guarantee that a bare clone will run learned models.

---

## Reproduce / verify (saved artifacts)

Prefer loading saved checkpoints and frozen splits. **Do not overwrite** `outputs/core_pipeline/` if you need the completed BACE paper run.

```text
# Frozen split integrity (read-only; needs data/moleculenet files)
python data/moleculenet/esol/verify_esol_split_frozen_v1.py
python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py

# BACE core pipeline (expensive if retraining GraphCL)
python main.py --smoke
python main.py --skip-graphcl --checkpoint outputs/core_pipeline/graphcl_checkpoint.pt

# Recompute validation metrics from saved configs (examples)
python outputs/esol/phase4e1_mean_baseline/recompute_mean_baseline.py
python outputs/esol/phase4e2_descriptor_baseline/recompute_descriptor_baseline.py
python outputs/esol/phase4e3_frozen_gin_regression/evaluate_frozen_gin_regression.py
python outputs/tox21/phase4f4_baselines/evaluate_tox21_baselines.py
python outputs/tox21/phase4f6_explainability/run_tox21_gnnexplainer_smoke.py
```

Interactive CLIs (separate from the research episodic protocol): see **Windows setup for teammates** above and [apps/mol_cli/README.md](apps/mol_cli/README.md).

Completed BACE follow-ons already recorded under `outputs/core_pipeline/`: `ablation_no_graphcl.py`, `ablation_multi_seed.py`, `prototype_explain.py`, `confidence_analysis.py`, GNNExplainer scripts.

---

## Consolidated report

**[outputs/FINAL_RESULTS_REPORT.md](outputs/FINAL_RESULTS_REPORT.md)** — methods, validation vs test tables, BACE protocol separation, XAI limits, reproducibility paths, and cautious conclusions.
