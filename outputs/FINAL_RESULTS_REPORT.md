# Final Results Report

**Project:** Graph Neural Networks with Few-Shot Learning: An Explainable Framework for Molecular Property Prediction
**Role of this document:** Consolidate metrics and claim boundaries from saved repository artifacts. No new experiments were run to produce this report.

Companion files: [README.md](../README.md), [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

---

## 1. Project overview and scope

This repository records a **shared GraphCL-pretrained GIN encoder** and several **downstream evaluation tracks**:

| Track | Task | Status in artifacts |
|---|---|---|
| BACE episodic ProtoNet | Few-shot binary classification (OGB scaffold test pool) | Completed (canonical paper number) |
| BACE fixed-support CLI | Fixed K=5 support from train; valid/test scored separately | Completed (separate protocol) |
| ESOL | Aqueous solubility regression (frozen split) | Validation + locked test |
| Tox21 | 12-task toxicity classification (frozen split) | Validation + locked test |
| Explainability | BACE ProtoNet XAI; Tox21 GNNExplainer smoke | Qualitative / smoke only |

**Out of scope for claims:** The complete multi-property “explainable framework” has **not** been validated end-to-end as a single product. Tracks share a checkpoint and coding conventions; they are not one joint evaluation.

**Shared encoder limitation:** `GINEncoder.forward` uses node features `x` and `edge_index` only; OGB **`edge_attr` bond features are unused**.

**Checkpoint provenance (all GraphCL-downstream tracks):**

- Path: `outputs/core_pipeline/graphcl_checkpoint.pt`
- SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Pretraining: one MolPCBA GraphCL NT-Xent run (10 epochs, seed 42). Ablations and heads **load** this file; they do not retrain GraphCL.

---

## 2. Methods and dataset / split descriptions

### Shared encoder

- Architecture: GIN, `in_dim=9`, hidden 128, 5 layers, dropout 0.1, sum pool (~184k parameters).
- Pretraining data: ogbg-molpcba graphs (labels unused for contrastive GraphCL).
- At few-shot / ESOL / Tox21 eval: projection head unused; encoder frozen unless a track trains a new head on top.

### BACE (few-shot)

- Dataset: ogbg-molbace (OGB scaffold splits).
- **Protocol A (episodic):** 2-way 5-shot 5-query Euclidean Prototypical Networks; pool = scaffold **test** only (152 molecules). Core run: 100 episodes after GraphCL training RNG in `main.py`, seed 42.
- **Protocol B (CLI fixed-support):** Stratified K=5 per class from scaffold **train only** (`bace_train_k5_v1`); validation and test scored separately. Not interchangeable with Protocol A.

### ESOL (regression)

- Source: `data/moleculenet/esol/delaney-processed.csv` (SHA-256 `8c06a76f…`).
- Target: **`measured log solubility in mols per litre`** only (never the predicted column).
- Frozen split: train 903 / valid 113 / test 112; mapping fingerprint `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac`.
- Models: train-mean constant; RDKit descriptors + Ridge (α=1); frozen GIN + trained regression head (validation-selected, then locked test).
- License/redistribution status from download headers remains **unresolved** (`data/moleculenet/esol/PROVENANCE.md`).

### Tox21 (multi-task classification)

- Source: `data/moleculenet/tox21/tox21.csv` (SHA-256 `7d7e7fac…`).
- Labels: 0 / 1 / **missing** — missing labels stay **masked** (never treated as negatives).
- Frozen split: train 6264 / valid 783 / test 784; mapping fingerprint `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd`.
- Graph-ineligible molecules (invalid SMILES for graph featurization) remain in the census but are **excluded from graph-model scoring**.
- Models: majority-class baseline (hard majority → ROC-AUC undefined); frozen GIN + multi-task head.
- License/redistribution status remains **unresolved** (`data/moleculenet/tox21/PROVENANCE.md`).

---

## 3. Results table (validation vs test)

Rounded display matches README / locked-test briefs. Exact floats live in the cited JSON files.

### 3.1 BACE episodic ProtoNet (Protocol A) — test-pool episodes

| Condition | Mean accuracy | Std | Episodes | Source |
|---|---:|---:|---:|---|
| GraphCL (core) | **0.5910** | **0.1698** | 100 | `outputs/core_pipeline/fewshot_summary.json` |
| No-GraphCL (replayed core episodes) | **0.6590** | 0.1727 | 100 | `outputs/core_pipeline/ablation/` |

On this recorded protocol, **No-GraphCL outperformed GraphCL**. Multi-seed episode means (seeds 42–46): GraphCL **0.5884** vs No-GraphCL **0.6540** (No-GraphCL higher on 5/5 seeds). This is episode-seed robustness, not GraphCL pretraining re-seeds.

Do **not** conflate 0.5910 (core), 0.5970 (fresh seed-42 episode list), and 0.596 (query-level distance-gap analysis).

### 3.2 BACE fixed-support CLI (Protocol B) — separate

| Split | Accuracy | Balanced accuracy | AUROC | Source |
|---|---:|---:|---:|---|
| Validation (n=151) | 0.4503 | 0.6408 | 0.6385 | `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` |
| Test (n=152) | 0.6447 | 0.6597 | 0.7279 | same |

**Not** the episodic paper number (0.5910).

### 3.3 ESOL — validation and locked test (n_test = 112; no exclusions)

| Model | Valid MAE | Valid RMSE | Valid R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Train-mean | 1.8864 | 2.3203 | −0.0013 | 1.8381 | 2.2291 | −0.115 |
| RDKit + Ridge (α=1) | 0.9652 | 1.2866 | 0.6921 | 0.7867 | 1.0239 | 0.765 |
| Frozen GIN + head | 0.8777 | 1.2239 | 0.7214 | 0.7245 | 1.0635 | 0.746 |

Sources: `outputs/esol/phase4e{1,2,3}_*/results.json`, `outputs/esol/phase4e_test_locked/results.json`.
R² values are supplementary (`r2_supplementary` in JSON). Test evaluation used validation-selected configs only (no test tuning).

### 3.4 Tox21 — validation and locked test

| Aggregate | Validation | Test |
|---|---:|---:|
| Majority-class **accuracy** | 0.9361 | **0.9367** |
| Majority ROC-AUC | undefined (null) | undefined (null) |
| Frozen GIN macro ROC-AUC (12/12 defined tasks) | **0.7687** | **0.7283** |
| Frozen GIN macro AP (12/12) | **0.2984** | **0.2844** |

Test census **784**; graph model scored **781**; **3** graph-ineligible excluded from graph scoring. Missing labels remain masked.

**Comparability warning:** Majority **accuracy is not directly comparable to ROC-AUC**. Class imbalance is severe; a hard majority predictor can achieve high accuracy while ROC-AUC remains undefined (constant scores).

Sources: `outputs/tox21/phase4f4_baselines/results.json`, `outputs/tox21/phase4f_test_locked/results.json`.

---

## 4. Two BACE evaluation protocols (do not mix)

### Protocol A — Episodic ProtoNet (canonical research number)

- Support and query sampled **per episode** from the OGB BACE scaffold **test** pool.
- Lead result: GraphCL **0.5910 ± 0.1698** (100 episodes).
- Ablation on the **same** support/query index sets: No-GraphCL **0.6590** (outperformed GraphCL on this protocol).
- Artifacts: `fewshot_results.csv`, `fewshot_summary.json`, `ablation/`.

### Protocol B — Fixed-support CLI

- Support: fixed stratified K=5 per class from scaffold **train only** (`apps/mol_cli/support_manifests/bace_train_k5_v1.json`, seed 42).
- Validation and test scored once each; neither enters support.
- Report: `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` (test balanced accuracy ≈ 0.66).
- CLI entry: `python -m apps.mol_cli.predict --bace`.

**Rule:** Never present Protocol B metrics as replicates or alternatives of the 0.5910 episodic mean without an explicit protocol label.

---

## 5. Explainability results and limitations

### BACE (ProtoNet track)

Documented in `outputs/core_pipeline/xai/XAI_SUMMARY.md`:

1. **Prototype distances** (primary): episode seed 7; query accuracy 0.300 on 10 queries. Gap `|d0−d1|` is **not** a calibrated probability.
2. **Distance-gap analysis:** association only (AUROC ≈ 0.589 on a resampled episode list with query-level acc 0.596); no ECE on the raw gap.
3. **GNNExplainer** on `PrototypeDistanceLogits`: treat as a **qualitative limitation**, not a positive chemical-XAI result. Soft GraphFramEx fidelity remained uninformative (fid+/fid− = 0 in recorded runs); temperature scaling changes mask sharpness but does not establish chemical causality.

### Tox21 (Phase 4F.6)

- **Two-molecule smoke test** on validation graph-eligible molecules for task **NR-AhR** (`outputs/tox21/phase4f6_explainability/`).
- Adapter logits matched the Phase 4F.4 head before explaining.
- **Not** a comprehensive explainability study; test split was not used for XAI.
- Masks do **not** establish chemical causality.
- Because the encoder ignores `edge_attr`, edge masks are connectivity-level, not bond-feature attributions.

---

## 6. Reproducibility instructions and artifact locations

Prefer verifying frozen fingerprints and loading saved checkpoints. Prefer **not** overwriting `outputs/core_pipeline/` if the completed BACE paper run must be preserved.

### Frozen-split verification

```text
python data/moleculenet/esol/verify_esol_split_frozen_v1.py
python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py
```

### Key artifact paths

| Content | Location |
|---|---|
| GraphCL checkpoint | `outputs/core_pipeline/graphcl_checkpoint.pt` |
| BACE episodic results | `outputs/core_pipeline/fewshot_summary.json`, `fewshot_results.csv` |
| BACE No-GraphCL ablation | `outputs/core_pipeline/ablation/` |
| BACE XAI | `outputs/core_pipeline/xai/` |
| BACE CLI fixed-support report | `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` |
| ESOL validation | `outputs/esol/phase4e{1,2,3}_*/` |
| ESOL locked test | `outputs/esol/phase4e_test_locked/` |
| Tox21 validation baselines | `outputs/tox21/phase4f4_baselines/` |
| Tox21 locked test | `outputs/tox21/phase4f_test_locked/` |
| Tox21 XAI smoke | `outputs/tox21/phase4f6_explainability/` |
| Protocol map | `REPRODUCIBILITY.md` |

### Example recompute commands (validation helpers; not claimed as re-run for this report)

```text
python outputs/esol/phase4e1_mean_baseline/recompute_mean_baseline.py
python outputs/esol/phase4e2_descriptor_baseline/recompute_descriptor_baseline.py
python outputs/esol/phase4e3_frozen_gin_regression/evaluate_frozen_gin_regression.py
python outputs/tox21/phase4f4_baselines/evaluate_tox21_baselines.py
python outputs/tox21/phase4f6_explainability/run_tox21_gnnexplainer_smoke.py
```

Environment for the recorded stack: Python 3.13.9, PyTorch 2.9.1+cpu, PyG 2.7.0, ogb 1.3.6, RDKit 2025.9.3, scikit-learn 1.8.0 (CPU). The repo `.venv` (Python 3.14) is a different environment. See `REPRODUCIBILITY.md`.

---

## 7. Limitations and cautious conclusions

1. **No end-to-end framework claim.** Individual tracks have saved results; the full multi-property explainable product was not jointly validated.
2. **GraphCL did not improve BACE few-shot accuracy** on the recorded episodic protocol relative to No-GraphCL (0.5910 vs 0.6590; same episode lists).
3. **ESOL:** Frozen GIN + head improved validation MAE/RMSE/R² over the mean baseline and edged Ridge on validation MAE/RMSE; on locked test, Ridge has higher R² (0.765) while GIN has lower MAE (0.7245 vs 0.7867) and slightly higher RMSE (1.0635 vs 1.0239). Report metrics separately; do not collapse to a single “winner” without stating the metric.
4. **Tox21:** Frozen GIN achieves macro ROC-AUC 0.7687 (valid) / 0.7283 (test) with low macro AP (0.2984 / 0.2844), consistent with severe imbalance. Majority accuracy (~0.936) must not be compared to ROC-AUC.
5. **Missing labels** on Tox21 remain masked; graph-ineligible rows are retained in the split census but excluded from graph scoring (test: 3 excluded → 781 scored).
6. **Explainability** is qualitative (BACE) or a two-molecule smoke (Tox21). Masks do not establish chemical causality; unused `edge_attr` limits edge-mask interpretation.
7. **Dataset redistribution licensing** for DeepChem-hosted ESOL/Tox21 CSVs remains unresolved where documented in `PROVENANCE.md`.
8. **Historical diagnostics** (`outputs/final_results.txt` 0.5180 ± 0.1590; legacy 9/10 explainer) are **not** paper numbers for this pipeline.

**Bottom line:** The repository supports cautious reporting of (i) BACE episodic few-shot with an honest No-GraphCL comparison, (ii) ESOL and Tox21 frozen-split validation and locked-test metrics for the listed models, and (iii) limited explainability artifacts—without claiming a fully validated multi-property explainable framework.
