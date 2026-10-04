# Reproducibility notes

This file maps **which number belongs to which protocol and artifact**. GraphCL was **trained once**. Saved validation and locked-test ESOL/Tox21 results must not be mixed with BACE episodic figures without labeling the protocol.

Consolidated narrative: [outputs/FINAL_RESULTS_REPORT.md](outputs/FINAL_RESULTS_REPORT.md).

## Environment

Recorded from the interpreter that ran the core pipeline and later phase experiments (`C:\Users\Conda\python.exe` on this machine):

| Package | Version |
|---|---|
| Python | 3.13.9 |
| PyTorch | 2.9.1+cpu |
| torchvision | 0.24.1 |
| torch-geometric | 2.7.0 |
| ogb | 1.3.6 |
| rdkit | 2025.9.3 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| pillow | 12.0.0 |
| tqdm | 4.67.1 |
| CUDA | unused (CPU) |

The repository `.venv` is a different Python (3.14) and is **not** the recorded run environment. Pins: `requirements.txt`, `environment.txt`.

## GraphCL was trained once

- One pretraining run: 10 NT-Xent epochs on ogbg-molpcba, seed **42**.
- Checkpoint: `outputs/core_pipeline/graphcl_checkpoint.pt`.
- SHA-256: `d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f`
- Ablations, CLI BACE, ESOL/Tox21 heads, and XAI **load this file**. They do not retrain GraphCL.
- There is no second GraphCL seed in the saved results.
- Encoder limitation: bond `edge_attr` is present in OGB graphs but **unused** by `GINEncoder.forward`.

---

## BACE protocol A — episodic ProtoNet (canonical paper number)

After GraphCL, `main.py` reloads the checkpoint and samples **100** 2-way 5-shot 5-query episodes with seed 42 **in a process whose RNG was already used for GraphCL**. Pool = OGB BACE scaffold **test** only.

| Quantity | Value |
|---|---|
| Mean accuracy | 0.5910000041872263 (**0.5910**) |
| Std (`np.std`, population) | 0.16976159420898532 (**0.1698**) |
| Episodes | 100 |
| Files | `outputs/core_pipeline/fewshot_results.csv`, `fewshot_summary.json` |

### No-GraphCL ablation

`ablation/no_graphcl_summary.json` (**0.6590**) **replays these exact support/query index sets**. On this recorded protocol, **No-GraphCL outperformed GraphCL**. Multi-seed episode means (seeds 42–46): GraphCL **0.5884** vs No-GraphCL **0.6540** (No-GraphCL higher on 5/5 seeds). This is robustness over **episode** seeds, not GraphCL pretraining seeds.

### Why 0.591, 0.597, and 0.596 are not the same result

| Figure | What it is | Episode source |
|---|---|---|
| **0.5910** | Mean of 100 **episode accuracies** on the **core** run | Sampled in `main.py` **after** GraphCL training RNG consumption |
| **0.5970** | GraphCL mean on multi-seed **seed 42** | Fresh `set_seed(42)` then 100 episodes (**no** GraphCL training in that process) |
| **0.596** | **Query-level** accuracy (596/1000) in the distance-gap analysis | `confidence_analysis.py`: `set_seed(42)` then 100 new episodes |

**Do not** average or swap these as replicates of one experiment.

Lead few-shot number: **0.5910 ± 0.1698**. Ablation on the same molecules: No-GraphCL **0.6590**.

## BACE protocol B — fixed-support CLI (separate)

- Support: stratified K=5 per class from OGB scaffold **train only** (`apps/mol_cli/support_manifests/bace_train_k5_v1.json`, seed 42).
- Evaluation report: `apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json` (validation and test scored separately).
- CLI: `python -m apps.mol_cli.predict --bace`.
- Explicitly **not interchangeable** with episodic test-pool ProtoNet (0.5910).

---

## ESOL — frozen split and results

| Item | Value |
|---|---|
| Source | `data/moleculenet/esol/delaney-processed.csv` |
| Source SHA-256 | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |
| Mapping fingerprint | `90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac` |
| Counts | train 903 / valid 113 / test 112 |
| Target | `measured log solubility in mols per litre` (never the predicted column) |
| Verifier | `python data/moleculenet/esol/verify_esol_split_frozen_v1.py` |

**Authoritative assignments** are the saved CSV/manifest. Seed 42 alone does **not** regenerate the split (Phase 4D.1: 307/1128 diffs). License/redistribution status from S3 headers remains **unresolved** (`PROVENANCE.md`).

| Model | Valid MAE / RMSE / R² | Test MAE / RMSE / R² |
|---|---|---|
| Train-mean | 1.8864 / 2.3203 / −0.0013 | 1.8381 / 2.2291 / −0.115 |
| RDKit + Ridge (α=1) | 0.9652 / 1.2866 / 0.6921 | 0.7867 / 1.0239 / 0.765 |
| Frozen GIN + head | 0.8777 / 1.2239 / 0.7214 | 0.7245 / 1.0635 / 0.746 |

Artifacts: `outputs/esol/phase4e{1,2,3}_*/`, locked test `outputs/esol/phase4e_test_locked/`. Test evaluation used validation-selected configs only (no test tuning).

Recompute helpers (validation):
`recompute_mean_baseline.py`, `recompute_descriptor_baseline.py`, `evaluate_frozen_gin_regression.py` under the respective phase folders.

---

## Tox21 — frozen split and results

| Item | Value |
|---|---|
| Source | `data/moleculenet/tox21/tox21.csv` |
| Source SHA-256 | `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761` |
| Mapping fingerprint | `bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd` |
| Counts | train 6264 / valid 783 / test 784 |
| Labels | 0 / 1 / missing — **missing stay masked** |
| Verifier | `python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py` |
| Graph-ineligible policy | retain all rows; exclude from graph scoring |

| Aggregate | Validation | Test |
|---|---|---|
| Majority accuracy | **0.9361** (`macro_accuracy` in phase4f4 `results.json`) | **0.9367** |
| Majority ROC-AUC | **null** (constant / hard majority) | **null** |
| Frozen GIN macro ROC-AUC | **0.7687** (12/12) | **0.7283** (12/12) |
| Frozen GIN macro AP | **0.2984** (12/12) | **0.2844** (12/12) |

Test census **784**; scored **781** graph-eligible; **3** ineligible excluded from graph scoring. Macro = unweighted mean over tasks with defined metrics. Majority **accuracy is not comparable to ROC-AUC**; class imbalance is severe.

Artifacts: `outputs/tox21/phase4f4_baselines/`, `outputs/tox21/phase4f_test_locked/`. License unresolved (`PROVENANCE.md`). Saved assignments are authoritative (do not rely on seed-only regeneration).

Recompute: `outputs/tox21/phase4f4_baselines/evaluate_tox21_baselines.py`.

---

## Explainability reproducibility

| Track | What is saved | Claim boundary |
|---|---|---|
| BACE prototype / gap / GNNExplainer | `outputs/core_pipeline/xai/` | Qualitative / limitation-focused; not calibrated confidence |
| Tox21 GNNExplainer | `outputs/tox21/phase4f6_explainability/` | **Two-molecule smoke** on validation NR-AhR; not a full XAI study |

Masks do not establish chemical causality. Edge masks are not bond-feature attributions (`edge_attr` unused by GIN).

Seeds used on purpose for BACE XAI: **7** (prototype + GNNExplainer molecule episode); **42** (GraphCL, core few-shot, ablations, confidence analysis).

---

## Frozen-split and checkpoint checklist

```text
python data/moleculenet/esol/verify_esol_split_frozen_v1.py
python data/moleculenet/tox21/verify_tox21_split_frozen_v1.py
python inspect_checkpoint.py   # if present; prints encoder config from graphcl_checkpoint.pt
```

Before trusting locked-test directories, confirm fingerprints in those folders’ `results.json` / `preflight.json` match the freeze records above.

---

## Historical diagnostics (not this pipeline)

See `outputs/HISTORICAL_DIAGNOSTIC_ONLY.txt`. Do not cite 0.5180 ± 0.1590 or the 9/10 explainer episode.
