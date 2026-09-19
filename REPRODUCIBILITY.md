# Reproducibility notes

Scientific outputs for the paper live under `outputs/core_pipeline/`. This file explains **which accuracy belongs to which episode list**. GraphCL was **trained once**.

## Environment

Recorded from the interpreter that ran the pipeline (`C:\Users\Conda\python.exe` on this machine):

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

The repository `.venv` is a different Python (3.14) and is **not** the run environment. Pins: `requirements.txt`, `environment.txt`.

## GraphCL was trained once

- One pretraining run: 10 NT-Xent epochs on ogbg-molpcba, seed **42**.
- Checkpoint: `outputs/core_pipeline/graphcl_checkpoint.pt`.
- Ablations and XAI **load this file**. They do not retrain GraphCL.
- There is no second GraphCL seed in the saved results.

## Seed 42 core run (main paper number)

After GraphCL, `main.py` reloads the checkpoint and samples **100** 2-way 5-shot 5-query episodes with seed 42 **in a process whose RNG was already used for GraphCL**.

| Quantity | Value |
|---|---|
| Mean accuracy | 0.5910000041872263 (**0.5910**) |
| Std (`np.std`, population) | 0.16976159420898532 (**0.1698**) |
| Episodes | 100 |
| Files | `outputs/core_pipeline/fewshot_results.csv`, `fewshot_summary.json` |

This is the **canonical** few-shot result. The No-GraphCL ablation in `ablation/no_graphcl_summary.json` (**0.6590**) **replays these exact support/query index sets**.

## 5-seed episode robustness

`ablation_multi_seed.py` keeps GraphCL weights fixed and resamples 100 episodes per seed in `{42, 43, 44, 45, 46}`. Pairing check: resampling seed 42 twice gave identical index lists; seed 42 vs 43 differed in 100/100 episodes. Both encoders share the same list within a seed.

| Condition | Mean of seed-means | Sample std of seed-means (ddof=1) |
|---|---:|---:|
| GraphCL | 0.5884 | 0.007797 |
| No-GraphCL | 0.6540 | 0.014900 |

Per-seed GraphCL means: 42 **0.597**, 43 0.591, 44 0.576, 45 0.591, 46 0.587. No-GraphCL is higher on **5/5** seeds.

This is robustness over **episode** seeds, not over GraphCL pretraining seeds.

## Why 0.591, 0.597, and 0.596 are not the same result

All three use the same checkpoint and the same 2-way 5-shot BACE-test protocol. They are **not** three measurements of one episode list.

| Figure | What it is | Episode source |
|---|---|---|
| **0.5910** | Mean of 100 **episode accuracies** on the **core** run | Sampled in `main.py` **after** GraphCL training RNG consumption |
| **0.5970** | GraphCL mean on multi-seed **seed 42** | Fresh `set_seed(42)` then 100 episodes (**no** GraphCL training in that process) |
| **0.596** | **Query-level** accuracy (596/1000) in the distance-gap analysis | `confidence_analysis.py`: `set_seed(42)` then 100 new episodes |

`MULTI_SEED_SUMMARY.md` already notes that the original core episodes are a different draw than multi-seed seed 42.

**Do not** average, swap, or “round to the same 0.59” as if they were replicates of one experiment. In the paper:

- Lead few-shot number: **0.5910 ± 0.1698** (core `fewshot_summary.json`).
- Ablation on the **same molecules**: No-GraphCL **0.6590** (`no_graphcl_summary.json`).
- Episode-seed robustness: **0.5884 vs 0.6540** across seeds 42–46.
- Distance-gap section: **0.596** on 1000 queries, with the resampling caveat.

## Other seeds used on purpose

| Seed | Use |
|---|---|
| 7 | Prototype XAI and all GNNExplainer molecule selections (one episode) |
| 42 | GraphCL, core few-shot, No-GraphCL replay, multi-seed, confidence analysis |

## Historical diagnostics (not this pipeline)

See `outputs/HISTORICAL_DIAGNOSTIC_ONLY.txt`. Do not cite 0.5180 ± 0.1590 or the 9/10 explainer episode.
