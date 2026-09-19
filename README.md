# GraphCL + few-shot ProtoNet for BACE (canonical pipeline)

Few-shot molecular property prediction with a frozen Graph Isomorphism Network (GIN) encoder. The encoder is pretrained with GraphCL (NT-Xent) on **ogbg-molpcba** (labels unused), then evaluated with Prototypical Networks on the official OGB **BACE scaffold test** split only.

This repository records one completed scientific pipeline. Historical MSE-to-noise numbers are **diagnostic only** and must not be cited as GraphCL results. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before mixing accuracy figures.

## Canonical pipeline

```
ogbg-molpcba graphs (labels unused)
        -> GraphCL NT-Xent on GINEncoder (5 layers, 128-d, sum pool)
        -> outputs/core_pipeline/graphcl_checkpoint.pt
        -> reload GINEncoder (projection head not used at eval)
        -> 2-way 5-shot 5-query Euclidean ProtoNet
        -> ogbg-molbace OGB scaffold TEST only
```

- **Pretraining:** MolPCBA, 437929 graphs, 10 epochs, batch 128, NT-Xent temperature 0.2, Adam lr 0.001, seed 42. BACE is never used for encoder training.
- **Evaluation:** BACE train (1210) and valid (151) reserved. Few-shot pool = 152 test molecules. Support and query index sets are disjoint within each episode.
- **Predictor:** class prototype = mean support embedding; `pred = argmin_c ||z - p_c||_2` (equivalently `argmax -d_c`). Temperature 1. Not a supervised classifier head.

Entry point: `main.py`. Do not run `experiment.py` (quarantined MSE-to-noise stub).

## Main result (core run)

**GraphCL few-shot accuracy: 0.5910 ± 0.1698** over 100 episodes (seed 42).

Exact floats: mean `0.5910000041872263`, population std `0.16976159420898532`.

Sources: `outputs/core_pipeline/fewshot_summary.json`, `fewshot_results.csv`, `RESULTS_SUMMARY.md`.

Checkpoint: `outputs/core_pipeline/graphcl_checkpoint.pt` (184,197 parameters). GraphCL epoch-10 loss: `1.4264949777891314` (`graphcl_training_history.csv`).

## No-GraphCL ablation

Same architecture, **zero** encoder training steps, frozen at eval.

| Comparison | GraphCL | No-GraphCL | Note |
|---|---:|---:|---|
| Replayed core episodes (seed 42 list) | 0.5910 | **0.6590** | `outputs/core_pipeline/ablation/` |
| Mean of 5 episode seeds (42–46) | 0.5884 | **0.6540** | GraphCL weights **not** retrained; No-GraphCL higher 5/5 |

Do not claim GraphCL improves few-shot BACE accuracy on this protocol.

Do not treat **0.591**, **0.597**, and **0.596** as the same result. They use different seed-42 episode draws. Details: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Explainability

1. **Prototype distances** (primary): per-query `d0`, `d1`, `|d0-d1|`, nearest support by cosine. Example episode seed 7: query accuracy 0.300 (10 queries). `|d0-d1|` is **not** a probability or calibrated confidence.
2. **Distance-gap analysis:** 1000 queries, overall acc 0.596; mean gap correct 13.85 vs incorrect 11.15; AUROC 0.589. Association only. No ECE.
3. **GNNExplainer:** PyG 2.7.0 `Explainer` + `GNNExplainer` on `PrototypeDistanceLogits`. Unscaled `-d` masks are diffuse (soft fid+/fid− = 0). `T=10` (`logit_c = -d_c / 10`) concentrates masks but GraphFramEx stays 0; top-k tests show a globally redundant decision. Treat GNNExplainer as a **qualitative limitation**, not a quantitative XAI result.

Status: `outputs/core_pipeline/xai/XAI_SUMMARY.md`.

## Do not report as paper numbers

| Item | Location | Number |
|---|---|---|
| MSE-to-noise BACE “pretrain” | `outputs/final_results.txt` | 0.5180 ± 0.1590 |
| Legacy 9/10 explainer episode | `archive/` / `outputs/final_model.pt` | 9/10 |
| Pipeline smoke few-shot | `outputs/core_pipeline/smoke/` | 0.70 on 1 episode |

Labels: `outputs/HISTORICAL_DIAGNOSTIC_ONLY.txt`, `archive/README.md`.

## Environment (actual run)

Results were produced with **CPU** Python, not the repo `.venv` (3.14, no `ogb`).

- Python 3.13.9
- PyTorch **2.9.1+cpu**
- PyTorch Geometric **2.7.0**
- ogb 1.3.6

Pinned list: `requirements.txt` and `environment.txt`.

## Reproduce the existing pipeline

Use the Conda/Python install that has the versions above. Commands from the repo root. **GraphCL pretraining is expensive (~4.4 h CPU).** Prefer loading the saved checkpoint unless you intend to retrain.

```text
python main.py --smoke
```

Tiny pipeline check only. Writes `outputs/core_pipeline/smoke/`. Not a paper result.

```text
python main.py
```

Full GraphCL (10 epochs) then 100-episode BACE test ProtoNet. Overwrites `outputs/core_pipeline/` if you use the default output dir. **Do not do this if you need to keep the completed paper run.**

```text
python main.py --skip-graphcl --checkpoint outputs/core_pipeline/graphcl_checkpoint.pt
```

Reload the canonical checkpoint and re-run few-shot only (new episode RNG unless you also replay saved index lists).

Completed follow-on scripts (already run; outputs exist):

- `ablation_no_graphcl.py` — random-init encoder on **replayed** core episodes
- `ablation_multi_seed.py` — episode seeds 42–46, GraphCL checkpoint read-only
- `prototype_explain.py` — prototype XAI, episode seed 7
- `confidence_analysis.py` — distance-gap vs correctness (resampled seed 42)
- `gnnexplainer_smoke.py`, `gnnexplainer_analysis.py`, `gnnexplainer_temperature.py`, `gnnexplainer_perturbation.py`

`inspect_checkpoint.py` prints encoder config from `graphcl_checkpoint.pt`.
