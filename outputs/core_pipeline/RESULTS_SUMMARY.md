# Core pipeline results summary

This document reports the **corrected core experiment** only.

The historical diagnostic numbers **0.5180 mean accuracy** (`outputs/final_results.txt`, MSE-to-random-noise on BACE) and the **9/10 (90%) single-episode** explainer demo are **not used**. They are not GraphCL results and are not this evaluation protocol.

Verification status: **all primary artifacts are consistent** with the completed terminal run (`exit code 0`).

---

## GraphCL pretraining (NT-Xent)

| Epoch | Loss | Wall time (s) | LR |
|------:|-----:|--------------:|---:|
| 1 | 1.8324954531507698 | 1998.7324459552765 | 0.001 |
| 2 | 1.5613343873127101 | 2828.121070623398 | 0.001 |
| 3 | 1.5076903425623003 | 2207.2151594161987 | 0.001 |
| 4 | 1.4793875113974393 | 1542.683307647705 | 0.001 |
| 5 | 1.4626832704388002 | 1318.1972308158875 | 0.001 |
| 6 | 1.4515298697106815 | 1382.5594553947449 | 0.001 |
| 7 | 1.4425618419829758 | 1115.5227780342102 | 0.001 |
| 8 | 1.4365129520815063 | 1197.285846710205 | 0.001 |
| 9 | 1.4314094891261107 | 1172.005663394928 | 0.001 |
| 10 | 1.4264949777891314 | 1140.0303900241852 | 0.001 |

- **Source:** `graphcl_training_history.csv` (10 rows, epochs 1–10). Matches terminal prints to 4 decimals.
- **Total pretraining time:** 15902.353348016739 s (**4.417 h**, **265.04 min**).
- **Final GraphCL loss (epoch 10):** 1.4264949777891314.
- **Dataset:** `ogbg-molpcba`, labels unused, **437929** graphs, **no BACE**.
- **Method:** GraphCL, NT-Xent, temperature 0.2, batch size 128, Adam lr 0.001, weight decay 1e-5.
- **Encoder:** `GINEncoder`, 5 layers, hidden 128, dropout 0.1, sum pool, `in_dim=9`, **184,197** parameters.
- **Device:** CPU. **Seed:** 42.

Terminal checkpoint line: `Checkpoint reload verified (max abs embedding diff=0.00e+00)`.

---

## Checkpoint

- **Path:** `outputs/core_pipeline/graphcl_checkpoint.pt`
- **Exists:** yes.
- **Used for evaluation:** yes. After GraphCL, `main.py` reloads this file into a new `GINEncoder` (`GINEncoder.from_checkpoint`) and only then runs ProtoNet. Config `paths.checkpoint` is the same path. Checkpoint metadata: `pretrain_dataset=ogbg-molpcba`, `pretrain_uses_labels=false`, `bace_used_in_pretrain=false`, `n_pretrain_graphs=437929`, `seed=42`, `smoke=false`, `encoder_config` matches the 5×128 GIN above.

---

## Few-shot evaluation protocol

- **Downstream dataset:** `ogbg-molbace`
- **Split:** official OGB **scaffold** split; episodes sampled from **test only** (`ogb_scaffold_test_only`)
- **Reserved (not used for pretraining or few-shot):** train 1210, valid 151
- **Few-shot pool:** 152 test molecules (class 0: 71, class 1: 81)
- **Setting:** 2-way, 5-shot, 5-query, Euclidean ProtoNet
- **Episodes:** 100 (IDs 0–99 in `fewshot_results.csv`)
- **Seed:** 42
- **Support/query overlap:** empty (`[]`) in every episode
- **Encoder:** reloaded GraphCL checkpoint (MolPCBA only); BACE labels are used only for episodic support/query assignment on the test pool

---

## Few-shot metrics

Source: `fewshot_results.csv` (100 episodes) and `fewshot_summary.json`. These match the terminal line `Accuracy: 0.5910 ± 0.1698 over 100 episodes` (4-decimal print of the same floats).

| Quantity | Value |
|---|---|
| Mean accuracy | 0.5910000041872263 |
| Standard deviation (population, `np.std`) | 0.16976159420898532 |
| Min accuracy | 0.10000000149011612 |
| Max accuracy | 1.0 |
| Mean episode loss | 4.513656648136675 |
| Number of episodes | 100 |
| Seed | 42 |

Terminal display: **0.5910 ± 0.1698** (100 episodes).

---

## Config check (`experiment_config.json`)

| Required | Recorded | OK |
|---|---|---|
| seed 42 | 42 | yes |
| 10 GraphCL epochs | 10 | yes |
| batch size 128 | 128 | yes |
| 2-way 5-shot | n_way 2, k_shot 5, q_query 5 | yes |
| 100 episodes | 100 | yes |
| MolPCBA pretraining | `ogbg-molpcba`, 437929 graphs, labels unused | yes |
| Held-out OGB BACE | `ogbg-molbace`, OGB scaffold test-only | yes |
| smoke / paper flag | `smoke: false`, `is_paper_result: true` | yes |

---

## Explicit exclusions

- Do **not** cite **51.8%** from `outputs/final_results.txt`.
- Do **not** cite the **90% / 9-of-10** diagnostic episode.
- Do **not** cite `outputs/core_pipeline/smoke/` (pipeline check only).
