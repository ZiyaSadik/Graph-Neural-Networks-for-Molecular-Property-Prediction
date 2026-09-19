# Multi-seed robustness: GraphCL vs No-GraphCL

GraphCL was **not** retrained. Checkpoint `outputs/core_pipeline/graphcl_checkpoint.pt` was loaded read-only.

## Pairing check (before evaluation)

- Resampling seed 42 twice produced **identical** support/query index sets (100 episodes).
- Seed 42 vs 43 differed in **100/100** episodes.
- For every reported seed, the same index list was used for both encoders.

## Protocol

OGB BACE scaffold **test** only; 2-way 5-shot 5-query; frozen encoder; Euclidean ProtoNet; 100 episodes per seed; seeds [42, 43, 44, 45, 46].

GraphCL weights: core MolPCBA checkpoint (fixed).  
No-GraphCL: random `GINEncoder` init with the same seed as episode sampling (seed reset before init, independent of the episode RNG stream after sampling).

## Per-seed results

| seed | GraphCL mean | GraphCL std | No-GraphCL mean | No-GraphCL std | GraphCL − No-GraphCL |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.597000 | 0.164593 | 0.680000 | 0.193391 | -0.083000 |
| 43 | 0.591000 | 0.167389 | 0.647000 | 0.184095 | -0.056000 |
| 44 | 0.576000 | 0.145685 | 0.645000 | 0.170514 | -0.069000 |
| 45 | 0.591000 | 0.170936 | 0.645000 | 0.171683 | -0.054000 |
| 46 | 0.587000 | 0.127793 | 0.653000 | 0.144537 | -0.066000 |

## Across seeds (n=5; sample std, ddof=1)

| condition | mean of seed-means | std of seed-means |
|---|---:|---:|
| GraphCL | 0.588400 | 0.007797 |
| No-GraphCL | 0.654000 | 0.014900 |

Mean difference (GraphCL − No-GraphCL): **-0.065600** ± 0.011632

Seeds with GraphCL higher: 0/5.  
Seeds with No-GraphCL higher: 5/5.

This is a **robustness analysis**. It does not by itself establish a general claim that GraphCL helps or hurts few-shot BACE accuracy.

## Limitations

- Five episode seeds only; one GraphCL pretraining run.
- No-GraphCL BatchNorm running stats remain at defaults.
- Original core-run episodes (sampled after GraphCL training RNG) are a different draw than seed-42 here.
