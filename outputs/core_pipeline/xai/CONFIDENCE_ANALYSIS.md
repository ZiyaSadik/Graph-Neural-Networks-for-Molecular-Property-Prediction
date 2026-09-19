# Distance-gap analysis (not calibrated confidence)

Checkpoint: `outputs/core_pipeline/graphcl_checkpoint.pt` (canonical 5-layer GraphCL GINEncoder).  
Protocol: OGB BACE scaffold test, frozen ProtoNet, 2-way 5-shot, 100 episodes, seed **42**.  
Class 0/1 prototypes use original BACE labels. Support and query are disjoint.  
GraphCL was **not** retrained. `|d0-d1|` is **not** a probability and **not** calibrated confidence.

## Query-level counts

- Predictions: 1000
- Overall accuracy: 0.596000
- Correct: 596; incorrect: 404

## Mean distance-gap

- Correct: **13.848330**
- Incorrect: **11.152119**
- Difference (correct − incorrect): **+2.696212**

## Association with correctness

- Point-biserial correlation (gap vs correct): **0.128387**
- AUROC treating gap as a score for being correct: **0.589283**

## Accuracy by gap quantile bins (q=5)

| bin | n | gap min | gap max | gap mean | accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.0238 | 3.8131 | 1.8692 | 0.4850 |
| 1 | 200 | 3.8186 | 8.1972 | 5.9993 | 0.5200 |
| 2 | 200 | 8.2381 | 13.3278 | 10.4812 | 0.6450 |
| 3 | 200 | 13.3320 | 20.5621 | 16.3354 | 0.6250 |
| 4 | 200 | 20.5733 | 55.8285 | 29.1102 | 0.7050 |

Monotone nondecreasing accuracy across bins: False  
Pearson(bin mean gap, bin accuracy): 0.904812

## Verdict

On this run, correct queries had a larger mean gap than incorrect queries, and gap ranked correctness better than chance (AUROC>0.5). That is an empirical association only; the gap is still not a probability.

## Calibration limitations

A calibrated confidence would be a predicted probability whose bin-wise frequency matches the value. Distance-gap is an unnormalized embedding-space margin. It has no mapping to [0, 1] in this experiment, so expected calibration error is not defined for the raw gap. Sum-pooling yields large Euclidean distances; gap magnitude is scale-dependent and not comparable to a likelihood.
