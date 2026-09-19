# Ablation: GraphCL vs No-GraphCL

## Method

**No-GraphCL encoder training/initialization**

- Same canonical `GINEncoder` (5 layers, hidden 128, dropout 0.1, sum pool, `in_dim=9`).
- Weights drawn from PyTorch default initialization after `seed=42`.
- **Zero optimization steps.** No GraphCL, no MSE-to-noise, no BACE supervised training, no ProtoNet meta-training.
- Encoder is frozen (`eval`, `no_grad`) during few-shot evaluation, matching the GraphCL condition.

This is a fair comparison of *whether GraphCL pretraining changes few-shot ProtoNet accuracy* because the downstream protocol is held fixed: OGB BACE scaffold test, 2-way 5-shot, 5-query, 100 episodes, Euclidean ProtoNet, same code path. Episode support/query **index sets are replayed** from `outputs/core_pipeline/fewshot_results.csv` so the two encoders see the same molecules.

GraphCL results are **read from the completed core experiment**, not recomputed. The GraphCL checkpoint was not modified and GraphCL was not rerun.

## Results

| condition | mean_accuracy | std_accuracy | min | max | episodes | seed |
|---|---:|---:|---:|---:|---:|---:|
| graphcl | 0.5910000041872263 | 0.16976159420898532 | 0.10000000149011612 | 1.0 | 100 | 42 |
| no_graphcl | 0.6590000019967556 | 0.17268178536105985 | 0.0 | 1.0 | 100 | 42 |

No-GraphCL minus GraphCL mean accuracy: **+0.068000**.

On this protocol, the No-GraphCL encoder mean accuracy is higher. Do not claim that GraphCL improves few-shot accuracy.

## Limitations

- Single seed (42).
- No-GraphCL BatchNorm running statistics are uninitialized defaults; GraphCL has statistics from MolPCBA. That is part of pretraining, not a protocol change.
- Replay uses the GraphCL run's episode list (those episodes were sampled after GraphCL RNG use). That is more controlled than resampling new episodes.
- ProtoNet does not update the encoder in either condition; this ablation does not test BACE fine-tuning.
- Historical 51.8% and 9/10 diagnostics are unused.
