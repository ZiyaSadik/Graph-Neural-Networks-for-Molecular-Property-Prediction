# GNNExplainer smoke test

- PyG version: `2.7.0`
- API: `torch_geometric.explain.Explainer + GNNExplainer`
- Checkpoint: `outputs\core_pipeline\graphcl_checkpoint.pt`
- Succeeded: **True**

## Limitation and wrapper

GNNExplainer (`torch_geometric.explain.Explainer`) requires a model callable as `model(x, edge_index, **kwargs)` that returns class scores. The project predictor is ProtoNet Euclidean distance to support prototypes, not a trained graph classifier and not the GraphCL projection head.

The smoke test therefore uses `PrototypeDistanceLogits`: frozen `GINEncoder` graph embeddings, **fixed** support prototypes, `logit_c = -||z - p_c||_2`. That is the same score as `PrototypicalNetwork` with temperature 1. Prototypes are not updated. Encoder weights are not trained.

`ExplainableGNN` in `src/models.py` attaches an unrelated MLP classifier and was **not** used.

## Result

- Molecule ID: `BACE_118`
- Dataset index: 116
- True label: 0
- Predicted label: 1
- Nodes/atoms: 34
- Edges/bonds: 76
- Wrapper matches ProtoNet: True
- GNNExplainer epochs: 40

- Node mask stats: `{'n': 34, 'mean': 0.41513991355895996, 'std': 0.03261922299861908, 'min': 0.36168792843818665, 'max': 0.49302545189857483, 'sum': 14.114757537841797}`
- Edge mask stats: `{'n': 76, 'mean': 0.40703311562538147, 'std': 0.0832936018705368, 'min': 0.24558091163635254, 'max': 0.7065698504447937, 'sum': 30.93451690673828}`

- Mask figure: `outputs\core_pipeline\xai\gnnexplainer_smoke_test\masks.png`
- Graph viz: `outputs\core_pipeline\xai\gnnexplainer_smoke_test\graph.png`

This is a **one-molecule API smoke test**, not a full XAI experiment.
