"""
GNNExplainer API smoke test for prototype-distance prediction.

One OGB BACE scaffold-test query. Does not retrain GraphCL or change
the checkpoint. Does not replace ProtoNet with a classifier head.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer

from prototype_explain import (
    load_bace_mapping,
    molecule_record,
    sample_original_label_episode,
)
from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.explainability import PrototypeDistanceLogits
from src.models import GINEncoder, PrototypicalNetwork
from ogb.graphproppred import PygGraphPropPredDataset

allow_pyg_torch_load()

CHECKPOINT = Path("outputs/core_pipeline/graphcl_checkpoint.pt")
OUT_DIR = Path("outputs/core_pipeline/xai/gnnexplainer_smoke_test")
SEED = 7
K_SHOT = 5
Q_QUERY = 5
QUERY_POSITION = 0
EXPLAIN_EPOCHS = 40


def mask_stats(mask: torch.Tensor) -> dict:
    t = mask.detach().float().reshape(-1)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "sum": float(t.sum().item()),
    }


def main() -> int:
    import torch_geometric as pyg

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    result = {
        "pyg_version": pyg.__version__,
        "gnnexplainer_api": "torch_geometric.explain.Explainer + GNNExplainer",
        "checkpoint": str(CHECKPOINT),
        "succeeded": False,
        "failure": None,
    }

    try:
        encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
        encoder = encoder.to(device)
        encoder.eval()
        cfg = encoder.get_config()
        if cfg["num_layers"] != 5:
            raise RuntimeError(f"Non-canonical encoder: {cfg}")
        result["encoder_config"] = cfg
        result["checkpoint_loaded"] = True

        dataset = PygGraphPropPredDataset(name="ogbg-molbace", root="dataset")
        protocol = get_bace_held_out_protocol(dataset)
        mapping = load_bace_mapping(dataset)
        episode = sample_original_label_episode(
            dataset, protocol, K_SHOT, Q_QUERY, SEED
        )
        q_idx = int(episode["query_indices"][QUERY_POSITION])
        rec = molecule_record(mapping, q_idx)
        query = dataset[q_idx]
        x = query.x.float()
        edge_index = query.edge_index
        n_nodes = int(x.size(0))
        n_edges = int(edge_index.size(1))

        protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
        support_batch = episode["support_batch"].to(device)
        support_labels = episode["support_labels"].to(device)
        query_batch = Batch.from_data_list([query]).to(device)
        with torch.no_grad():
            _, support_emb, _ = encoder(support_batch)
            _, query_emb, _ = encoder(query_batch)
            logits, prototypes = protonet(
                support_emb,
                support_labels,
                query_emb,
                n_way=2,
            )
            pred = int(logits.argmax(dim=1).item())
            true = int(episode["query_labels"][QUERY_POSITION].item())
            distances = torch.cdist(query_emb, prototypes, p=2).squeeze(0)

        result.update(
            {
                "molecule_id": rec["molecule_id"],
                "dataset_index": q_idx,
                "true_label": true,
                "predicted_label": pred,
                "n_nodes_atoms": n_nodes,
                "n_edges_bonds": n_edges,
                "distance_to_class_0_prototype": float(distances[0].item()),
                "distance_to_class_1_prototype": float(distances[1].item()),
                "encoder_forward_ok": True,
                "prototype_prediction_ok": True,
                "episode_seed": SEED,
                "query_position": QUERY_POSITION,
                "support_indices": episode["support_indices"],
                "explanation_objective": (
                    "Prototype-distance logits: logit_c = -||z_query - prototype_c||_2 "
                    "with prototypes fixed from the episode support set. "
                    "Matches PrototypicalNetwork (euclidean, temperature=1). "
                    "Not the projection head. Not a supervised classifier."
                ),
            }
        )

        model = PrototypeDistanceLogits(encoder, prototypes.to(device))
        model.eval()
        with torch.no_grad():
            wrap_logits = model(x.to(device), edge_index.to(device))
            wrap_pred = int(wrap_logits.argmax(dim=1).item())
        if wrap_pred != pred:
            raise RuntimeError(
                f"Wrapper prediction {wrap_pred} != ProtoNet prediction {pred}"
            )
        result["wrapper_matches_protonet"] = True

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS, lr=0.01),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        explanation = explainer(
            x.to(device),
            edge_index.to(device),
            index=0,
            batch=batch,
        )
        node_mask = explanation.get("node_mask")
        edge_mask = explanation.get("edge_mask")
        if node_mask is None or edge_mask is None:
            raise RuntimeError(
                f"Missing masks: node_mask={node_mask is not None}, "
                f"edge_mask={edge_mask is not None}"
            )

        node_stats = mask_stats(node_mask)
        edge_stats = mask_stats(edge_mask)
        result.update(
            {
                "gnnexplainer_succeeded": True,
                "succeeded": True,
                "node_mask_stats": node_stats,
                "edge_mask_stats": edge_stats,
                "node_mask_shape": list(node_mask.shape),
                "edge_mask_shape": list(edge_mask.shape),
                "explain_epochs": EXPLAIN_EPOCHS,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].bar(np.arange(n_nodes), node_mask.detach().cpu().view(-1).numpy())
        axes[0].set_title("Node/atom mask")
        axes[0].set_xlabel("atom index")
        axes[1].hist(edge_mask.detach().cpu().view(-1).numpy(), bins=20, color="steelblue")
        axes[1].set_title("Edge/bond mask distribution")
        fig.suptitle(
            f"{rec['molecule_id']}  true={true} pred={pred}  "
            f"(prototype-distance GNNExplainer smoke)",
            fontsize=10,
        )
        fig.tight_layout()
        fig_path = OUT_DIR / "masks.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        result["figure"] = str(fig_path)

        try:
            explanation.visualize_graph(str(OUT_DIR / "graph.png"))
            result["graph_visualization"] = str(OUT_DIR / "graph.png")
        except Exception as vis_err:
            result["graph_visualization"] = None
            result["graph_visualization_error"] = str(vis_err)

    except Exception as exc:
        result["succeeded"] = False
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (OUT_DIR / "smoke_test.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        md = _render_md(result)
        (OUT_DIR / "GNNEXPLAINER_SMOKE_TEST.md").write_text(md, encoding="utf-8")

    print("=" * 50)
    print("GNNExplainer smoke:", "OK" if result.get("succeeded") else "FAILED")
    print(json.dumps({k: result[k] for k in ("pyg_version", "molecule_id", "true_label", "predicted_label", "succeeded") if k in result}, indent=2))
    print("=" * 50)
    return 0 if result.get("succeeded") else 1


def _render_md(result: dict) -> str:
    ok = result.get("succeeded")
    fail = result.get("failure")
    lines = [
        "# GNNExplainer smoke test",
        "",
        f"- PyG version: `{result.get('pyg_version')}`",
        f"- API: `{result.get('gnnexplainer_api')}`",
        f"- Checkpoint: `{result.get('checkpoint')}`",
        f"- Succeeded: **{ok}**",
        "",
        "## Limitation and wrapper",
        "",
        "GNNExplainer (`torch_geometric.explain.Explainer`) requires a model callable as `model(x, edge_index, **kwargs)` that returns class scores. The project predictor is ProtoNet Euclidean distance to support prototypes, not a trained graph classifier and not the GraphCL projection head.",
        "",
        "The smoke test therefore uses `PrototypeDistanceLogits`: frozen `GINEncoder` graph embeddings, **fixed** support prototypes, `logit_c = -||z - p_c||_2`. That is the same score as `PrototypicalNetwork` with temperature 1. Prototypes are not updated. Encoder weights are not trained.",
        "",
        "`ExplainableGNN` in `src/models.py` attaches an unrelated MLP classifier and was **not** used.",
        "",
    ]
    if fail:
        lines += ["## Failure", "", f"`{fail}`", ""]
        return "\n".join(lines)
    lines += [
        "## Result",
        "",
        f"- Molecule ID: `{result.get('molecule_id')}`",
        f"- Dataset index: {result.get('dataset_index')}",
        f"- True label: {result.get('true_label')}",
        f"- Predicted label: {result.get('predicted_label')}",
        f"- Nodes/atoms: {result.get('n_nodes_atoms')}",
        f"- Edges/bonds: {result.get('n_edges_bonds')}",
        f"- Wrapper matches ProtoNet: {result.get('wrapper_matches_protonet')}",
        f"- GNNExplainer epochs: {result.get('explain_epochs')}",
        "",
        f"- Node mask stats: `{result.get('node_mask_stats')}`",
        f"- Edge mask stats: `{result.get('edge_mask_stats')}`",
        "",
        f"- Mask figure: `{result.get('figure')}`",
        f"- Graph viz: `{result.get('graph_visualization')}`",
        "",
        "This is a **one-molecule API smoke test**, not a full XAI experiment.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
