"""Optional GNNExplainer for BACE ProtoNet (exploratory; may fail gracefully)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ExplanationResult:
    succeeded: bool
    method: str
    validated_status: str  # "exploratory" | "not_run" | "failed"
    node_mask: list[float] | None
    edge_mask: list[float] | None
    n_nodes: int | None
    n_edges: int | None
    index_mapping_ok: bool | None
    epochs: int | None
    seed: int | None
    error: str | None
    caveats: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CAVEATS = [
    "GNNExplainer on prototype-distance logits is exploratory in this project.",
    "Historical soft fidelity metrics were often uninformative (fid±≈0).",
    "High atom/bond importance is not chemical causality.",
    "GIN does not use edge_attr; edge masks reflect connectivity only.",
    "Distances and explanation scores are not calibrated probabilities.",
]


def verify_index_mapping(smiles: str, n_nodes: int) -> bool:
    from rdkit import Chem
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    mol = Chem.MolFromSmiles(smiles)
    data = smiles_to_pyg_data(smiles)
    return (
        mol is not None
        and mol.GetNumAtoms() == n_nodes
        and int(data.x.size(0)) == n_nodes
    )


def explain_bace_query(
    encoder,
    prototypes: torch.Tensor,
    smiles: str,
    *,
    epochs: int = 40,
    seed: int = 0,
) -> ExplanationResult:
    """
    Run PyG GNNExplainer via PrototypeDistanceLogits.
    Never invents a mask on failure.
    """
    caveats = list(DEFAULT_CAVEATS)
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        from apps.mol_cli.ogb_graph import smiles_to_pyg_data
        from src.explainability import PrototypeDistanceLogits
    except ImportError as exc:
        return ExplanationResult(
            succeeded=False,
            method="GNNExplainer+PrototypeDistanceLogits",
            validated_status="failed",
            node_mask=None,
            edge_mask=None,
            n_nodes=None,
            n_edges=None,
            index_mapping_ok=None,
            epochs=epochs,
            seed=seed,
            error=f"Missing dependency for GNNExplainer: {exc}",
            caveats=caveats,
        )

    try:
        data = smiles_to_pyg_data(smiles)
        x = data.x.float()
        edge_index = data.edge_index
        n_nodes = int(x.size(0))
        n_edges = int(edge_index.size(1))
        mapping_ok = verify_index_mapping(smiles, n_nodes)
        if not mapping_ok:
            return ExplanationResult(
                succeeded=False,
                method="GNNExplainer+PrototypeDistanceLogits",
                validated_status="failed",
                node_mask=None,
                edge_mask=None,
                n_nodes=n_nodes,
                n_edges=n_edges,
                index_mapping_ok=False,
                epochs=epochs,
                seed=seed,
                error="Atom/node index mapping check failed; refusing to explain.",
                caveats=caveats,
            )

        wrapper = PrototypeDistanceLogits(encoder, prototypes)
        wrapper.eval()
        torch.manual_seed(int(seed))
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=int(epochs)),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )
        explanation = explainer(x, edge_index)
        node_mask = explanation.node_mask.detach().float().reshape(-1).cpu().numpy()
        edge_mask = explanation.edge_mask.detach().float().reshape(-1).cpu().numpy()
        if node_mask.shape[0] != n_nodes:
            return ExplanationResult(
                succeeded=False,
                method="GNNExplainer+PrototypeDistanceLogits",
                validated_status="failed",
                node_mask=None,
                edge_mask=None,
                n_nodes=n_nodes,
                n_edges=n_edges,
                index_mapping_ok=mapping_ok,
                epochs=epochs,
                seed=seed,
                error=(
                    f"Explainer returned node_mask length {node_mask.shape[0]} "
                    f"!= n_nodes {n_nodes}"
                ),
                caveats=caveats,
            )
        return ExplanationResult(
            succeeded=True,
            method="GNNExplainer+PrototypeDistanceLogits",
            validated_status="exploratory",
            node_mask=[float(v) for v in node_mask.tolist()],
            edge_mask=[float(v) for v in edge_mask.tolist()],
            n_nodes=n_nodes,
            n_edges=n_edges,
            index_mapping_ok=True,
            epochs=epochs,
            seed=seed,
            error=None,
            caveats=caveats,
        )
    except Exception as exc:
        return ExplanationResult(
            succeeded=False,
            method="GNNExplainer+PrototypeDistanceLogits",
            validated_status="failed",
            node_mask=None,
            edge_mask=None,
            n_nodes=None,
            n_edges=None,
            index_mapping_ok=None,
            epochs=epochs,
            seed=seed,
            error=str(exc),
            caveats=caveats,
        )
