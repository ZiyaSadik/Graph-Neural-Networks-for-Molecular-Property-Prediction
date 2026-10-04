"""
Convert SMILES to OGB molecular graphs for the research GINEncoder.

Uses the official OGB path:
  ogb.utils.mol.smiles2graph
  + PyG Data construction matching ogb.io.read_graph_pyg

This module is optional. RDKit descriptor CLI paths must not import it at
module load time. Graph construction is a prerequisite for future encoder
inference; it does not validate or produce learned predictions.

Encoder note (research GINEncoder):
  - Expects node features x with in_dim=9 (OGB atom feature vector).
  - Uses edge_index connectivity; does not consume edge_attr in forward().
  - Casts non-floating x to float internally.
  - If data.batch is missing, treats the graph as a single molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

# Matches ogb.utils.mol.smiles2graph and experiment_config encoder.in_dim=9
EXPECTED_NODE_FEAT_DIM = 9
EXPECTED_EDGE_FEAT_DIM = 3


class GraphConversionError(ValueError):
    """User-facing failure converting SMILES to an OGB molecular graph."""


@dataclass(frozen=True)
class OgbGraphSummary:
    """Plain metadata for reports (no tensors)."""

    smiles: str
    num_nodes: int
    num_edges: int
    node_feat_dim: int
    edge_feat_dim: int
    x_dtype: str
    edge_index_dtype: str
    edge_attr_dtype: str
    converter: str


def ogb_graph_deps_available() -> tuple[bool, str]:
    """
    Return (ok, detail). Does not install packages.
    """
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        missing.append("torch_geometric")
    try:
        from ogb.utils.mol import smiles2graph  # noqa: F401
    except ImportError:
        missing.append("ogb")

    if missing:
        return (
            False,
            "Missing packages for OGB graph conversion: "
            + ", ".join(missing)
            + ". Use the Conda environment documented in environment.txt. "
            "This phase does not install packages.",
        )
    return True, "torch, torch_geometric, and ogb are importable."


def _require_deps() -> None:
    ok, detail = ogb_graph_deps_available()
    if not ok:
        raise GraphConversionError(detail)


def smiles_to_ogb_dict(smiles: str) -> dict[str, Any]:
    """
    Convert a SMILES string using official ogb.utils.mol.smiles2graph.

    Returns the OGB graph dict with keys:
      edge_index, edge_feat, node_feat, num_nodes
    """
    _require_deps()

    if smiles is None or not str(smiles).strip():
        raise GraphConversionError(
            "Cannot build an OGB graph: empty SMILES."
        )

    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GraphConversionError(
            f"Cannot build an OGB graph: invalid SMILES {smiles!r}."
        )
    if int(mol.GetNumAtoms()) <= 0:
        raise GraphConversionError(
            f"Cannot build an OGB graph: {smiles!r} has no atoms."
        )

    from ogb.utils.mol import smiles2graph

    try:
        graph = smiles2graph(smiles)
    except Exception as exc:
        raise GraphConversionError(
            f"OGB smiles2graph failed for {smiles!r}: {exc}"
        ) from None

    if not isinstance(graph, dict):
        raise GraphConversionError(
            "OGB smiles2graph returned an unexpected type; refusing approximate features."
        )

    required = ("edge_index", "edge_feat", "node_feat", "num_nodes")
    for key in required:
        if key not in graph:
            raise GraphConversionError(
                f"OGB smiles2graph omitted key {key!r}; refusing approximate features."
            )

    _validate_ogb_arrays(graph, smiles)
    return graph


def _validate_ogb_arrays(graph: dict[str, Any], smiles: str) -> None:
    import numpy as np

    node_feat = graph["node_feat"]
    edge_feat = graph["edge_feat"]
    edge_index = graph["edge_index"]
    num_nodes = int(graph["num_nodes"])

    if not isinstance(node_feat, np.ndarray) or node_feat.ndim != 2:
        raise GraphConversionError(
            f"Invalid node_feat for {smiles!r}; refusing approximate features."
        )
    if node_feat.shape[0] != num_nodes:
        raise GraphConversionError(
            f"num_nodes ({num_nodes}) does not match node_feat rows "
            f"({node_feat.shape[0]}) for {smiles!r}."
        )
    if node_feat.shape[1] != EXPECTED_NODE_FEAT_DIM:
        raise GraphConversionError(
            f"Expected node feature dim {EXPECTED_NODE_FEAT_DIM}, "
            f"got {node_feat.shape[1]} for {smiles!r}."
        )
    if node_feat.dtype != np.int64:
        raise GraphConversionError(
            f"Expected node_feat dtype int64, got {node_feat.dtype} for {smiles!r}."
        )

    if not isinstance(edge_index, np.ndarray) or edge_index.shape[0] != 2:
        raise GraphConversionError(
            f"Invalid edge_index for {smiles!r}; refusing approximate features."
        )
    if edge_index.dtype != np.int64:
        raise GraphConversionError(
            f"Expected edge_index dtype int64, got {edge_index.dtype} for {smiles!r}."
        )

    n_edges = int(edge_index.shape[1])
    if not isinstance(edge_feat, np.ndarray) or edge_feat.ndim != 2:
        raise GraphConversionError(
            f"Invalid edge_feat for {smiles!r}; refusing approximate features."
        )
    if edge_feat.shape[0] != n_edges:
        raise GraphConversionError(
            f"edge_feat rows ({edge_feat.shape[0]}) != edge_index columns "
            f"({n_edges}) for {smiles!r}."
        )
    if edge_feat.shape[1] != EXPECTED_EDGE_FEAT_DIM:
        raise GraphConversionError(
            f"Expected edge feature dim {EXPECTED_EDGE_FEAT_DIM}, "
            f"got {edge_feat.shape[1]} for {smiles!r}."
        )
    if edge_feat.dtype != np.int64:
        raise GraphConversionError(
            f"Expected edge_feat dtype int64, got {edge_feat.dtype} for {smiles!r}."
        )

    if n_edges > 0:
        if n_edges % 2 != 0:
            raise GraphConversionError(
                f"OGB graphs are bidirectional; odd edge count {n_edges} for {smiles!r}."
            )
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise GraphConversionError(
                f"edge_index out of range for {smiles!r}."
            )


def smiles_to_pyg_data(smiles: str):
    """
    Build a torch_geometric.data.Data object matching OGB dataset loading.

    Mirrors ogb.io.read_graph_pyg conversion of smiles2graph outputs:
      x = node_feat (int64)
      edge_index (int64, shape [2, E], bidirectional)
      edge_attr = edge_feat (int64, shape [E, 3])
      num_nodes

    Does not load checkpoints or run the encoder.
    """
    _require_deps()
    import torch
    from torch_geometric.data import Data

    graph = smiles_to_ogb_dict(smiles)

    data = Data()
    data.num_nodes = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"])
    data.edge_attr = torch.from_numpy(graph["edge_feat"])
    data.x = torch.from_numpy(graph["node_feat"])

    if data.x.shape[1] != EXPECTED_NODE_FEAT_DIM:
        raise GraphConversionError(
            f"PyG Data x must have dim {EXPECTED_NODE_FEAT_DIM}, got {tuple(data.x.shape)}."
        )
    if data.edge_attr.shape[1] != EXPECTED_EDGE_FEAT_DIM and data.edge_attr.numel() > 0:
        raise GraphConversionError(
            f"PyG Data edge_attr must have dim {EXPECTED_EDGE_FEAT_DIM}, "
            f"got {tuple(data.edge_attr.shape)}."
        )

    return data


def summarize_pyg_data(smiles: str, data) -> OgbGraphSummary:
    """Build a tensor-free summary for terminal display."""
    return OgbGraphSummary(
        smiles=smiles,
        num_nodes=int(data.num_nodes),
        num_edges=int(data.edge_index.size(1)),
        node_feat_dim=int(data.x.size(1)),
        edge_feat_dim=int(data.edge_attr.size(1)) if data.edge_attr.numel() else EXPECTED_EDGE_FEAT_DIM,
        x_dtype=str(data.x.dtype).replace("torch.", ""),
        edge_index_dtype=str(data.edge_index.dtype).replace("torch.", ""),
        edge_attr_dtype=str(data.edge_attr.dtype).replace("torch.", ""),
        converter="ogb.utils.mol.smiles2graph",
    )
