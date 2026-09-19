"""
models.py

Graph Neural Network models for:
- GIN Encoder
- Prototypical Networks
- Graph Classification
- Explainability

Author: Project Revision
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import (
    GINConv,
    global_add_pool,
    global_mean_pool,
)


# ==========================================================
# GIN ENCODER
# ==========================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder used for
    GraphCL pretraining and Few-Shot learning.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.1,
        pool_type: str = "sum",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.pool_type = pool_type

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):

            input_dim = in_dim if layer == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINConv(
                    mlp,
                    train_eps=True,
                )
            )

            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim)
            )

        self.dropout = nn.Dropout(dropout)

        if pool_type == "sum":
            self.pool = global_add_pool

        elif pool_type == "mean":
            self.pool = global_mean_pool

        else:
            raise ValueError(
                "pool_type must be 'sum' or 'mean'"
            )

        # Projection head for GraphCL

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        data: Data,
        return_node_embeddings: bool = False,
    ):

        x = data.x
        if x is None:
            raise ValueError("Graph is missing node features x.")
        if not torch.is_floating_point(x):
            x = x.float()
        edge_index = data.edge_index

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        graph_embedding = self.pool(x, batch)
        projection = self.projection(graph_embedding)

        if return_node_embeddings:
            return projection, graph_embedding, x

        return projection, graph_embedding, None

    def get_config(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "pool_type": self.pool_type,
        }

    @classmethod
    def from_config(cls, config: dict) -> "GINEncoder":
        return cls(
            in_dim=config["in_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 5),
            dropout=config.get("dropout", 0.1),
            pool_type=config.get("pool_type", "sum"),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        map_location="cpu",
    ) -> "GINEncoder":
        if isinstance(checkpoint, str):
            checkpoint = torch.load(
                checkpoint,
                map_location=map_location,
                weights_only=False,
            )

        if "encoder_config" not in checkpoint:
            raise KeyError(
                "Checkpoint is missing encoder_config. "
                "Refusing to guess architecture (old 3-layer "
                "weights are not compatible with GINEncoder)."
            )

        encoder = cls.from_config(checkpoint["encoder_config"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        return encoder

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        payload = {
            "encoder_state_dict": self.state_dict(),
            "encoder_config": self.get_config(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
# ==========================================================
# PROTOTYPICAL NETWORK
# ==========================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network head for Few-Shot Learning.

    Computes one prototype (mean embedding) per class and
    classifies query samples by distance to prototypes.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
    ):
        super().__init__()

        self.distance_metric = distance_metric
        self.temperature = temperature

    def compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:

        prototypes = []

        for c in range(n_way):

            class_embeddings = embeddings[labels == c]

            if class_embeddings.size(0) == 0:
                raise ValueError(
                    f"No support samples for class {c}"
                )

            prototype = class_embeddings.mean(dim=0)

            prototypes.append(prototype)

        return torch.stack(prototypes)

    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:

        if self.distance_metric == "euclidean":

            distances = torch.cdist(
                query_embeddings,
                prototypes,
                p=2,
            )

        elif self.distance_metric == "cosine":

            query_norm = F.normalize(
                query_embeddings,
                dim=1,
            )

            proto_norm = F.normalize(
                prototypes,
                dim=1,
            )

            distances = 1 - torch.matmul(
                query_norm,
                proto_norm.t(),
            )

        else:

            raise ValueError(
                f"Unknown distance metric: {self.distance_metric}"
            )

        return distances

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int,
    ):

        prototypes = self.compute_prototypes(
            support_embeddings,
            support_labels,
            n_way,
        )

        distances = self.compute_distances(
            query_embeddings,
            prototypes,
        )

        logits = -distances / self.temperature

        return logits, prototypes

# ==========================================================
# GRAPH CLASSIFIER
# ==========================================================

class GraphClassifier(nn.Module):
    """
    Graph classification model built on top of the GIN encoder.
    Used for supervised fine-tuning after GraphCL pretraining.
    """

    def __init__(
        self,
        encoder: GINEncoder,
        num_classes: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder.hidden_dim, num_classes),
        )

    def forward(self, data: Data):

        _, graph_embedding, _ = self.encoder(data)

        logits = self.classifier(graph_embedding)

        return logits


# ==========================================================
# EXPLAINABLE GNN
# ==========================================================

class ExplainableGNN(nn.Module):
    """
    Wrapper around the encoder and classifier to support
    explainability methods such as GNNExplainer.
    """

    def __init__(
        self,
        encoder: GINEncoder,
        classifier: Optional[nn.Module] = None,
        num_classes: int = 2,
    ):
        super().__init__()

        self.encoder = encoder

        if classifier is None:
            classifier = nn.Sequential(
                nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
                nn.ReLU(),
                nn.Linear(encoder.hidden_dim, num_classes),
            )

        self.classifier = classifier

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):

        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            batch=batch,
        )

        _, graph_embedding, node_embeddings = self.encoder(
            data,
            return_node_embeddings=True,
        )

        logits = self.classifier(graph_embedding)

        return logits

    @torch.no_grad()
    def get_graph_embedding(self, data: Data):

        _, graph_embedding, _ = self.encoder(data)

        return graph_embedding

    @torch.no_grad()
    def get_node_embeddings(self, data: Data):

        _, _, node_embeddings = self.encoder(
            data,
            return_node_embeddings=True,
        )

        return node_embeddings