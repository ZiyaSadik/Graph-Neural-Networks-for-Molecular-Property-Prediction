"""
Prototype-based explanations for the canonical GINEncoder.

This module does not implement GNNExplainer. It reports query-to-prototype
distances and the nearest support molecule after a ProtoNet episode.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


class PrototypeExplainer:
    def __init__(self, encoder, protonet, device="cpu"):
        self.encoder = encoder.to(device)
        self.protonet = protonet
        self.device = device

    @torch.no_grad()
    def explain_episode(
        self,
        support_batch,
        support_labels,
        query_batch,
        query_labels,
        n_way: int,
        query_indices: Optional[List[int]] = None,
        support_indices: Optional[List[int]] = None,
        query_ids: Optional[List[str]] = None,
        support_ids: Optional[List[str]] = None,
        query_smiles: Optional[List[str]] = None,
        support_smiles: Optional[List[str]] = None,
    ):
        self.encoder.eval()

        support_batch = support_batch.to(self.device)
        query_batch = query_batch.to(self.device)
        support_labels = support_labels.to(self.device)
        query_labels = query_labels.to(self.device)

        _, support_emb, _ = self.encoder(support_batch)
        _, query_emb, _ = self.encoder(query_batch)
        logits, prototypes = self.protonet(
            support_emb, support_labels, query_emb, n_way
        )
        preds = logits.argmax(dim=1)
        distances = torch.cdist(query_emb, prototypes, p=2)

        rows = []
        for i in range(query_emb.size(0)):
            dist_row = distances[i]
            d0 = float(dist_row[0].item())
            d1 = float(dist_row[1].item()) if dist_row.numel() > 1 else float("nan")
            pred = int(preds[i].item())
            true = int(query_labels[i].item())
            sims = F.cosine_similarity(query_emb[i].unsqueeze(0), support_emb)
            nearest = int(sims.argmax().item())
            q_idx = None if query_indices is None else query_indices[i]
            s_idx = None if support_indices is None else support_indices[nearest]
            q_id = None if query_ids is None else query_ids[i]
            s_id = None if support_ids is None else support_ids[nearest]
            rows.append(
                {
                    "query_position": i,
                    "query_index": q_idx,
                    "query_molecule_id": q_id if q_id is not None else (
                        None if q_idx is None else f"ogbg-molbace/{q_idx}"
                    ),
                    "query_smiles": (
                        None if query_smiles is None else query_smiles[i]
                    ),
                    "ground_truth_label": true,
                    "predicted_class": pred,
                    "correct": pred == true,
                    "distance_to_class_0_prototype": d0,
                    "distance_to_class_1_prototype": d1,
                    "distance_gap_confidence": abs(d0 - d1),
                    "most_similar_support_position": nearest,
                    "most_similar_support_index": s_idx,
                    "most_similar_support_id": s_id if s_id is not None else (
                        None if s_idx is None else f"ogbg-molbace/{s_idx}"
                    ),
                    "most_similar_support_smiles": (
                        None if support_smiles is None else support_smiles[nearest]
                    ),
                    "cosine_similarity_to_nearest_support": float(
                        sims[nearest].item()
                    ),
                    "true_label": true,
                    "pred_label": pred,
                    "distances_to_prototypes": [d0, d1],
                    "confidence_abs_diff": abs(d0 - d1),
                    "nearest_support_position": nearest,
                    "nearest_support_index": s_idx,
                    "nearest_support_cosine": float(sims[nearest].item()),
                }
            )

        return {
            "n_query": len(rows),
            "accuracy": float(sum(r["correct"] for r in rows) / max(len(rows), 1)),
            "explanations": rows,
            "n_support": int(support_emb.size(0)),
        }


class PrototypeDistanceLogits(torch.nn.Module):
    """
    Smallest wrapper that makes prototype-distance classification look like
    a graph-level model GNNExplainer can call: model(x, edge_index, batch).

    Prototypes are fixed buffers (from support embeddings). Encoder weights
    stay frozen. Logits are -Euclidean distances, matching PrototypicalNetwork
    with temperature=1. This is not a new classifier head and does not use
    the GraphCL projection head.
    """

    def __init__(self, encoder, prototypes: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        if prototypes.dim() != 2:
            raise ValueError("prototypes must have shape [n_way, hidden_dim]")
        self.register_buffer("prototypes", prototypes.detach().clone())

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.data import Data

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        _, graph_embedding, _ = self.encoder(data)
        distances = torch.cdist(graph_embedding, self.prototypes, p=2)
        return -distances


class TemperatureScaledPrototypeLogits(PrototypeDistanceLogits):
    """
    Explainer-only scaling: logit_c = -d_c / T.

    Argmax is identical to PrototypeDistanceLogits for any T > 0.
    This does not change the ProtoNet classifier or encoder weights.
    """

    def __init__(self, encoder, prototypes: torch.Tensor, temperature: float = 10.0):
        super().__init__(encoder, prototypes)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, x, edge_index, batch=None):
        return super().forward(x, edge_index, batch) / self.temperature

