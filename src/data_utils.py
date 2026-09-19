"""
data_utils.py

Utilities for:
- Loading molecular datasets
- Graph augmentations (GraphCL)
- Few-shot episode sampling
- Feature projection
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import MoleculeNet
from ogb.graphproppred import PygGraphPropPredDataset

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


def allow_pyg_torch_load():
    """OGB processed graphs need PyG classes registered for torch.load."""
    add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])


allow_pyg_torch_load()


# ==========================================================
# DATA LOADER
# ==========================================================

class MolecularDataLoader:
    """
    Utility functions for loading molecular datasets.
    """

    @staticmethod
    def load_ogb(
        name: str = "ogbg-molpcba",
        root: str = "data/ogb",
    ):

        dataset = PygGraphPropPredDataset(
            name=name,
            root=root,
        )

        print(f"✓ Loaded {name}: {len(dataset)} molecules")

        return dataset

    @staticmethod
    def load_moleculenet(
        name: str = "BACE",
        root: str = "data/moleculenet",
    ):

        dataset = MoleculeNet(
            root=root,
            name=name,
        )

        print(f"✓ Loaded MoleculeNet {name}: {len(dataset)} molecules")

        return dataset

    @staticmethod
    def ensure_node_features(
        dataset,
        default_dim: int = 9,
    ):
        """
        Create simple node features if a graph has none.
        """

        for graph in dataset:

            if graph.x is None or graph.x.size(0) == 0:

                graph.x = torch.ones(
                    (
                        graph.num_nodes,
                        default_dim,
                    ),
                    dtype=torch.float,
                )

        return dataset

    @staticmethod
    def get_scaffold(
        smiles: str,
    ):

        try:

            mol = Chem.MolFromSmiles(smiles)

            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )

            return scaffold

        except Exception:

            return ""

    @staticmethod
    def scaffold_split(
        dataset,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
    ):

        assert abs(
            frac_train +
            frac_val +
            frac_test -
            1.0
        ) < 1e-6

        scaffold_dict = {}

        for idx, graph in enumerate(dataset):

            if hasattr(graph, "smiles"):

                scaffold = MolecularDataLoader.get_scaffold(
                    graph.smiles
                )

            else:

                scaffold = str(idx)

            scaffold_dict.setdefault(
                scaffold,
                [],
            ).append(idx)

        scaffold_sets = sorted(
            scaffold_dict.values(),
            key=len,
            reverse=True,
        )

        train_idx = []
        val_idx = []
        test_idx = []

        total = len(dataset)

        for group in scaffold_sets:

            if len(train_idx) / total < frac_train:

                train_idx.extend(group)

            elif len(val_idx) / total < frac_val:

                val_idx.extend(group)

            else:

                test_idx.extend(group)

        print(
            f"✓ Scaffold split:"
            f" train={len(train_idx)},"
            f" val={len(val_idx)},"
            f" test={len(test_idx)}"
        )

        return (
            train_idx,
            val_idx,
            test_idx,
        )


def get_bace_held_out_protocol(dataset):
    """
    Official OGB scaffold split for ogbg-molbace.

    Pretraining must NOT use this dataset at all.
    Few-shot support and query are drawn only from the test split.
    Train and valid indices are reserved for future BACE supervised
    training and are excluded from few-shot evaluation.
    """
    if not hasattr(dataset, "get_idx_split"):
        raise ValueError(
            "BACE held-out protocol requires an OGB dataset with get_idx_split()."
        )

    split = dataset.get_idx_split()
    train_idx = [int(i) for i in split["train"].tolist()]
    valid_idx = [int(i) for i in split["valid"].tolist()]
    test_idx = [int(i) for i in split["test"].tolist()]

    train_set = set(train_idx)
    valid_set = set(valid_idx)
    test_set = set(test_idx)

    if (train_set & test_set) or (valid_set & test_set) or (train_set & valid_set):
        raise RuntimeError(
            "OGB BACE split is not disjoint; refusing to evaluate."
        )

    return {
        "dataset_name": "ogbg-molbace",
        "split_name": "ogb_scaffold",
        "train_reserved": train_idx,
        "valid_reserved": valid_idx,
        "fewshot_pool": test_idx,
        "n_train_reserved": len(train_idx),
        "n_valid_reserved": len(valid_idx),
        "n_fewshot_pool": len(test_idx),
        "rule": (
            "Encoder is pretrained only on ogbg-molpcba graphs (labels unused). "
            "BACE train/valid are reserved and never used for pretraining or "
            "few-shot support/query. Episodes are sampled from BACE test only; "
            "within an episode, support and query index sets are disjoint."
        ),
    }


def _graph_binary_label(graph):
    if not hasattr(graph, "y") or graph.y is None:
        return None
    if graph.y.numel() != 1:
        return None
    return int(graph.y.item())


# ==========================================================
# GRAPH AUGMENTATION
# ==========================================================

class GraphAugmentation:
    """
    Graph augmentations used in GraphCL.
    """

    @staticmethod
    def node_drop(
        data: Data,
        drop_prob: float = 0.1,
    ) -> Data:

        device = data.x.device

        node_mask = (
            torch.rand(
                data.num_nodes,
                device=device,
            )
            > drop_prob
        )

        if not node_mask.any():
            node_mask[0] = True

        mapping = torch.full(
            (data.num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )

        mapping[node_mask] = torch.arange(
            node_mask.sum(),
            device=device,
        )

        new_x = data.x[node_mask]

        src = data.edge_index[0]
        dst = data.edge_index[1]

        edge_mask = (
            node_mask[src]
            &
            node_mask[dst]
        )

        new_edge_index = mapping[
            data.edge_index[:, edge_mask]
        ]

        return Data(
            x=new_x,
            edge_index=new_edge_index,
            y=data.y,
        )

    @staticmethod
    def edge_perturbation(
        data: Data,
        perturb_ratio: float = 0.1,
    ) -> Data:

        device = data.x.device

        keep_mask = (
            torch.rand(
                data.edge_index.size(1),
                device=device,
            )
            > perturb_ratio
        )

        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index[:, keep_mask],
            y=data.y,
        )

    @staticmethod
    def attribute_masking(
        data: Data,
        mask_prob: float = 0.1,
    ) -> Data:

        x = data.x.clone()

        device = x.device

        mask = (
            torch.rand(
                x.size(0),
                device=device,
            )
            < mask_prob
        )

        if mask.any():

            x[mask] = 0

        return Data(
            x=x,
            edge_index=data.edge_index.clone(),
            y=data.y,
        )

    @staticmethod
    def random_augment(
        data: Data,
        aug_prob: float = 0.33,
    ) -> Data:

        r = random.random()

        if r < aug_prob:

            return GraphAugmentation.node_drop(
                data,
                drop_prob=0.15,
            )

        elif r < 2 * aug_prob:

            return GraphAugmentation.edge_perturbation(
                data,
                perturb_ratio=0.15,
            )

        else:

            return GraphAugmentation.attribute_masking(
                data,
                mask_prob=0.12,
            )

# ==========================================================
# EPISODE SAMPLER
# ==========================================================

class EpisodeSampler:
    """
    Creates N-way K-shot few-shot learning episodes from an allowed index pool.
    Support and query indices in one episode are always disjoint.
    """

    def __init__(
        self,
        dataset,
        device: str = "cpu",
        allowed_indices: Optional[List[int]] = None,
        forbidden_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ):

        self.dataset = dataset
        self.device = device

        n = len(dataset)
        if allowed_indices is None:
            pool = set(range(n))
        else:
            pool = set(int(i) for i in allowed_indices)

        if forbidden_indices:
            forbidden = set(int(i) for i in forbidden_indices)
            overlap = pool & forbidden
            if overlap:
                raise ValueError(
                    f"{len(overlap)} allowed indices are also forbidden "
                    "(leakage). Refusing to sample."
                )
            pool -= forbidden

        self.allowed_indices = sorted(pool)
        self.label_to_indices = {}

        for idx in self.allowed_indices:
            label = _graph_binary_label(dataset[idx])
            if label is None:
                continue
            self.label_to_indices.setdefault(label, []).append(idx)

        if verbose:
            print(
                f"Episode sampler: {len(self.allowed_indices)} allowed graphs, "
                f"{len(self.label_to_indices)} classes."
            )
            for label in sorted(self.label_to_indices):
                print(
                    f"  Class {label}: "
                    f"{len(self.label_to_indices[label])} samples"
                )

    def sample_episode(
        self,
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 5,
    ):

        needed = k_shot + q_query
        available_classes = [
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) >= needed
        ]

        if len(available_classes) < n_way:
            counts = {
                c: len(self.label_to_indices.get(c, []))
                for c in sorted(self.label_to_indices)
            }
            raise ValueError(
                "Not enough classes in the allowed pool to create an episode "
                f"(need {n_way} classes with at least {needed} molecules each; "
                f"found {counts})."
            )

        selected_classes = random.sample(available_classes, n_way)

        support_graphs = []
        support_labels = []
        query_graphs = []
        query_labels = []
        support_indices = []
        query_indices = []

        for new_label, original_label in enumerate(selected_classes):
            candidates = self.label_to_indices[original_label]
            sampled = random.sample(candidates, needed)
            support_idx = sampled[:k_shot]
            query_idx = sampled[k_shot:]

            if set(support_idx) & set(query_idx):
                raise RuntimeError("Support/query overlap inside a class.")

            support_indices.extend(support_idx)
            query_indices.extend(query_idx)

            for idx in support_idx:
                support_graphs.append(self.dataset[idx])
                support_labels.append(new_label)

            for idx in query_idx:
                query_graphs.append(self.dataset[idx])
                query_labels.append(new_label)

        overlap = sorted(set(support_indices) & set(query_indices))
        if overlap:
            raise RuntimeError(f"Support/query overlap in episode: {overlap}")

        support_batch = Batch.from_data_list(support_graphs).to(self.device)
        query_batch = Batch.from_data_list(query_graphs).to(self.device)
        support_labels = torch.tensor(
            support_labels, dtype=torch.long, device=self.device
        )
        query_labels = torch.tensor(
            query_labels, dtype=torch.long, device=self.device
        )

        meta = {
            "support_indices": support_indices,
            "query_indices": query_indices,
            "support_query_overlap": overlap,
            "selected_original_classes": selected_classes,
        }

        return (
            support_batch,
            support_labels,
            query_batch,
            query_labels,
            meta,
        )

    def sample_multiple_episodes(
        self,
        num_episodes: int,
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 5,
    ):

        episodes = []

        for _ in range(num_episodes):

            try:

                episode = self.sample_episode(
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                )

                episodes.append(episode)

            except ValueError:

                break

        return episodes

# ==========================================================
# FEATURE PROJECTOR
# ==========================================================

class FeatureProjector(torch.nn.Module):
    """
    Projects node features to a common feature dimension.
    Useful when combining datasets with different feature sizes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            in_dim,
            out_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.linear(x)

    @staticmethod
    def apply_to_dataset(
        dataset,
        projector,
        device: str = "cpu",
    ):

        projector = projector.to(device)
        projector.eval()

        with torch.no_grad():

            for graph in dataset:

                if graph.x is None:
                    continue

                graph.x = projector(
                    graph.x.to(device)
                ).cpu()

        if len(dataset) > 0 and dataset[0].x is not None:

            print(
                f"✓ Applied feature projection "
                f"(feature dimension = {dataset[0].x.size(1)})"
            )

        return dataset