"""
training.py
Training procedures: GraphCL pretraining, few-shot ProtoNet evaluation,
and supervised baseline training
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, accuracy_score


# ======================================================
# NT-Xent Loss (GraphCL)
# ======================================================
class NTXentLoss(nn.Module):
    """InfoNCE / NT-Xent over two augmented views of a batch."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) projection-head outputs from two augmentations.
        For index i, the positive is the other view of the same graph.
        """
        if z1.size(0) != z2.size(0):
            raise ValueError("NT-Xent requires paired views with equal batch size.")

        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        reps = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(reps, reps.T) / self.temperature

        logits_mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
        sim = sim.masked_fill(logits_mask, float("-inf"))

        positives = torch.cat(
            [
                torch.arange(B, device=z1.device) + B,
                torch.arange(B, device=z1.device),
            ]
        )
        return F.cross_entropy(sim, positives)


# ======================================================
# GraphCL Trainer
# ======================================================
class GraphCLTrainer:
    def __init__(self, encoder, augmentation_fn, device="cpu", temperature=0.2):
        self.encoder = encoder.to(device)
        self.augmentation_fn = augmentation_fn
        self.device = device
        self.criterion = NTXentLoss(temperature=temperature)

    def train_epoch(self, dataloader, optimizer):
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            graphs = batch.to_data_list()
            if len(graphs) < 2:
                continue

            aug1 = Batch.from_data_list(
                [self.augmentation_fn(g) for g in graphs]
            ).to(self.device)
            aug2 = Batch.from_data_list(
                [self.augmentation_fn(g) for g in graphs]
            ).to(self.device)

            z1, _, _ = self.encoder(aug1)
            z2, _, _ = self.encoder(aug2)
            loss = self.criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_encoder_checkpoint(self, save_path, extra=None):
        extra = extra or {}
        self.encoder.save_checkpoint(save_path, extra=extra)

    def pretrain(
        self,
        dataset,
        epochs=50,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-5,
        save_path=None,
        save_interval=10,
        history_csv_path=None,
        extra_checkpoint_meta=None,
    ):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: Batch.from_data_list(x),
        )

        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        history_rows = []

        print("=" * 60)
        print("Starting GraphCL Pretraining (NT-Xent)")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            start = time.time()
            loss = self.train_epoch(loader, optimizer)
            scheduler.step(loss)
            elapsed = time.time() - start
            lr_now = optimizer.param_groups[0]["lr"]

            history_rows.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "time_sec": float(elapsed),
                    "lr": float(lr_now),
                }
            )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if save_path and (epoch % save_interval == 0 or epoch == epochs):
                meta = {
                    "pretrain": {
                        "method": "GraphCL",
                        "loss": "NT-Xent",
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    }
                }
                if extra_checkpoint_meta:
                    meta.update(extra_checkpoint_meta)
                self.save_encoder_checkpoint(save_path, extra=meta)

        if history_csv_path:
            history_csv_path = Path(history_csv_path)
            history_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with history_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "loss", "time_sec", "lr"]
                )
                writer.writeheader()
                writer.writerows(history_rows)

        print("Pretraining finished.")
        print("=" * 60)

        return {"loss": [row["loss"] for row in history_rows], "rows": history_rows}


# ======================================================
# ProtoNet Evaluator (Few-shot)
# ======================================================
class ProtoNetEvaluator:
    def __init__(self, encoder, protonet, device="cpu"):
        self.encoder = encoder.to(device)
        self.protonet = protonet
        self.device = device

    def evaluate(
        self,
        episode_sampler,
        num_episodes=100,
        n_way=2,
        k_shot=5,
        q_query=5,
    ):
        accs, losses = [], []
        episode_rows = []

        self.encoder.eval()

        for ep in range(num_episodes):
            (
                support_batch,
                support_labels,
                query_batch,
                query_labels,
                episode_meta,
            ) = episode_sampler.sample_episode(n_way, k_shot, q_query)

            acc, loss = self._eval_episode(
                support_batch, support_labels, query_batch, query_labels, n_way
            )
            accs.append(acc)
            losses.append(loss)
            episode_rows.append(
                {
                    "episode": ep,
                    "accuracy": float(acc),
                    "loss": float(loss),
                    "n_way": n_way,
                    "k_shot": k_shot,
                    "q_query": q_query,
                    "support_indices": episode_meta["support_indices"],
                    "query_indices": episode_meta["query_indices"],
                    "support_query_overlap": episode_meta["support_query_overlap"],
                }
            )

        return {
            "mean_acc": float(np.mean(accs)) if accs else float("nan"),
            "std_acc": float(np.std(accs)) if accs else float("nan"),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "num_episodes": len(accs),
            "episodes": episode_rows,
        }

    def _eval_episode(
        self,
        support_batch,
        support_labels,
        query_batch,
        query_labels,
        n_way,
    ):
        support_batch = support_batch.to(self.device)
        query_batch = query_batch.to(self.device)
        support_labels = support_labels.to(self.device)
        query_labels = query_labels.to(self.device)

        self.encoder.eval()
        with torch.no_grad():
            _, support_emb, _ = self.encoder(support_batch)
            _, query_emb, _ = self.encoder(query_batch)
            logits, _ = self.protonet(
                support_emb, support_labels, query_emb, n_way
            )
            loss = F.cross_entropy(logits, query_labels)
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()
        return acc, float(loss.item())

    def evaluate_fixed_episodes(
        self,
        dataset,
        episode_specs,
        n_way=2,
        k_shot=5,
        q_query=5,
    ):
        """Evaluate frozen ProtoNet on pre-recorded support/query indices."""
        from torch_geometric.data import Batch

        accs, losses, episode_rows = [], [], []
        for spec in episode_specs:
            support_indices = list(spec["support_indices"])
            query_indices = list(spec["query_indices"])
            overlap = sorted(set(support_indices) & set(query_indices))
            if overlap:
                raise RuntimeError(f"Support/query overlap: {overlap}")

            n_support = len(support_indices)
            n_query = len(query_indices)
            support_labels = torch.tensor(
                [i // k_shot for i in range(n_support)], dtype=torch.long
            )
            query_labels = torch.tensor(
                [i // q_query for i in range(n_query)], dtype=torch.long
            )
            support_batch = Batch.from_data_list(
                [dataset[i] for i in support_indices]
            )
            query_batch = Batch.from_data_list(
                [dataset[i] for i in query_indices]
            )
            acc, loss = self._eval_episode(
                support_batch, support_labels, query_batch, query_labels, n_way
            )
            accs.append(acc)
            losses.append(loss)
            episode_rows.append(
                {
                    "episode": int(spec.get("episode", len(episode_rows))),
                    "accuracy": float(acc),
                    "loss": float(loss),
                    "n_way": n_way,
                    "k_shot": k_shot,
                    "q_query": q_query,
                    "support_indices": support_indices,
                    "query_indices": query_indices,
                    "support_query_overlap": overlap,
                }
            )

        return {
            "mean_acc": float(np.mean(accs)) if accs else float("nan"),
            "std_acc": float(np.std(accs)) if accs else float("nan"),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "num_episodes": len(accs),
            "episodes": episode_rows,
        }


# ======================================================
# Supervised Trainer (Baseline)
# ======================================================
class SupervisedTrainer:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            labels = batch.y.view(-1).long()

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def evaluate(self, loader):
        self.model.eval()
        preds, labels, probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                p = F.softmax(logits, dim=1)

                preds.extend(p.argmax(dim=1).cpu().numpy())
                probs.extend(p[:, 1].cpu().numpy())
                labels.extend(batch.y.view(-1).cpu().numpy())

        return {
            "accuracy": accuracy_score(labels, preds),
            "auc": roc_auc_score(labels, probs),
        }
