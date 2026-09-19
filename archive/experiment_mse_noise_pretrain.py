"""
ARCHIVED DIAGNOSTIC SCRIPT — not GraphCL, not for paper results.

Original experiment.py: 3-layer GIN + MSE(projection, randn) on 300 BACE
molecules, then 2-way 5-shot on later BACE indices.

Produced outputs/final_results.txt (0.5180 ± 0.1590) and outputs/final_model.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Batch
from torch_geometric.datasets import MoleculeNet
from torch.utils.data import DataLoader
import random
import numpy as np
import time
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("QUARANTINED MSE-to-noise diagnostic (do not use as GraphCL)")
print("="*70)
print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: cpu")
print("="*70)

print("\nStep 1: Loading BACE dataset...")
dataset = MoleculeNet(root="data", name="BACE")
for d in dataset:
    if d.x is None:
        d.x = torch.ones((d.num_nodes, 9))
print(f"  Loaded {len(dataset)} molecules")

label_to_idx = {0: [], 1: []}
for i, d in enumerate(dataset):
    if d.y is not None and d.y.numel() == 1:
        label = int(d.y.item())
        if label in [0, 1]:
            label_to_idx[label].append(i)

print(f"  Class 0 (inactive): {len(label_to_idx[0])} molecules")
print(f"  Class 1 (active): {len(label_to_idx[1])} molecules")

class GIN(nn.Module):
    def __init__(self):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(9, 128), nn.ReLU(), nn.Linear(128, 128))
        mlp2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        mlp3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.conv3 = GINConv(mlp3)
        self.pool = global_add_pool
        self.proj = nn.Linear(128, 128)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        g = self.pool(x, batch)
        return self.proj(g), g

print("\nStep 2: Building model...")
model = GIN()
total_params = sum(p.numel() for p in model.parameters())
print(f"  Created 3-layer GIN (diagnostic only)")
print(f"  Hidden dimension: 128")
print(f"  Total parameters: {total_params:,}")

print("\nStep 3: MSE-to-noise (NOT GraphCL)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_idx = label_to_idx[0][:150] + label_to_idx[1][:150]
train_set = [dataset[i] for i in train_idx]
loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))

print(f"  Training set: {len(train_set)} molecules")
print(f"  Batch size: 32")
print()

losses = []
for epoch in range(1, 21):
    t0 = time.time()
    total = 0
    model.train()
    for batch in loader:
        z, g = model(batch)
        loss = F.mse_loss(z, torch.randn_like(z))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    avg_loss = total / len(loader)
    losses.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/20 | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

print(f"\n  Final loss: {losses[-1]:.4f}")

print("\nStep 4: Few-shot evaluation...")
print("  Configuration: 2-way 5-shot")
print("  Episodes: 100")
print()

test_idx_0 = label_to_idx[0][150:400]
test_idx_1 = label_to_idx[1][150:400]

accuracies = []
model.eval()

for ep in range(100):
    if len(test_idx_0) < 10 or len(test_idx_1) < 10:
        break

    sampled_0 = random.sample(test_idx_0, 10)
    sampled_1 = random.sample(test_idx_1, 10)

    support = [dataset[i] for i in sampled_0[:5]] + [dataset[i] for i in sampled_1[:5]]
    query = [dataset[i] for i in sampled_0[5:10]] + [dataset[i] for i in sampled_1[5:10]]

    support_batch = Batch.from_data_list(support)
    query_batch = Batch.from_data_list(query)

    with torch.no_grad():
        _, s_emb = model(support_batch)
        _, q_emb = model(query_batch)

    proto0 = s_emb[:5].mean(0)
    proto1 = s_emb[5:10].mean(0)
    protos = torch.stack([proto0, proto1])

    dists = torch.cdist(q_emb, protos)
    preds = dists.argmin(1)
    labels = torch.tensor([0]*5 + [1]*5)
    acc = (preds == labels).float().mean().item()
    accuracies.append(acc)

    if (ep + 1) % 25 == 0:
        print(f"  Episode {ep+1:3d}/100 | Recent avg: {np.mean(accuracies[-25:]):.4f}")

print(f"\n{'='*70}")
print("DIAGNOSTIC RESULTS (not for the paper)")
print(f"{'='*70}")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
min_acc = np.min(accuracies)
max_acc = np.max(accuracies)
improvement = (mean_acc - 0.5) / 0.5 * 100

print(f"\nFew-Shot Performance (2-way 5-shot):")
print(f"  Mean Accuracy:    {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  Min Accuracy:     {min_acc:.4f}")
print(f"  Max Accuracy:     {max_acc:.4f}")
print(f"  Episodes:         {len(accuracies)}")
print(f"\nBaseline Comparison:")
print(f"  Random baseline:  0.5000")
print(f"  Improvement:      {improvement:+.1f}%")

os.makedirs('outputs', exist_ok=True)

with open('outputs/final_results.txt', 'w') as f:
    f.write("DIAGNOSTIC ONLY — MSE-to-noise, not GraphCL\n")
    f.write("GNN + Few-Shot Learning Results\n")
    f.write("="*50 + "\n\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: BACE (1513 molecules)\n")
    f.write(f"Model: 3-layer GIN (128 hidden, {total_params:,} params)\n")
    f.write(f"Training: 20 epochs, 300 molecules\n")
    f.write(f"Evaluation: 2-way 5-shot, {len(accuracies)} episodes\n\n")
    f.write(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
    f.write(f"Min/Max: {min_acc:.4f} / {max_acc:.4f}\n")
    f.write(f"Improvement over random: {improvement:+.1f}%\n")

torch.save(model.state_dict(), 'outputs/final_model.pt')

print(f"\n{'='*70}")
print("Files saved:")
print(f"  - outputs/final_results.txt")
print(f"  - outputs/final_model.pt")
print(f"{'='*70}")
print("\nDiagnostic complete.")
