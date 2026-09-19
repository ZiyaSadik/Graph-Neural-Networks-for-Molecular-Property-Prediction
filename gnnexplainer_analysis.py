"""
Five-molecule GNNExplainer analysis for prototype-distance ProtoNet.

Uses the canonical GraphCL checkpoint and PrototypeDistanceLogits.
Does not retrain GraphCL or change core few-shot results.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from ogb.graphproppred import PygGraphPropPredDataset
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer, fidelity

from prototype_explain import (
    load_bace_mapping,
    molecule_record,
    sample_original_label_episode,
    set_seed,
)
from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.explainability import PrototypeDistanceLogits
from src.models import GINEncoder, PrototypicalNetwork

allow_pyg_torch_load()

CHECKPOINT = Path("outputs/core_pipeline/graphcl_checkpoint.pt")
OUT_DIR = Path("outputs/core_pipeline/xai/gnnexplainer")
EPISODE_SEED = 7
K_SHOT = 5
Q_QUERY = 5
EXPLAIN_EPOCHS = 80
HARD_THRESHOLD = 0.5
CLASS_NAME = {0: "inactive", 1: "active"}


def soft_sparsity(mask: torch.Tensor) -> float:
    """1 - mean(mask). Masks are in (0, 1); higher means less mass kept."""
    t = mask.detach().float().reshape(-1)
    return float(1.0 - t.mean().item())


def hard_sparsity(mask: torch.Tensor, threshold: float) -> float:
    """Fraction of elements with mask < threshold (not in the hard subgraph)."""
    t = mask.detach().float().reshape(-1)
    return float((t < threshold).float().mean().item())


def pyg_version() -> str:
    try:
        return importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def binary_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return (mask.detach().float() >= threshold).to(mask.dtype)


def graphfram_ex_from_preds(original_pred: int, expl_pred: int, complement_pred: int) -> tuple[float, float]:
    """Model-explanation GraphFramEx: fid+ / fid− as 0/1 prediction-change flags."""
    fid_plus = 0.0 if complement_pred == original_pred else 1.0
    fid_minus = 0.0 if expl_pred == original_pred else 1.0
    return fid_plus, fid_minus


def masked_logits(
    model: PrototypeDistanceLogits,
    explainer: Explainer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_mask: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
    batch: torch.Tensor,
) -> torch.Tensor:
    out = explainer.get_masked_prediction(
        x, edge_index, node_mask, edge_mask, batch=batch
    )
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


def mask_stats(mask: torch.Tensor) -> dict:
    t = mask.detach().float().reshape(-1)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def select_query_positions(correct_flags: list[bool]) -> list[int]:
    """Deterministic mix: first 3 incorrect and first 2 correct by episode order."""
    incorrect = [i for i, ok in enumerate(correct_flags) if not ok]
    correct = [i for i, ok in enumerate(correct_flags) if ok]
    if len(incorrect) < 1 or len(correct) < 1:
        raise RuntimeError(
            f"Need both correct and incorrect queries; got correct={correct}, incorrect={incorrect}"
        )
    chosen = incorrect[:3] + correct[:2]
    if len(chosen) < 5:
        rest = [i for i in range(len(correct_flags)) if i not in chosen]
        chosen = chosen + rest[: 5 - len(chosen)]
    return chosen[:5]


def draw_explained_molecule(
    smiles: str,
    node_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    out_path: Path,
    title: str,
) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    node = node_mask.detach().float().view(-1).cpu()
    if mol is None or mol.GetNumAtoms() != int(node.numel()):
        return False

    node_np = node.numpy()
    nmin, nmax = float(node_np.min()), float(node_np.max())
    node_n = (node_np - nmin) / (nmax - nmin + 1e-8)
    cmap = cm.YlOrRd
    atom_colors = {i: tuple(float(c) for c in cmap(float(s))[:3]) for i, s in enumerate(node_n)}

    e = edge_index.cpu()
    em = edge_mask.detach().float().view(-1).cpu().numpy()
    emin, emax = float(em.min()), float(em.max())
    em_n = (em - emin) / (emax - emin + 1e-8)
    pair_scores: dict[tuple[int, int], list[float]] = {}
    for k in range(e.size(1)):
        a, b = int(e[0, k]), int(e[1, k])
        pair_scores.setdefault(tuple(sorted((a, b))), []).append(float(em_n[k]))

    highlight_bonds = []
    bond_colors = {}
    for bond in mol.GetBonds():
        key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if key in pair_scores:
            highlight_bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = tuple(
                float(c) for c in cmap(float(np.mean(pair_scores[key])))[:3]
            )

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 620)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    mol_img = Image.open(BytesIO(drawer.GetDrawingText()))

    fig = plt.figure(figsize=(9.2, 7.2), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[6.2, 0.35], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(mol_img)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    cax = fig.add_subplot(gs[1])
    norm = plt.Normalize(nmin, nmax)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label("Atom importance (GNNExplainer node mask)", fontsize=9)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> int:
    pyg_ver = pyg_version()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    set_seed(EPISODE_SEED)

    encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
    encoder = encoder.to(device)
    encoder.eval()
    cfg = encoder.get_config()
    if cfg["num_layers"] != 5:
        raise RuntimeError(cfg)

    dataset = PygGraphPropPredDataset(name="ogbg-molbace", root="dataset")
    protocol = get_bace_held_out_protocol(dataset)
    mapping = load_bace_mapping(dataset)
    episode = sample_original_label_episode(
        dataset, protocol, K_SHOT, Q_QUERY, EPISODE_SEED
    )

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    support_batch = episode["support_batch"].to(device)
    support_labels = episode["support_labels"].to(device)
    query_batch = episode["query_batch"].to(device)
    query_labels = episode["query_labels"].to(device)
    with torch.no_grad():
        _, support_emb, _ = encoder(support_batch)
        _, query_emb, _ = encoder(query_batch)
        logits, prototypes = protonet(
            support_emb, support_labels, query_emb, n_way=2
        )
        preds = logits.argmax(dim=1)
        distances = torch.cdist(query_emb, prototypes, p=2)

    correct_flags = [
        bool(preds[i].item() == query_labels[i].item())
        for i in range(preds.numel())
    ]
    positions = select_query_positions(correct_flags)
    model = PrototypeDistanceLogits(encoder, prototypes.to(device))
    model.eval()

    rows = []
    panel_paths = []
    for panel_i, pos in enumerate(positions):
        set_seed(EPISODE_SEED * 100 + pos)
        q_idx = int(episode["query_indices"][pos])
        rec = molecule_record(mapping, q_idx)
        query = dataset[q_idx]
        x = query.x.float().to(device)
        edge_index = query.edge_index.to(device)
        n_nodes = int(x.size(0))
        n_edges = int(edge_index.size(1))
        pred = int(preds[pos].item())
        true = int(query_labels[pos].item())
        d0 = float(distances[pos, 0].item())
        d1 = float(distances[pos, 1].item())

        with torch.no_grad():
            wrap_pred = int(model(x, edge_index).argmax(dim=1).item())
        if wrap_pred != pred:
            raise RuntimeError(f"{rec['molecule_id']}: wrapper {wrap_pred} != ProtoNet {pred}")

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
        explanation = explainer(x, edge_index, index=0, batch=batch)
        node_mask = explanation.get("node_mask")
        edge_mask = explanation.get("edge_mask")
        if node_mask is None or edge_mask is None:
            raise RuntimeError("GNNExplainer did not return both masks")

        fid_plus, fid_minus = fidelity(explainer, explanation)
        node_hard = binary_mask(node_mask, HARD_THRESHOLD)
        edge_hard = binary_mask(edge_mask, HARD_THRESHOLD)
        with torch.no_grad():
            orig_logits = model(x, edge_index, batch=batch)
            expl_logits_hard = masked_logits(
                model, explainer, x, edge_index, node_hard, edge_hard, batch
            )
            comp_logits_hard = masked_logits(
                model,
                explainer,
                x,
                edge_index,
                1.0 - node_hard,
                1.0 - edge_hard,
                batch,
            )
            expl_pred_hard = int(expl_logits_hard.argmax(dim=1).item())
            comp_pred_hard = int(comp_logits_hard.argmax(dim=1).item())
            orig_prob = torch.softmax(orig_logits, dim=1)[0, pred].item()
            expl_prob_soft = torch.softmax(
                masked_logits(model, explainer, x, edge_index, node_mask, edge_mask, batch),
                dim=1,
            )[0, pred].item()
            comp_prob_soft = torch.softmax(
                masked_logits(
                    model,
                    explainer,
                    x,
                    edge_index,
                    1.0 - node_mask,
                    1.0 - edge_mask,
                    batch,
                ),
                dim=1,
            )[0, pred].item()
        fid_plus_hard, fid_minus_hard = graphfram_ex_from_preds(
            pred, expl_pred_hard, comp_pred_hard
        )
        node_sp_soft = soft_sparsity(node_mask)
        edge_sp_soft = soft_sparsity(edge_mask)
        node_sp_hard = hard_sparsity(node_mask, HARD_THRESHOLD)
        edge_sp_hard = hard_sparsity(edge_mask, HARD_THRESHOLD)

        fig_path = OUT_DIR / f"explanation_{panel_i}_{rec['molecule_id']}.png"
        title = (
            f"{rec['molecule_id']}   true={CLASS_NAME[true]} ({true})   "
            f"pred={CLASS_NAME[pred]} ({pred})   "
            f"{'correct' if pred == true else 'incorrect'}\n"
            f"d0={d0:.2f}   d1={d1:.2f}   "
            f"fid+={fid_plus:.3f}   fid−={fid_minus:.3f}   "
            f"node sparsity={node_sp_soft:.3f}"
        )
        drawn = draw_explained_molecule(
            rec["smiles"], node_mask, edge_index, edge_mask, fig_path, title
        )
        if not drawn:
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)
            ax.bar(np.arange(n_nodes), node_mask.detach().cpu().view(-1).numpy())
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("atom index")
            ax.set_ylabel("node mask")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        row = {
            "molecule_id": rec["molecule_id"],
            "dataset_index": q_idx,
            "episode_query_position": pos,
            "true_label": true,
            "predicted_label": pred,
            "correct": int(pred == true),
            "distance_to_class_0_prototype": d0,
            "distance_to_class_1_prototype": d1,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "node_sparsity_soft": node_sp_soft,
            "edge_sparsity_soft": edge_sp_soft,
            "node_sparsity_hard_0_5": node_sp_hard,
            "edge_sparsity_hard_0_5": edge_sp_hard,
            "fidelity_plus": float(fid_plus),
            "fidelity_minus": float(fid_minus),
            "fidelity_plus_hard_0_5": float(fid_plus_hard),
            "fidelity_minus_hard_0_5": float(fid_minus_hard),
            "orig_pred_softmax": float(orig_prob),
            "expl_pred_softmax_softmask": float(expl_prob_soft),
            "comp_pred_softmax_softmask": float(comp_prob_soft),
            "softmax_drop_on_explanation": float(orig_prob - expl_prob_soft),
            "softmax_drop_on_complement": float(orig_prob - comp_prob_soft),
            "node_mask_mean": mask_stats(node_mask)["mean"],
            "edge_mask_mean": mask_stats(edge_mask)["mean"],
            "figure": str(fig_path),
            "node_importance": [float(v) for v in node_mask.view(-1).detach().cpu()],
            "edge_importance": [float(v) for v in edge_mask.view(-1).detach().cpu()],
        }
        rows.append(row)
        panel_paths.append(fig_path)
        print(
            f"{rec['molecule_id']}: true={true} pred={pred} "
            f"fid+={fid_plus:.3f} fid-={fid_minus:.3f} "
            f"node_sparsity={node_sp_soft:.3f}"
        )

    csv_fields = [
        "molecule_id",
        "dataset_index",
        "true_label",
        "predicted_label",
        "correct",
        "distance_to_class_0_prototype",
        "distance_to_class_1_prototype",
        "n_nodes",
        "n_edges",
        "node_sparsity_soft",
        "edge_sparsity_soft",
        "node_sparsity_hard_0_5",
        "edge_sparsity_hard_0_5",
        "fidelity_plus",
        "fidelity_minus",
        "fidelity_plus_hard_0_5",
        "fidelity_minus_hard_0_5",
        "orig_pred_softmax",
        "expl_pred_softmax_softmask",
        "comp_pred_softmax_softmask",
        "softmax_drop_on_explanation",
        "softmax_drop_on_complement",
        "node_mask_mean",
        "edge_mask_mean",
        "figure",
    ]
    with (OUT_DIR / "gnnexplainer_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "encoder_config": cfg,
        "pyg_version": pyg_ver,
        "api": "torch_geometric.explain.Explainer + GNNExplainer",
        "objective": (
            "PrototypeDistanceLogits: logit_c = -||z_query - prototype_c||_2, "
            "prototypes fixed from episode support. Not projection head."
        ),
        "episode_seed": EPISODE_SEED,
        "support_indices": episode["support_indices"],
        "query_indices": episode["query_indices"],
        "selected_query_positions": positions,
        "explain_epochs": EXPLAIN_EPOCHS,
        "sparsity_definitions": {
            "node_sparsity_soft": "1 - mean(node_mask), mask in (0,1)",
            "edge_sparsity_soft": "1 - mean(edge_mask), mask in (0,1)",
            "node_sparsity_hard_0_5": f"fraction of nodes with mask < {HARD_THRESHOLD}",
            "edge_sparsity_hard_0_5": f"fraction of edges with mask < {HARD_THRESHOLD}",
        },
        "fidelity_definition": {
            "source": "PyG torch_geometric.explain.fidelity, GraphFramEx model explanations",
            "masking": (
                "Soft: PyG multiplies node features by the mask and sets MessagePassing "
                "_edge_mask. Hard 0.5: same GraphFramEx formula after binarizing masks."
            ),
            "fid_plus": (
                "1 - 1[pred(complement subgraph) == original pred]. "
                "High means removing the explanation subgraph changes the prototype-distance prediction."
            ),
            "fid_minus": (
                "1 - 1[pred(explanation subgraph) == original pred]. "
                "Low means the explanation subgraph alone is sufficient to keep the prediction."
            ),
            "softmax_companion": (
                "Change in softmax mass of the original predicted class after soft masking. "
                "Used because 0/1 GraphFramEx can saturate when masks are diffuse."
            ),
            "not_causal": True,
        },
        "n_molecules": len(rows),
        "n_correct": int(sum(r["correct"] for r in rows)),
        "n_incorrect": int(sum(1 - r["correct"] for r in rows)),
        "mean_fidelity_plus": float(np.mean([r["fidelity_plus"] for r in rows])),
        "mean_fidelity_minus": float(np.mean([r["fidelity_minus"] for r in rows])),
        "mean_fidelity_plus_hard_0_5": float(np.mean([r["fidelity_plus_hard_0_5"] for r in rows])),
        "mean_fidelity_minus_hard_0_5": float(np.mean([r["fidelity_minus_hard_0_5"] for r in rows])),
        "mean_softmax_drop_on_explanation": float(
            np.mean([r["softmax_drop_on_explanation"] for r in rows])
        ),
        "mean_softmax_drop_on_complement": float(
            np.mean([r["softmax_drop_on_complement"] for r in rows])
        ),
        "mean_node_sparsity_soft": float(np.mean([r["node_sparsity_soft"] for r in rows])),
        "mean_edge_sparsity_soft": float(np.mean([r["edge_sparsity_soft"] for r in rows])),
        "molecules": [{k: r[k] for k in csv_fields} for r in rows],
        "per_molecule_masks": [
            {
                "molecule_id": r["molecule_id"],
                "node_importance": r["node_importance"],
                "edge_importance": r["edge_importance"],
            }
            for r in rows
        ],
        "caveat": (
            "n=5 molecules from one 2-way 5-shot episode. Explanations are not causal "
            "and do not generalize beyond this sample."
        ),
    }
    (OUT_DIR / "gnnexplainer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), dpi=200)
    for ax, path, row in zip(axes, panel_paths, rows):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{row['molecule_id']}\n"
            f"true {row['true_label']} pred {row['predicted_label']}\n"
            f"fid+ {row['fidelity_plus']:.2f}  sp {row['node_sparsity_soft']:.2f}",
            fontsize=8,
        )
    fig.suptitle(
        "GNNExplainer on prototype-distance ProtoNet (5 BACE test queries, episode seed 7)",
        fontsize=12,
    )
    fig.tight_layout()
    summary_png = OUT_DIR / "gnnexplainer_summary.png"
    fig.savefig(summary_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    table = [
        "| ID | true | pred | correct | d0 | d1 | node sp. | edge sp. | fid+ | fid− |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r['molecule_id']} | {r['true_label']} | {r['predicted_label']} | "
            f"{'yes' if r['correct'] else 'no'} | "
            f"{r['distance_to_class_0_prototype']:.2f} | "
            f"{r['distance_to_class_1_prototype']:.2f} | "
            f"{r['node_sparsity_soft']:.3f} | {r['edge_sparsity_soft']:.3f} | "
            f"{r['fidelity_plus']:.3f} | {r['fidelity_minus']:.3f} |"
        )
    md = f"""# GNNExplainer analysis (5 molecules)

Canonical frozen GraphCL `GINEncoder` from `{CHECKPOINT.as_posix()}`.  
Objective: `PrototypeDistanceLogits`, `logit_c = -||z_query - p_c||_2`, prototypes from the **same** 2-way 5-shot support set (OGB BACE scaffold test, episode seed **{EPISODE_SEED}**). Not the projection head. Not a supervised classifier.

This is **n = 5**. It is not causal and does not generalize.

## Selection

Queries come from one deterministic episode (seed {EPISODE_SEED}). Positions {positions}: first three incorrect then first two correct in episode order ({summary['n_incorrect']} incorrect, {summary['n_correct']} correct).

Support indices: `{episode['support_indices']}`

## Sparsity

- Soft node/edge sparsity = `1 - mean(mask)` with GNNExplainer masks in (0, 1). Larger means less average mass.
- Hard sparsity = fraction of atoms/bonds with mask `< {HARD_THRESHOLD}`.

## Fidelity (GraphFramEx, PyG `fidelity`, model explanations)

- **fid+**: `1 - 1[pred(graph with explanation removed) == original pred]`. 1 means the complement flips the prototype-distance class.
- **fid−**: `1 - 1[pred(explanation subgraph only) == original pred]`. 0 means the subgraph alone keeps the original class.

A useful explanation typically has high fid+ and low fid−. These are prediction-change checks, not causal effects.

## Results

{chr(10).join(table)}

Mean fid+: {summary['mean_fidelity_plus']:.3f}  
Mean fid−: {summary['mean_fidelity_minus']:.3f}  
Mean node soft sparsity: {summary['mean_node_sparsity_soft']:.3f}  
Mean edge soft sparsity: {summary['mean_edge_sparsity_soft']:.3f}

GNNExplainer epochs: {EXPLAIN_EPOCHS}. PyG {pyg_ver}.
"""
    (OUT_DIR / "GNNEXPLAINER_SUMMARY.md").write_text(md, encoding="utf-8")
    print("=" * 50)
    print(f"Wrote {OUT_DIR}  n={len(rows)}  mean fid+={summary['mean_fidelity_plus']:.3f}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
