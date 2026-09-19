"""
Reproducible prototype-based explanations for the canonical GraphCL GINEncoder.

Loads outputs/core_pipeline/graphcl_checkpoint.pt only.
Does not use the old 3-layer GIN or outputs/final_model.pt.
Does not run GNNExplainer.

Default: one 2-way 5-shot episode on OGB BACE scaffold test, seed 7.
Class 0 / class 1 prototypes use original BACE labels (inactive / active).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

from src.data_utils import (
    EpisodeSampler,
    allow_pyg_torch_load,
    get_bace_held_out_protocol,
)
from src.explainability import PrototypeExplainer
from src.models import GINEncoder, PrototypicalNetwork

allow_pyg_torch_load()

CHECKPOINT = Path("outputs/core_pipeline/graphcl_checkpoint.pt")
OUT_DIR = Path("outputs/core_pipeline/xai")
BACE_ROOT = "dataset"
CLASS_NAME = {0: "inactive (class 0)", 1: "active (class 1)"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bace_mapping(dataset) -> pd.DataFrame:
    root = Path(dataset.root)
    candidates = [
        root / "mapping" / "mol.csv.gz",
        root / "mapping" / "mol.csv",
        root.parent / "ogbg_molbace" / "mapping" / "mol.csv.gz",
        Path("dataset") / "ogbg_molbace" / "mapping" / "mol.csv.gz",
    ]
    mapping = next((p for p in candidates if p.exists()), None)
    if mapping is None:
        raise FileNotFoundError(
            f"Could not find BACE mapping CSV. Tried: {candidates}"
        )
    df = pd.read_csv(mapping)
    if len(df) != len(dataset):
        raise RuntimeError(
            f"Mapping length {len(df)} != dataset length {len(dataset)}"
        )
    return df


def molecule_record(mapping: pd.DataFrame, idx: int) -> dict:
    row = mapping.iloc[idx]
    smiles_col = "smiles" if "smiles" in mapping.columns else None
    if smiles_col is None:
        for c in mapping.columns:
            if c.lower() == "smiles":
                smiles_col = c
                break
    mol_id = None
    for c in ("mol_id", "molecule_id", "id"):
        if c in mapping.columns:
            mol_id = str(row[c])
            break
    if mol_id is None:
        mol_id = f"ogbg-molbace/{idx}"
    smiles = None if smiles_col is None else str(row[smiles_col])
    return {"molecule_id": mol_id, "smiles": smiles, "dataset_index": int(idx)}


def sample_original_label_episode(dataset, protocol, k_shot, q_query, seed):
    """2-way episode with prototype 0 = BACE label 0, prototype 1 = BACE label 1."""
    set_seed(seed)
    sampler = EpisodeSampler(
        dataset,
        device="cpu",
        allowed_indices=protocol["fewshot_pool"],
        forbidden_indices=protocol["train_reserved"] + protocol["valid_reserved"],
        verbose=True,
    )
    needed = k_shot + q_query
    support_indices = []
    query_indices = []
    support_labels = []
    query_labels = []
    for original_label in (0, 1):
        candidates = sampler.label_to_indices.get(original_label, [])
        if len(candidates) < needed:
            raise RuntimeError(
                f"Not enough test molecules for original class {original_label}"
            )
        sampled = random.sample(candidates, needed)
        supp = sampled[:k_shot]
        qry = sampled[k_shot:]
        support_indices.extend(supp)
        query_indices.extend(qry)
        support_labels.extend([original_label] * k_shot)
        query_labels.extend([original_label] * q_query)

    overlap = sorted(set(support_indices) & set(query_indices))
    if overlap:
        raise RuntimeError(f"Support/query overlap: {overlap}")

    support_batch = Batch.from_data_list([dataset[i] for i in support_indices])
    query_batch = Batch.from_data_list([dataset[i] for i in query_indices])
    return {
        "support_indices": support_indices,
        "query_indices": query_indices,
        "support_labels": torch.tensor(support_labels, dtype=torch.long),
        "query_labels": torch.tensor(query_labels, dtype=torch.long),
        "support_batch": support_batch,
        "query_batch": query_batch,
        "support_query_overlap": overlap,
    }


def render_query_figure(row: dict, out_path: Path) -> None:
    q_mol = Chem.MolFromSmiles(row["query_smiles"] or "")
    s_mol = Chem.MolFromSmiles(row["most_similar_support_smiles"] or "")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mol, title in (
        (axes[0], q_mol, "Query"),
        (axes[1], s_mol, "Closest support (cosine)"),
    ):
        ax.axis("off")
        if mol is None:
            ax.text(0.5, 0.5, "Could not parse SMILES", ha="center", va="center")
        else:
            img = Draw.MolToImage(mol, size=(400, 320))
            ax.imshow(img)
        ax.set_title(title, fontsize=11)

    pred = CLASS_NAME.get(row["predicted_class"], str(row["predicted_class"]))
    true = CLASS_NAME.get(row["ground_truth_label"], str(row["ground_truth_label"]))
    fig.suptitle(
        f"Query {row['query_molecule_id']}  |  predicted: {pred}  |  true: {true}\n"
        f"d0={row['distance_to_class_0_prototype']:.3f}  "
        f"d1={row['distance_to_class_1_prototype']:.3f}  "
        f"gap-confidence={row['distance_gap_confidence']:.3f}\n"
        f"closest support: {row['most_similar_support_id']}  "
        f"cosine={row['cosine_similarity_to_nearest_support']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--bace-root", type=str, default=BACE_ROOT)
    p.add_argument("--max-figures", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.name == "final_model.pt":
        raise RuntimeError("Refusing to load the old 3-layer diagnostic checkpoint.")
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GINEncoder.from_checkpoint(str(ckpt_path), map_location=device)
    encoder = encoder.to(device)
    encoder.eval()
    cfg = encoder.get_config()
    if cfg.get("num_layers") != 5:
        raise RuntimeError(f"Expected 5-layer GINEncoder, got {cfg}")
    print(f"Loaded canonical encoder from {ckpt_path}: {cfg}")

    dataset = PygGraphPropPredDataset(name="ogbg-molbace", root=args.bace_root)
    protocol = get_bace_held_out_protocol(dataset)
    mapping = load_bace_mapping(dataset)
    episode = sample_original_label_episode(
        dataset, protocol, args.k_shot, args.q_query, args.seed
    )

    q_recs = [molecule_record(mapping, i) for i in episode["query_indices"]]
    s_recs = [molecule_record(mapping, i) for i in episode["support_indices"]]

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    explainer = PrototypeExplainer(encoder, protonet, device=device)
    explained = explainer.explain_episode(
        episode["support_batch"],
        episode["support_labels"],
        episode["query_batch"],
        episode["query_labels"],
        n_way=2,
        query_indices=episode["query_indices"],
        support_indices=episode["support_indices"],
        query_ids=[r["molecule_id"] for r in q_recs],
        support_ids=[r["molecule_id"] for r in s_recs],
        query_smiles=[r["smiles"] for r in q_recs],
        support_smiles=[r["smiles"] for r in s_recs],
    )

    csv_fields = [
        "query_molecule_id",
        "query_index",
        "query_smiles",
        "ground_truth_label",
        "predicted_class",
        "correct",
        "distance_to_class_0_prototype",
        "distance_to_class_1_prototype",
        "distance_gap_confidence",
        "most_similar_support_id",
        "most_similar_support_index",
        "cosine_similarity_to_nearest_support",
        "most_similar_support_smiles",
        "figure_path",
    ]
    figure_paths = []
    for i, row in enumerate(explained["explanations"]):
        fig_path = ""
        if i < args.max_figures:
            fig_path = str(fig_dir / f"query_{i}_{row['query_index']}.png")
            render_query_figure(row, Path(fig_path))
        figure_paths.append(fig_path)
        row["figure_path"] = fig_path

    csv_path = out_dir / "prototype_explanations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(explained["explanations"])

    payload = {
        "checkpoint": str(ckpt_path),
        "encoder_config": cfg,
        "uses_old_3_layer_model": False,
        "uses_final_model_pt": False,
        "gnnexplainer": False,
        "seed": args.seed,
        "protocol": {
            "dataset": "ogbg-molbace",
            "split": "ogb_scaffold_test_only",
            "n_way": 2,
            "k_shot": args.k_shot,
            "q_query": args.q_query,
            "class_0": "original BACE label 0 (inactive)",
            "class_1": "original BACE label 1 (active)",
            "support_indices": episode["support_indices"],
            "query_indices": episode["query_indices"],
            "support_query_overlap": episode["support_query_overlap"],
            "n_train_reserved": protocol["n_train_reserved"],
            "n_valid_reserved": protocol["n_valid_reserved"],
            "n_fewshot_pool": protocol["n_fewshot_pool"],
        },
        "methodology": {
            "embeddings": "GINEncoder graph embeddings (not GraphCL projection head)",
            "prototypes": "mean support embedding per original BACE class",
            "prediction": "argmin Euclidean distance to prototypes",
            "confidence": "absolute difference |d0 - d1|",
            "nearest_support": "argmax cosine similarity in graph-embedding space",
        },
        "episode_accuracy": explained["accuracy"],
        "n_query": explained["n_query"],
        "explanations": explained["explanations"],
    }
    json_path = out_dir / "prototype_explanations.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Prototype-based XAI (canonical GraphCL GINEncoder)",
        "",
        "This is **not** the old 3-layer / `final_model.pt` / 9-of-10 diagnostic.",
        "",
        f"- Checkpoint: `{ckpt_path.as_posix()}`",
        f"- Encoder: {cfg}",
        f"- Episode seed: {args.seed} (one test episode only)",
        f"- Split: OGB BACE scaffold test; support={episode['support_indices']}",
        f"- Query indices: {episode['query_indices']}",
        f"- Support ∩ query: {episode['support_query_overlap']}",
        f"- Episode query accuracy: {explained['accuracy']:.3f} ({explained['n_query']} queries)",
        "",
        "## Method",
        "",
        "Frozen 5-layer GraphCL `GINEncoder` graph embeddings. Class prototypes are means of 5 support graphs per **original** BACE label. Prediction is nearest Euclidean prototype. Confidence is `|d0-d1|`. Nearest neighbor is the support graph with highest cosine similarity.",
        "",
        "GNNExplainer is **not** used.",
        "",
        "## Queries",
        "",
        "| id | true | pred | d0 | d1 | gap | nearest support | cosine |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in explained["explanations"]:
        md_lines.append(
            f"| {row['query_molecule_id']} | {row['ground_truth_label']} | "
            f"{row['predicted_class']} | {row['distance_to_class_0_prototype']:.4f} | "
            f"{row['distance_to_class_1_prototype']:.4f} | "
            f"{row['distance_gap_confidence']:.4f} | {row['most_similar_support_id']} | "
            f"{row['cosine_similarity_to_nearest_support']:.4f} |"
        )
    md_lines += [
        "",
        "## Figures",
        "",
    ]
    md_lines += [f"- `{p}`" for p in figure_paths if p]
    (out_dir / "XAI_SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 50)
    print("PROTOTYPE XAI TEST EPISODE")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Accuracy: {explained['accuracy']:.3f} on {explained['n_query']} queries")
    print(f"Outputs: {out_dir}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
