"""
Top-k keep/remove perturbations of T=10 GNNExplainer masks.

Uses saved masks. Does not retrain or overwrite GNNExplainer outputs.
Predictions are the canonical classifier: argmin(d_c).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from prototype_explain import sample_original_label_episode, set_seed
from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.explainability import PrototypeDistanceLogits
from src.models import GINEncoder, PrototypicalNetwork

allow_pyg_torch_load()

CHECKPOINT = Path("outputs/core_pipeline/graphcl_checkpoint.pt")
MASK_JSON = Path("outputs/core_pipeline/xai/gnnexplainer_temperature/gnnexplainer_temperature_summary.json")
OUT_DIR = Path("outputs/core_pipeline/xai/gnnexplainer_diagnostics_v2")
EPISODE_SEED = 7
K_SHOT = 5
Q_QUERY = 5
TARGET_IDS = ["BACE_118", "BACE_171", "BACE_111", "BACE_52", "BACE_253"]
FRACS = (0.10, 0.25, 0.50)


def n_at_frac(n: int, frac: float) -> int:
    """At least 1 element; ceil so 10% of 34 atoms is 4."""
    return max(1, int(np.ceil(frac * n - 1e-12)))


def topk_binary(scores: torch.Tensor, k: int, keep_top: bool) -> torch.Tensor:
    """1 on the top-k scores if keep_top else 1 on the complement."""
    flat = scores.detach().float().reshape(-1)
    order = torch.argsort(flat, descending=True, stable=True)
    mask = torch.zeros_like(flat)
    if keep_top:
        mask[order[:k]] = 1.0
    else:
        mask[order[k:]] = 1.0
        if k >= flat.numel():
            mask.zero_()
    return mask.view_as(scores)


@torch.no_grad()
def predict_with_masks(model, x, edge_index, batch, node_bin, edge_bin):
    set_masks(model, edge_bin.reshape(-1), edge_index, apply_sigmoid=False)
    logits = model(x * node_bin.view(-1, 1), edge_index, batch=batch)
    clear_masks(model)
    distances = -logits.reshape(-1)
    pred = int(distances.argmin().item())
    return pred, float(distances[0].item()), float(distances[1].item())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = json.loads(MASK_JSON.read_text(encoding="utf-8"))
    masks_by_id = {m["molecule_id"]: m for m in saved["per_molecule_masks"]}
    mol_meta = {m["molecule_id"]: m for m in saved["molecules"]}
    if [m["molecule_id"] for m in saved["molecules"]] != TARGET_IDS:
        raise RuntimeError("T=10 molecule order does not match the required five IDs")

    device = torch.device("cpu")
    set_seed(EPISODE_SEED)
    encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
    encoder.eval()
    dataset = PygGraphPropPredDataset(name="ogbg-molbace", root="dataset")
    protocol = get_bace_held_out_protocol(dataset)
    episode = sample_original_label_episode(dataset, protocol, K_SHOT, Q_QUERY, EPISODE_SEED)

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    with torch.no_grad():
        _, support_emb, _ = encoder(episode["support_batch"].to(device))
        _, query_emb, _ = encoder(episode["query_batch"].to(device))
        logits, prototypes = protonet(
            support_emb, episode["support_labels"].to(device), query_emb, n_way=2
        )
        preds = logits.argmax(dim=1)
        distances = torch.cdist(query_emb, prototypes, p=2)

    classifier = PrototypeDistanceLogits(encoder, prototypes.to(device))
    classifier.eval()
    positions = saved["selected_query_positions"]

    rows = []
    for mol_id, pos in zip(TARGET_IDS, positions):
        q_idx = int(episode["query_indices"][pos])
        query = dataset[q_idx]
        x = query.x.float().to(device)
        edge_index = query.edge_index.to(device)
        n_nodes = int(x.size(0))
        n_edges = int(edge_index.size(1))
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        orig_pred = int(preds[pos].item())
        orig_d0 = float(distances[pos, 0].item())
        orig_d1 = float(distances[pos, 1].item())
        true = int(mol_meta[mol_id]["true_label"])
        if orig_pred != int(mol_meta[mol_id]["predicted_label"]):
            raise RuntimeError(f"{mol_id}: episode prediction drifted")

        node_imp = torch.tensor(masks_by_id[mol_id]["node_importance"], device=device)
        edge_imp = torch.tensor(masks_by_id[mol_id]["edge_importance"], device=device)
        if node_imp.numel() != n_nodes or edge_imp.numel() != n_edges:
            raise RuntimeError(f"{mol_id}: mask size mismatch")

        full_pred, full_d0, full_d1 = predict_with_masks(
            classifier,
            x,
            edge_index,
            batch,
            torch.ones(n_nodes, device=device),
            torch.ones(n_edges, device=device),
        )
        if full_pred != orig_pred:
            raise RuntimeError(f"{mol_id}: full-graph wrapper mismatch")

        specs = [("full_graph", None, "none", n_nodes, n_edges)]
        for frac in FRACS:
            kn = n_at_frac(n_nodes, frac)
            ke = n_at_frac(n_edges, frac)
            specs.append((f"keep_top_{int(frac * 100)}pct", frac, "keep_top", kn, ke))
            specs.append((f"mask_top_{int(frac * 100)}pct", frac, "mask_top", kn, ke))

        for name, frac, mode, kn, ke in specs:
            if mode == "none":
                pred, d0, d1 = orig_pred, orig_d0, orig_d1
                n_kept, e_kept = n_nodes, n_edges
            else:
                keep_top = mode == "keep_top"
                node_bin = topk_binary(node_imp, kn, keep_top=keep_top)
                edge_bin = topk_binary(edge_imp, ke, keep_top=keep_top)
                pred, d0, d1 = predict_with_masks(
                    classifier, x, edge_index, batch, node_bin, edge_bin
                )
                n_kept = int(node_bin.sum().item())
                e_kept = int(edge_bin.sum().item())
            rows.append(
                {
                    "molecule_id": mol_id,
                    "dataset_index": q_idx,
                    "true_label": true,
                    "original_predicted_label": orig_pred,
                    "perturbation": name,
                    "frac": "" if frac is None else frac,
                    "n_atoms_kept": n_kept,
                    "n_edges_kept": e_kept,
                    "predicted_label": pred,
                    "flipped": int(pred != orig_pred),
                    "distance_to_class_0_prototype": d0,
                    "distance_to_class_1_prototype": d1,
                    "distance_gap": abs(d0 - d1),
                    "closer_prototype": int(np.argmin([d0, d1])),
                }
            )

    fields = [
        "molecule_id",
        "dataset_index",
        "true_label",
        "original_predicted_label",
        "perturbation",
        "frac",
        "n_atoms_kept",
        "n_edges_kept",
        "predicted_label",
        "flipped",
        "distance_to_class_0_prototype",
        "distance_to_class_1_prototype",
        "distance_gap",
        "closer_prototype",
    ]
    with (OUT_DIR / "perturbation_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    tests = [r for r in rows if r["perturbation"] != "full_graph"]
    n_flip = sum(r["flipped"] for r in tests)
    keep_flip = [r for r in tests if r["perturbation"].startswith("keep_top") and r["flipped"]]
    mask_flip = [r for r in tests if r["perturbation"].startswith("mask_top") and r["flipped"]]
    sensitive = n_flip > 0

    by_mol = []
    for mol_id in TARGET_IDS:
        mrows = [r for r in rows if r["molecule_id"] == mol_id]
        flips = [r["perturbation"] for r in mrows if r["flipped"]]
        by_mol.append(f"- **{mol_id}**: original pred {mrows[0]['original_predicted_label']}; flips: {', '.join(flips) if flips else 'none'}")

    def table_for(kind: str) -> str:
        header = (
            "| ID | orig pred | d0 | d1 | pred | flipped | atoms kept | edges kept |\n"
            "|---|---:|---:|---:|---:|:---:|---:|---:|"
        )
        lines = [header]
        for r in rows:
            if r["perturbation"] != kind:
                continue
            lines.append(
                f"| {r['molecule_id']} | {r['original_predicted_label']} | "
                f"{r['distance_to_class_0_prototype']:.2f} | {r['distance_to_class_1_prototype']:.2f} | "
                f"{r['predicted_label']} | {'yes' if r['flipped'] else 'no'} | "
                f"{r['n_atoms_kept']} | {r['n_edges_kept']} |"
            )
        return "\n".join(lines)

    if sensitive:
        verdict = (
            "The T=10 masks contain **some** decision-relevant information: at least one "
            "keep/remove setting flipped `argmin(d_c)`. GraphFramEx fid+/fid− = 0 therefore "
            "understates sensitivity. Still n=5 and not causal."
        )
        paper = (
            "Do **not** treat GNNExplainer as a finished quantitative result. Soft GraphFramEx "
            "remains uninformative. Top-k perturbations can be reported as a limited sensitivity "
            "check, not as faithful chemical explanations."
        )
    else:
        verdict = (
            "The T=10 masks do **not** show decision-relevant sensitivity at 10/25/50% keep or "
            "remove: `argmin(d_c)` never flipped. Distances may shift, but the prototype-distance "
            "class is stable. Soft fid+/fid− = 0 is consistent with this, not just a metric artifact."
        )
        paper = (
            "Treat GNNExplainer as a **qualitative limitation** of this prototype-distance setup, "
            "not as a quantitative explainability result. Do not add more molecules or claim "
            "atom-level BACE rationales from these maps."
        )

    md = f"""# T=10 GNNExplainer top-k perturbation diagnostic

Canonical frozen GraphCL GIN. Classifier: `pred = argmin(d_c)`. Masks from `outputs/core_pipeline/xai/gnnexplainer_temperature/` (T=10). Existing GNNExplainer folders were not modified.

For each molecule, atoms and directed edges are ranked by the saved GNNExplainer scores. Top *k%* uses `ceil(k * n)` (at least 1). **Keep** zeros everything outside that set. **Mask** zeros the top set and keeps the rest. Node features are multiplied by 0/1; edges use PyG `_edge_mask`.

n = 5. Not causal.

## Verdict

{verdict}

**Paper use:** {paper}

Flips in 30 keep/mask tests: **{n_flip}**. Keep-top flips: {len(keep_flip)}. Mask-top flips: {len(mask_flip)}.

{chr(10).join(by_mol)}

## Full graph

{table_for("full_graph")}

## Keep top 10%

{table_for("keep_top_10pct")}

## Keep top 25%

{table_for("keep_top_25pct")}

## Keep top 50%

{table_for("keep_top_50pct")}

## Mask top 10%

{table_for("mask_top_10pct")}

## Mask top 25%

{table_for("mask_top_25pct")}

## Mask top 50%

{table_for("mask_top_50pct")}
"""
    (OUT_DIR / "PERTURBATION_SUMMARY.md").write_text(md, encoding="utf-8")
    print(f"flips={n_flip}/30  sensitive={sensitive}")
    print(f"Wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
