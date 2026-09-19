"""
Temperature-scaled GNNExplainer (T=10) on the same 5 BACE queries.

Classifier stays argmin(d_c). Only the logits passed to GNNExplainer are / T.
Does not overwrite outputs/core_pipeline/xai/gnnexplainer/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.explain import Explainer, GNNExplainer, fidelity

from gnnexplainer_analysis import (
    CHECKPOINT,
    CLASS_NAME,
    EPISODE_SEED,
    EXPLAIN_EPOCHS,
    HARD_THRESHOLD,
    K_SHOT,
    Q_QUERY,
    binary_mask,
    draw_explained_molecule,
    graphfram_ex_from_preds,
    hard_sparsity,
    mask_stats,
    masked_logits,
    pyg_version,
    select_query_positions,
    soft_sparsity,
)
from prototype_explain import (
    load_bace_mapping,
    molecule_record,
    sample_original_label_episode,
    set_seed,
)
from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.explainability import (
    PrototypeDistanceLogits,
    TemperatureScaledPrototypeLogits,
)
from src.models import GINEncoder, PrototypicalNetwork

allow_pyg_torch_load()

OLD_DIR = Path("outputs/core_pipeline/xai/gnnexplainer")
OUT_DIR = Path("outputs/core_pipeline/xai/gnnexplainer_temperature")
TEMPERATURE = 10.0
TARGET_IDS = ["BACE_118", "BACE_171", "BACE_111", "BACE_52", "BACE_253"]


def frac_ge(mask: torch.Tensor, threshold: float) -> float:
    t = mask.detach().float().reshape(-1)
    return float((t >= threshold).float().mean().item())


def old_mask_row(old_json: dict, molecule_id: str) -> dict:
    mol = next(m for m in old_json["molecules"] if m["molecule_id"] == molecule_id)
    masks = next(m for m in old_json["per_molecule_masks"] if m["molecule_id"] == molecule_id)
    node = torch.tensor(masks["node_importance"], dtype=torch.float32)
    edge = torch.tensor(masks["edge_importance"], dtype=torch.float32)
    ns = mask_stats(node)
    es = mask_stats(edge)
    return {
        "node_mask_mean": ns["mean"],
        "node_mask_std": ns["std"],
        "edge_mask_mean": es["mean"],
        "edge_mask_std": es["std"],
        "frac_atoms_ge_0_5": frac_ge(node, HARD_THRESHOLD),
        "frac_edges_ge_0_5": frac_ge(edge, HARD_THRESHOLD),
        "node_sparsity_soft": float(mol["node_sparsity_soft"]),
        "edge_sparsity_soft": float(mol["edge_sparsity_soft"]),
        "node_sparsity_hard_0_5": float(mol["node_sparsity_hard_0_5"]),
        "edge_sparsity_hard_0_5": float(mol["edge_sparsity_hard_0_5"]),
        "fidelity_plus": float(mol["fidelity_plus"]),
        "fidelity_minus": float(mol["fidelity_minus"]),
        "pred_softmax": float(mol.get("orig_pred_softmax", float("nan"))),
    }


def comparison_figure(rows: list[dict], out_path: Path) -> None:
    ids = [r["molecule_id"] for r in rows]
    x = np.arange(len(ids))
    w = 0.35
    panels = [
        ("node_mask_mean", "Node mask mean"),
        ("node_mask_std", "Node mask std"),
        ("frac_atoms_ge_0_5", "Fraction of atoms ≥ 0.5"),
        ("frac_edges_ge_0_5", "Fraction of edges ≥ 0.5"),
        ("fidelity_plus", "fid+ (soft GraphFramEx)"),
        ("fidelity_minus", "fid− (soft GraphFramEx)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), dpi=200)
    for ax, (key, title) in zip(axes.ravel(), panels):
        old = [r["old"][key] for r in rows]
        new = [r[key] for r in rows]
        ax.bar(x - w / 2, old, w, label="T=1 (previous)", color="#7f8c8d")
        ax.bar(x + w / 2, new, w, label="T=10 (this run)", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(1.05, max(old + new) * 1.15 if max(old + new) > 0 else 1.05))
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "GNNExplainer masks: unscaled −d vs logit = −d / 10  (same 5 BACE queries)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    pyg_ver = pyg_version()
    old_json = json.loads((OLD_DIR / "gnnexplainer_summary.json").read_text(encoding="utf-8"))
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

    classifier = PrototypeDistanceLogits(encoder, prototypes.to(device))
    classifier.eval()
    explainer_model = TemperatureScaledPrototypeLogits(
        encoder, prototypes.to(device), temperature=TEMPERATURE
    )
    explainer_model.eval()

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
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

        with torch.no_grad():
            cls_logits = classifier(x, edge_index, batch=batch)
            scl_logits = explainer_model(x, edge_index, batch=batch)
            cls_pred = int(cls_logits.argmax(dim=1).item())
            scl_pred = int(scl_logits.argmax(dim=1).item())
        if cls_pred != pred:
            raise RuntimeError(f"{rec['molecule_id']}: classifier {cls_pred} != ProtoNet {pred}")
        if scl_pred != pred:
            raise RuntimeError(
                f"{rec['molecule_id']}: T={TEMPERATURE} changed prediction {scl_pred} != {pred}"
            )
        argmin_d = int(torch.tensor([d0, d1]).argmin().item())
        if argmin_d != pred:
            raise RuntimeError(f"{rec['molecule_id']}: pred {pred} != argmin(d) {argmin_d}")

        explainer = Explainer(
            model=explainer_model,
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
        explanation = explainer(x, edge_index, index=0, batch=batch)
        node_mask = explanation.get("node_mask")
        edge_mask = explanation.get("edge_mask")
        if node_mask is None or edge_mask is None:
            raise RuntimeError("GNNExplainer did not return both masks")

        fid_plus, fid_minus = fidelity(explainer, explanation)
        node_hard = binary_mask(node_mask, HARD_THRESHOLD)
        edge_hard = binary_mask(edge_mask, HARD_THRESHOLD)
        ns = mask_stats(node_mask)
        es = mask_stats(edge_mask)

        with torch.no_grad():
            expl_t10 = masked_logits(
                explainer_model, explainer, x, edge_index, node_mask, edge_mask, batch
            )
            comp_t10 = masked_logits(
                explainer_model,
                explainer,
                x,
                edge_index,
                1.0 - node_mask,
                1.0 - edge_mask,
                batch,
            )
            expl_hard = masked_logits(
                explainer_model, explainer, x, edge_index, node_hard, edge_hard, batch
            )
            comp_hard = masked_logits(
                explainer_model,
                explainer,
                x,
                edge_index,
                1.0 - node_hard,
                1.0 - edge_hard,
                batch,
            )
            # Classifier-level (T=1) softmax of the original predicted class, same masks.
            cls_explainer = Explainer(
                model=classifier,
                algorithm=GNNExplainer(epochs=1, lr=0.01),
                explanation_type="model",
                node_mask_type="object",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="graph",
                    return_type="raw",
                ),
            )
            cls_expl = masked_logits(
                classifier, cls_explainer, x, edge_index, node_mask, edge_mask, batch
            )
            orig_softmax_t1 = float(torch.softmax(cls_logits, dim=1)[0, pred].item())
            orig_softmax_t10 = float(torch.softmax(scl_logits, dim=1)[0, pred].item())
            expl_softmax_t10 = float(torch.softmax(expl_t10, dim=1)[0, pred].item())
            expl_softmax_t1 = float(torch.softmax(cls_expl, dim=1)[0, pred].item())

        fid_plus_hard, fid_minus_hard = graphfram_ex_from_preds(
            pred,
            int(expl_hard.argmax(dim=1).item()),
            int(comp_hard.argmax(dim=1).item()),
        )
        if rec["molecule_id"] != TARGET_IDS[panel_i]:
            raise RuntimeError(
                f"Molecule order drifted: expected {TARGET_IDS[panel_i]}, got {rec['molecule_id']}"
            )

        old = old_mask_row(old_json, rec["molecule_id"])
        fig_path = OUT_DIR / f"explanation_{panel_i}_{rec['molecule_id']}_T10.png"
        title = (
            f"{rec['molecule_id']}   T=10   true={CLASS_NAME[true]} ({true})   "
            f"pred={CLASS_NAME[pred]} ({pred})   "
            f"{'correct' if pred == true else 'incorrect'}\n"
            f"d0={d0:.2f}   d1={d1:.2f}   "
            f"fid+={fid_plus:.3f}   fid−={fid_minus:.3f}   "
            f"node std={ns['std']:.3f}"
        )
        drawn = draw_explained_molecule(
            rec["smiles"], node_mask, edge_index, edge_mask, fig_path, title
        )
        if not drawn:
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)
            ax.bar(np.arange(n_nodes), node_mask.detach().cpu().view(-1).numpy())
            ax.set_title(title, fontsize=9)
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
            "prediction_unchanged_by_T": 1,
            "distance_to_class_0_prototype": d0,
            "distance_to_class_1_prototype": d1,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "temperature": TEMPERATURE,
            "node_mask_mean": ns["mean"],
            "node_mask_std": ns["std"],
            "edge_mask_mean": es["mean"],
            "edge_mask_std": es["std"],
            "frac_atoms_ge_0_5": frac_ge(node_mask, HARD_THRESHOLD),
            "frac_edges_ge_0_5": frac_ge(edge_mask, HARD_THRESHOLD),
            "node_sparsity_soft": soft_sparsity(node_mask),
            "edge_sparsity_soft": soft_sparsity(edge_mask),
            "node_sparsity_hard_0_5": hard_sparsity(node_mask, HARD_THRESHOLD),
            "edge_sparsity_hard_0_5": hard_sparsity(edge_mask, HARD_THRESHOLD),
            "fidelity_plus": float(fid_plus),
            "fidelity_minus": float(fid_minus),
            "fidelity_plus_hard_0_5": float(fid_plus_hard),
            "fidelity_minus_hard_0_5": float(fid_minus_hard),
            "pred_softmax_T1_classifier": orig_softmax_t1,
            "pred_softmax_T10_explainer": orig_softmax_t10,
            "pred_softmax_T10_after_soft_mask": expl_softmax_t10,
            "pred_softmax_T1_after_soft_mask": expl_softmax_t1,
            "figure": str(fig_path),
            "old": old,
            "node_importance": [float(v) for v in node_mask.view(-1).detach().cpu()],
            "edge_importance": [float(v) for v in edge_mask.view(-1).detach().cpu()],
        }
        rows.append(row)
        panel_paths.append(fig_path)
        print(
            f"{rec['molecule_id']}: pred={pred} (T=10 same) "
            f"node_std {old['node_mask_std']:.3f}->{ns['std']:.3f} "
            f"frac>=0.5 {old['frac_atoms_ge_0_5']:.3f}->{frac_ge(node_mask, HARD_THRESHOLD):.3f} "
            f"fid+ {old['fidelity_plus']:.0f}->{fid_plus:.0f} "
            f"fid- {old['fidelity_minus']:.0f}->{fid_minus:.0f}"
        )

    csv_fields = [
        "molecule_id",
        "dataset_index",
        "true_label",
        "predicted_label",
        "correct",
        "prediction_unchanged_by_T",
        "distance_to_class_0_prototype",
        "distance_to_class_1_prototype",
        "n_nodes",
        "n_edges",
        "temperature",
        "node_mask_mean",
        "node_mask_std",
        "edge_mask_mean",
        "edge_mask_std",
        "frac_atoms_ge_0_5",
        "frac_edges_ge_0_5",
        "node_sparsity_soft",
        "edge_sparsity_soft",
        "node_sparsity_hard_0_5",
        "edge_sparsity_hard_0_5",
        "fidelity_plus",
        "fidelity_minus",
        "fidelity_plus_hard_0_5",
        "fidelity_minus_hard_0_5",
        "pred_softmax_T1_classifier",
        "pred_softmax_T10_explainer",
        "pred_softmax_T10_after_soft_mask",
        "pred_softmax_T1_after_soft_mask",
        "figure",
    ]
    with (OUT_DIR / "gnnexplainer_temperature_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def mean_key(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def mean_old(key: str) -> float:
        return float(np.mean([r["old"][key] for r in rows]))

    more_concentrated = mean_key("node_mask_std") > mean_old("node_mask_std")
    more_atoms_ge = mean_key("frac_atoms_ge_0_5") > mean_old("frac_atoms_ge_0_5")
    fid_plus_improved = mean_key("fidelity_plus") > mean_old("fidelity_plus")
    fid_minus_improved = mean_key("fidelity_minus") < mean_old("fidelity_minus")

    summary = {
        "checkpoint": str(CHECKPOINT),
        "encoder_config": cfg,
        "pyg_version": pyg_ver,
        "temperature": TEMPERATURE,
        "explainer_logits": "logit_c = -d_c / T, T=10",
        "classifier": "unchanged argmin(d_c) / argmax(-d_c), PrototypeDistanceLogits T=1",
        "predictions_unchanged": True,
        "previous_results_dir": str(OLD_DIR),
        "episode_seed": EPISODE_SEED,
        "selected_query_positions": positions,
        "molecule_ids": [r["molecule_id"] for r in rows],
        "explain_epochs": EXPLAIN_EPOCHS,
        "n_molecules": len(rows),
        "means_T10": {
            "node_mask_mean": mean_key("node_mask_mean"),
            "node_mask_std": mean_key("node_mask_std"),
            "edge_mask_mean": mean_key("edge_mask_mean"),
            "edge_mask_std": mean_key("edge_mask_std"),
            "frac_atoms_ge_0_5": mean_key("frac_atoms_ge_0_5"),
            "frac_edges_ge_0_5": mean_key("frac_edges_ge_0_5"),
            "node_sparsity_soft": mean_key("node_sparsity_soft"),
            "edge_sparsity_soft": mean_key("edge_sparsity_soft"),
            "node_sparsity_hard_0_5": mean_key("node_sparsity_hard_0_5"),
            "edge_sparsity_hard_0_5": mean_key("edge_sparsity_hard_0_5"),
            "fidelity_plus": mean_key("fidelity_plus"),
            "fidelity_minus": mean_key("fidelity_minus"),
            "pred_softmax_T10_explainer": mean_key("pred_softmax_T10_explainer"),
            "pred_softmax_T1_classifier": mean_key("pred_softmax_T1_classifier"),
        },
        "means_T1_previous": {
            "node_mask_mean": mean_old("node_mask_mean"),
            "node_mask_std": mean_old("node_mask_std"),
            "edge_mask_mean": mean_old("edge_mask_mean"),
            "edge_mask_std": mean_old("edge_mask_std"),
            "frac_atoms_ge_0_5": mean_old("frac_atoms_ge_0_5"),
            "frac_edges_ge_0_5": mean_old("frac_edges_ge_0_5"),
            "node_sparsity_soft": mean_old("node_sparsity_soft"),
            "edge_sparsity_soft": mean_old("edge_sparsity_soft"),
            "node_sparsity_hard_0_5": mean_old("node_sparsity_hard_0_5"),
            "edge_sparsity_hard_0_5": mean_old("edge_sparsity_hard_0_5"),
            "fidelity_plus": mean_old("fidelity_plus"),
            "fidelity_minus": mean_old("fidelity_minus"),
            "pred_softmax": mean_old("pred_softmax"),
        },
        "more_concentrated_node_std": more_concentrated,
        "more_atoms_ge_0_5": more_atoms_ge,
        "fid_plus_improved": fid_plus_improved,
        "fid_minus_improved": fid_minus_improved,
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
            "n=5. T is an explainer hyperparameter. Explanations are not causal "
            "and do not generalize. Previous unscaled results were not overwritten."
        ),
    }
    (OUT_DIR / "gnnexplainer_temperature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    comparison_png = OUT_DIR / "gnnexplainer_temperature_comparison.png"
    comparison_figure(rows, comparison_png)

    def row_line(r: dict) -> str:
        return (
            f"| {r['molecule_id']} | {r['true_label']} | {r['predicted_label']} | "
            f"{r['node_mask_mean']:.3f} | {r['node_mask_std']:.3f} | "
            f"{r['frac_atoms_ge_0_5']:.3f} | {r['frac_edges_ge_0_5']:.3f} | "
            f"{r['node_sparsity_soft']:.3f} | {r['fidelity_plus']:.3f} | {r['fidelity_minus']:.3f} | "
            f"{r['pred_softmax_T10_explainer']:.3f} |"
        )

    md = f"""# GNNExplainer temperature scaling (T=10)

Canonical frozen GraphCL GIN from `{CHECKPOINT.as_posix()}`. Classifier unchanged: `pred = argmin(d_c) = argmax(-d_c)`.
GNNExplainer sees `logit_c = -d_c / 10`. Previous `outputs/core_pipeline/xai/gnnexplainer/` files were **not** overwritten.

Same episode seed **{EPISODE_SEED}**, same five molecules: {", ".join(TARGET_IDS)}.
All five T=10 argmax values matched the unscaled ProtoNet predictions.

This is **n = 5**. Not causal. Does not generalize.

## Did T=10 concentrate the masks?

- Mean node-mask **std**: {mean_old("node_mask_std"):.4f} (T=1) → {mean_key("node_mask_std"):.4f} (T=10). Higher std: **{more_concentrated}**.
- Mean fraction of atoms ≥ 0.5: {mean_old("frac_atoms_ge_0_5"):.3f} → {mean_key("frac_atoms_ge_0_5"):.3f}. Increase: **{more_atoms_ge}**.
- Mean node-mask mean: {mean_old("node_mask_mean"):.3f} → {mean_key("node_mask_mean"):.3f}.

## Did fidelity improve?

Soft GraphFramEx on the explainer model (argmax identical to the classifier):

- Mean fid+: {mean_old("fidelity_plus"):.3f} → {mean_key("fidelity_plus"):.3f} (higher is better). Improved: **{fid_plus_improved}**.
- Mean fid−: {mean_old("fidelity_minus"):.3f} → {mean_key("fidelity_minus"):.3f} (lower is better). Improved: **{fid_minus_improved}**.

Predicted-class softmax at T=10 is the explainer scale, not a calibrated probability. Classifier T=1 softmax remains saturated (~1).

## Per-molecule T=10

| ID | true | pred | node mean | node std | atoms≥0.5 | edges≥0.5 | node soft sp. | fid+ | fid− | softmax T=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(row_line(r) for r in rows)}

## T=1 previous (same molecules)

| ID | node mean | node std | atoms≥0.5 | edges≥0.5 | node soft sp. | fid+ | fid− | softmax T=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {r['molecule_id']} | {r['old']['node_mask_mean']:.3f} | {r['old']['node_mask_std']:.3f} | "
    f"{r['old']['frac_atoms_ge_0_5']:.3f} | {r['old']['frac_edges_ge_0_5']:.3f} | "
    f"{r['old']['node_sparsity_soft']:.3f} | {r['old']['fidelity_plus']:.3f} | "
    f"{r['old']['fidelity_minus']:.3f} | {r['old']['pred_softmax']:.3f} |"
    for r in rows
)}

GNNExplainer epochs: {EXPLAIN_EPOCHS}. PyG {pyg_ver}.
"""
    (OUT_DIR / "GNNEXPLAINER_TEMPERATURE_SUMMARY.md").write_text(md, encoding="utf-8")
    print("=" * 50)
    print(f"Wrote {OUT_DIR}")
    print(f"more_concentrated={more_concentrated}  more_atoms>=0.5={more_atoms_ge}")
    print(f"fid+ improved={fid_plus_improved}  fid- improved={fid_minus_improved}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
