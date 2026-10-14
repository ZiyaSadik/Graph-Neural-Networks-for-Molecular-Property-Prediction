#!/usr/bin/env python3
"""
Phase 4 — Reproducible BACE explainability evaluation (new experiment directory).

Uses Protocol B fixed train support (canonical K=5 seed=42) and a
*predeclared* validation index list (stratified by true label, not cherry-picked
for favorable explanations).

Includes:
- Prototype-distance explanations (primary, matching project practice)
- Optional GNNExplainer masks via PrototypeDistanceLogits
- Index-mapping checks (node index ↔ atom index)
- Stability across repeated explainer seeds
- Simple fidelity vs random-mask baseline (prediction flip / score change)
- Explicit non-causality / non-probability caveats

Does not overwrite outputs/core_pipeline/xai/* or locked-test artifacts.
Tox21: documents that only the existing 2-molecule smoke is available unless
extended separately (this script does not expand Tox21).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
DEFAULT_MANIFEST = (
    REPO_ROOT / "apps" / "mol_cli" / "support_manifests" / "bace_train_k5_v1.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "phase_strengthen_2026-10-11" / "phase4_explainability"
)

# Predeclared BEFORE seeing explanation quality. Stratified by true label from
# official validation indices (sorted), taking up to 4 inactive + 4 active.
N_PER_CLASS = 4
EXPLAIN_EPOCHS = 50
EXPLAIN_SEEDS = (0, 1)  # stability repeats
TOPK_FRAC = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def predeclare_validation_indices(inventory) -> list[int]:
    by = {0: [], 1: []}
    for idx in sorted(inventory.valid_indices):
        by[int(inventory.labels[idx])].append(int(idx))
    chosen: list[int] = []
    for cls in (0, 1):
        chosen.extend(by[cls][:N_PER_CLASS])
    return sorted(chosen)


def verify_atom_index_mapping(smiles: str, n_nodes: int) -> dict[str, Any]:
    """Check RDKit atom count matches OGB graph node count; indices 0..N-1 align."""
    from rdkit import Chem
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    mol = Chem.MolFromSmiles(smiles)
    data = smiles_to_pyg_data(smiles)
    ok = (
        mol is not None
        and mol.GetNumAtoms() == n_nodes
        and int(data.x.size(0)) == n_nodes
        and int(data.x.size(0)) == mol.GetNumAtoms()
    )
    return {
        "smiles": smiles,
        "rdkit_n_atoms": None if mol is None else mol.GetNumAtoms(),
        "graph_n_nodes": int(data.x.size(0)),
        "reported_n_nodes": n_nodes,
        "index_mapping_ok": bool(ok),
        "note": "Atom i in RDKit corresponds to node i in OGB smiles2graph for this converter.",
    }


def draw_importance(smiles: str, node_mask: np.ndarray, out_path: Path, title: str) -> bool:
    """Highlight atoms by node_mask; fails soft if drawing deps missing."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != len(node_mask):
        return False
    # Normalize mask to [0,1] for colors
    m = node_mask.astype(np.float64)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    else:
        m = np.zeros_like(m)
    atom_weights = {i: float(m[i]) for i in range(len(m))}
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors={
            i: (1.0, 1.0 - atom_weights[i], 1.0 - atom_weights[i]) for i in atom_weights
        },
    )
    drawer.FinishDrawing()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(drawer.GetDrawingText())
    return True


def fidelity_and_random(
    wrapper,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
    original_pred: int,
    original_logits: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Keep/mask top-k nodes by importance; compare to random masks of same size.

    Perturbation: zero selected node feature rows (not bond deletion).
    Limitation: feature-zeroing ≠ chemically valid substructure removal.
    """
    n = x.size(0)
    k = max(1, int(round(TOPK_FRAC * n)))
    scores = node_mask.detach().float().reshape(-1)
    top_idx = torch.argsort(scores, descending=True)[:k]
    rand_idx = torch.tensor(
        rng.choice(n, size=k, replace=False), dtype=torch.long
    )

    def run_with_zeroed(idx: torch.Tensor) -> tuple[int, list[float]]:
        x2 = x.clone()
        x2[idx] = 0.0
        with torch.no_grad():
            logits = wrapper(x2, edge_index)
        pred = int(logits.argmax(dim=-1).item())
        return pred, [float(v) for v in logits.detach().cpu().reshape(-1).tolist()]

    pred_keep_complement, logits_mask_top = run_with_zeroed(top_idx)  # remove top-k
    # keep only top-k: zero the complement
    complement = torch.tensor([i for i in range(n) if i not in set(top_idx.tolist())], dtype=torch.long)
    pred_keep_top, logits_keep_top = run_with_zeroed(complement) if len(complement) else (
        original_pred,
        [float(v) for v in original_logits.tolist()],
    )
    pred_rand, logits_rand = run_with_zeroed(rand_idx)

    return {
        "k": k,
        "topk_frac": TOPK_FRAC,
        "perturbation": "zero_node_feature_rows",
        "limitation": (
            "Zeroing atom features is not a chemically valid substructure ablation "
            "and does not establish causality."
        ),
        "original_pred": original_pred,
        "original_logits": [float(v) for v in original_logits.tolist()],
        "mask_topk_pred": pred_keep_complement,
        "mask_topk_logits": logits_mask_top,
        "mask_topk_flipped": pred_keep_complement != original_pred,
        "keep_topk_pred": pred_keep_top,
        "keep_topk_logits": logits_keep_top,
        "keep_topk_flipped": pred_keep_top != original_pred,
        "random_mask_pred": pred_rand,
        "random_mask_logits": logits_rand,
        "random_mask_flipped": pred_rand != original_pred,
    }


def run_gnnexplainer_once(
    wrapper,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target: int,
    seed: int,
) -> dict[str, Any]:
    from torch_geometric.explain import Explainer, GNNExplainer

    torch.manual_seed(seed)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    # explanation_type='model' — do not pass target (PyG warning / API contract)
    del target
    explanation = explainer(x, edge_index)
    node_mask = explanation.node_mask.detach().float().reshape(-1)
    edge_mask = explanation.edge_mask.detach().float().reshape(-1)
    return {
        "seed": seed,
        "epochs": EXPLAIN_EPOCHS,
        "node_mask": node_mask.cpu().numpy(),
        "edge_mask": edge_mask.cpu().numpy(),
        "node_mask_stats": {
            "mean": float(node_mask.mean()),
            "std": float(node_mask.std(unbiased=False)) if node_mask.numel() > 1 else 0.0,
            "min": float(node_mask.min()),
            "max": float(node_mask.max()),
        },
        "edge_mask_stats": {
            "mean": float(edge_mask.mean()),
            "std": float(edge_mask.std(unbiased=False)) if edge_mask.numel() > 1 else 0.0,
            "min": float(edge_mask.min()),
            "max": float(edge_mask.max()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-gnnexplainer", action="store_true")
    args = parser.parse_args(argv)

    from apps.mol_cli.bace_infer import BacePrototypeModel
    from apps.mol_cli.bace_support import load_bace_split_inventory
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data
    from src.explainability import PrototypeDistanceLogits
    from src.models import GINEncoder

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_sha = sha256_file(args.checkpoint)
    inventory = load_bace_split_inventory()
    query_indices = predeclare_validation_indices(inventory)

    model = BacePrototypeModel.from_checkpoint_and_manifest(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        device="cpu",
        dataset_inventory="required",
    )
    encoder = model.encoder
    encoder.eval()

    molecules: list[dict[str, Any]] = []
    for idx in query_indices:
        smiles = inventory.smiles[idx]
        true_y = int(inventory.labels[idx])
        result = model.score_query_smiles(smiles)
        data = smiles_to_pyg_data(smiles)
        mapping = verify_atom_index_mapping(smiles, int(data.x.size(0)))
        molecules.append(
            {
                "dataset_index": idx,
                "split": "valid",
                "mol_id": inventory.mol_ids[idx],
                "smiles": smiles,
                "true_label": true_y,
                "predicted_class": int(result.predicted_class),
                "distance_to_class_0": result.distance_to_class_0,
                "distance_to_class_1": result.distance_to_class_1,
                "distance_gap": result.distance_gap,
                "correct": int(result.predicted_class) == true_y,
                "index_mapping": mapping,
                "n_nodes": int(data.x.size(0)),
                "n_edges": int(data.edge_index.size(1)),
            }
        )

    n_pos_pred = sum(1 for m in molecules if m["predicted_class"] == 1)
    n_neg_pred = sum(1 for m in molecules if m["predicted_class"] == 0)

    gnn_rows: list[dict[str, Any]] = []
    gnn_status = {"attempted": False, "succeeded": 0, "failed": 0, "errors": []}

    if not args.skip_gnnexplainer:
        gnn_status["attempted"] = True
        wrapper = PrototypeDistanceLogits(encoder, model.prototypes)
        wrapper.eval()
        for mol in molecules:
            smiles = mol["smiles"]
            data = smiles_to_pyg_data(smiles)
            x = data.x.float()
            edge_index = data.edge_index
            target = int(mol["predicted_class"])
            with torch.no_grad():
                logits = wrapper(x, edge_index).reshape(-1)
            row: dict[str, Any] = {
                "dataset_index": mol["dataset_index"],
                "mol_id": mol["mol_id"],
                "target_prediction": target,
                "logits": [float(v) for v in logits.tolist()],
                "runs": [],
                "stability": None,
                "fidelity": None,
                "figure": None,
            }
            try:
                run_masks = []
                for seed in EXPLAIN_SEEDS:
                    out = run_gnnexplainer_once(wrapper, x, edge_index, target, seed)
                    run_masks.append(out["node_mask"])
                    row["runs"].append(
                        {
                            "seed": seed,
                            "epochs": out["epochs"],
                            "node_mask_stats": out["node_mask_stats"],
                            "edge_mask_stats": out["edge_mask_stats"],
                            "node_mask": out["node_mask"].tolist(),
                            "edge_mask": out["edge_mask"].tolist(),
                        }
                    )
                # stability: pearson / spearman of node masks across seeds
                if len(run_masks) >= 2:
                    a, b = run_masks[0], run_masks[1]
                    if a.std() > 0 and b.std() > 0:
                        pearson = float(np.corrcoef(a, b)[0, 1])
                    else:
                        pearson = float("nan")
                    row["stability"] = {
                        "seeds": list(EXPLAIN_SEEDS),
                        "node_mask_pearson": pearson,
                        "mean_abs_delta": float(np.mean(np.abs(a - b))),
                    }
                # fidelity using first seed mask
                rng = np.random.default_rng(123 + mol["dataset_index"])
                row["fidelity"] = fidelity_and_random(
                    wrapper,
                    x,
                    edge_index,
                    torch.tensor(run_masks[0]),
                    target,
                    logits,
                    rng,
                )
                fig_path = fig_dir / f"bace_{mol['dataset_index']}_atom_importance.png"
                ok = draw_importance(
                    smiles,
                    run_masks[0],
                    fig_path,
                    title=f"{mol['mol_id']} pred={target}",
                )
                if ok:
                    row["figure"] = str(fig_path.relative_to(REPO_ROOT)).replace("\\", "/")
                gnn_status["succeeded"] += 1
            except Exception as exc:
                gnn_status["failed"] += 1
                gnn_status["errors"].append(
                    {"dataset_index": mol["dataset_index"], "error": str(exc)}
                )
                row["error"] = str(exc)
            gnn_rows.append(row)

    report = {
        "report_id": "bace_explainability_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase4_explainability",
        "caveats": [
            "High atom/bond importance is not chemical causality.",
            "Prototype distance gaps are not calibrated probabilities.",
            "GNNExplainer masks on PrototypeDistanceLogits have historically shown "
            "poor soft fidelity in this project; treat new fidelity numbers cautiously.",
            "Feature-zeroing perturbations are not chemically valid ablations.",
            "Tox21 explainability remains limited to the existing 2-molecule smoke test "
            "under outputs/tox21/phase4f6_explainability/; not extended here.",
        ],
        "checkpoint": {
            "path": str(args.checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": ckpt_sha,
        },
        "support_manifest": str(args.manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
        "support_dataset_indices": list(model.support_dataset_indices),
        "protocol": (
            "Protocol B: fixed train K=5 support (manifest). Queries = predeclared "
            "validation indices (stratified by true label, sorted), not cherry-picked."
        ),
        "predeclared_query_indices": query_indices,
        "selection_rule": (
            f"From official valid split, take first {N_PER_CLASS} indices per class "
            "after sorting indices ascending."
        ),
        "n_molecules": len(molecules),
        "n_predicted_active": n_pos_pred,
        "n_predicted_inactive": n_neg_pred,
        "prototype_explanations": molecules,
        "gnnexplainer": {
            "settings": {
                "api": "torch_geometric.explain.Explainer + GNNExplainer",
                "wrapper": "PrototypeDistanceLogits (logit = -distance)",
                "epochs": EXPLAIN_EPOCHS,
                "seeds": list(EXPLAIN_SEEDS),
                "node_mask_type": "object",
                "edge_mask_type": "object",
                "edge_attr_used_by_encoder": False,
            },
            "status": gnn_status,
            "molecules": gnn_rows,
        },
        "index_mapping_all_ok": all(
            m["index_mapping"]["index_mapping_ok"] for m in molecules
        ),
    }

    json_path = out_dir / "bace_explainability_eval.json"
    # Store a lighter JSON without huge duplicate masks in MD; keep full in JSON
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Phase 4 — BACE Explainability Evaluation",
        "",
        f"**Created (UTC):** {report['created_utc']}",
        "",
        "## Caveats",
        "",
    ]
    for c in report["caveats"]:
        md.append(f"- {c}")
    md += [
        "",
        "## Setup",
        "",
        f"- Checkpoint SHA: `{ckpt_sha}`",
        f"- Support manifest: `{report['support_manifest']}`",
        f"- Predeclared valid indices: `{query_indices}`",
        f"- Predictions: active={n_pos_pred}, inactive={n_neg_pred}",
        f"- Index mapping OK: **{report['index_mapping_all_ok']}**",
        "",
        "## Prototype-distance explanations",
        "",
        "| idx | mol_id | true | pred | d0 | d1 | gap | correct |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in molecules:
        md.append(
            f"| {m['dataset_index']} | {m['mol_id']} | {m['true_label']} | "
            f"{m['predicted_class']} | {m['distance_to_class_0']:.3f} | "
            f"{m['distance_to_class_1']:.3f} | {m['distance_gap']:.3f} | {m['correct']} |"
        )
    md += [
        "",
        "## GNNExplainer",
        "",
        f"- Attempted: {gnn_status['attempted']}",
        f"- Succeeded / failed: {gnn_status['succeeded']} / {gnn_status['failed']}",
        "",
    ]
    for row in gnn_rows:
        st = row.get("stability") or {}
        fid = row.get("fidelity") or {}
        md.append(
            f"- `{row['mol_id']}` target={row.get('target_prediction')} "
            f"pearson={st.get('node_mask_pearson')} "
            f"mask_topk_flip={fid.get('mask_topk_flipped')} "
            f"random_flip={fid.get('random_mask_flipped')} "
            f"fig={row.get('figure')}"
        )
    md += [
        "",
        "## Tox21",
        "",
        "Not extended beyond the existing two-molecule smoke under "
        "`outputs/tox21/phase4f6_explainability/`. A broader reproducible Tox21 "
        "explanation study would require a predeclared multi-molecule protocol "
        "and fidelity design that this phase did not add.",
        "",
    ]
    (out_dir / "BACE_EXPLAINABILITY_EVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'BACE_EXPLAINABILITY_EVAL.md'}")
    print(
        f"molecules={len(molecules)} gnn_ok={gnn_status['succeeded']} "
        f"mapping_ok={report['index_mapping_all_ok']}"
    )
    return 0 if report["index_mapping_all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
