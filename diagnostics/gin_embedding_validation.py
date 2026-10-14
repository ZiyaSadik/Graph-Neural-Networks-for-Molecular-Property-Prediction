#!/usr/bin/env python3
"""
Phase 2 — Reproducible GIN embedding validation for the existing GraphCL checkpoint.

Implementation checks only. Nonzero distances between different molecules are
diagnostics, not proof that the representation is chemically meaningful.
Does not train, does not touch locked-test results, does not modify the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "phase_strengthen_2026-10-11" / "phase2_gin_embedding"
)

# Chemically distinct, valid SMILES for diagnostic comparison only.
DIAGNOSTIC_MOLECULES: list[dict[str, str]] = [
    {"id": "ethanol", "smiles": "CCO", "note": "small alcohol"},
    {"id": "benzene", "smiles": "c1ccccc1", "note": "aromatic hydrocarbon"},
    {"id": "acetic_acid", "smiles": "CC(=O)O", "note": "carboxylic acid"},
    {"id": "aniline", "smiles": "Nc1ccccc1", "note": "aromatic amine"},
    {"id": "glucose_open", "smiles": "OCC(O)C(O)C(O)C(O)C=O", "note": "polyol aldehyde"},
    {"id": "caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "note": "heterocycle"},
]

NEAR_DUP_THRESHOLD = 1e-4
CONSTANT_STD_THRESHOLD = 1e-8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(checkpoint_path: Path):
    from src.models import GINEncoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "encoder_config" not in ckpt or "encoder_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing encoder_config or encoder_state_dict")
    encoder = GINEncoder.from_checkpoint(ckpt)
    encoder.eval()
    return encoder, ckpt


def embed_molecule(encoder, smiles: str):
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    data = smiles_to_pyg_data(smiles)
    with torch.no_grad():
        proj, graph_emb, node_emb = encoder(data, return_node_embeddings=True)
    return data, proj, graph_emb, node_emb


def pairwise_l2(embs: np.ndarray) -> np.ndarray:
    n = embs.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.linalg.norm(embs[i] - embs[j]))
    return dist


def run_validation(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_hash = sha256_file(checkpoint_path)
    encoder, ckpt = load_encoder(checkpoint_path)
    cfg = dict(ckpt["encoder_config"])
    n_params = sum(p.numel() for p in encoder.parameters())

    checks: dict[str, Any] = {
        "checkpoint_loads": True,
        "architecture_compatible": True,
        "node_emb_shape_matches_hidden": True,
        "graph_emb_shape_single": True,
        "graph_emb_shape_batch": True,
        "all_finite": True,
        "eval_repeatable": True,
        "not_constant_across_molecules": True,
        "not_numerically_degenerate": True,
    }
    failures: list[str] = []
    molecule_rows: list[dict[str, Any]] = []
    graph_vectors: list[np.ndarray] = []

    # --- Per-molecule checks ---
    for mol in DIAGNOSTIC_MOLECULES:
        data, proj, graph_emb, node_emb = embed_molecule(encoder, mol["smiles"])
        n_atoms = int(data.x.size(0))
        expected_node = (n_atoms, cfg["hidden_dim"])
        expected_graph = (1, cfg["hidden_dim"])

        node_ok = tuple(node_emb.shape) == expected_node
        graph_ok = tuple(graph_emb.shape) == expected_graph
        finite_ok = bool(
            torch.isfinite(node_emb).all()
            and torch.isfinite(graph_emb).all()
            and torch.isfinite(proj).all()
        )

        # Repeatability in eval mode
        _, _, graph_emb2, node_emb2 = embed_molecule(encoder, mol["smiles"])
        max_abs_graph = float((graph_emb - graph_emb2).abs().max())
        max_abs_node = float((node_emb - node_emb2).abs().max())
        repeat_ok = max_abs_graph == 0.0 and max_abs_node == 0.0

        if not node_ok:
            checks["node_emb_shape_matches_hidden"] = False
            failures.append(f"{mol['id']}: node shape {tuple(node_emb.shape)} != {expected_node}")
        if not graph_ok:
            checks["graph_emb_shape_single"] = False
            failures.append(f"{mol['id']}: graph shape {tuple(graph_emb.shape)} != {expected_graph}")
        if not finite_ok:
            checks["all_finite"] = False
            failures.append(f"{mol['id']}: non-finite values")
        if not repeat_ok:
            checks["eval_repeatable"] = False
            failures.append(
                f"{mol['id']}: eval not repeatable (max|Δ| graph={max_abs_graph}, node={max_abs_node})"
            )

        vec = graph_emb.squeeze(0).cpu().numpy().astype(np.float64)
        graph_vectors.append(vec)
        molecule_rows.append(
            {
                "id": mol["id"],
                "smiles": mol["smiles"],
                "note": mol["note"],
                "n_atoms": n_atoms,
                "n_edges": int(data.edge_index.size(1)),
                "node_emb_shape": list(node_emb.shape),
                "graph_emb_shape": list(graph_emb.shape),
                "projection_shape": list(proj.shape),
                "graph_emb_l2_norm": float(np.linalg.norm(vec)),
                "graph_emb_mean": float(vec.mean()),
                "graph_emb_std": float(vec.std()),
                "node_emb_l2_norm_mean": float(
                    torch.norm(node_emb, dim=1).mean().item()
                ),
                "finite": finite_ok,
                "eval_repeat_max_abs_delta_graph": max_abs_graph,
                "eval_repeat_max_abs_delta_node": max_abs_node,
            }
        )

    # --- Batch shape check ---
    graphs = [smiles_to_pyg_data(m["smiles"]) for m in DIAGNOSTIC_MOLECULES]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        proj_b, emb_b, node_b = encoder(batch, return_node_embeddings=True)
    b = len(DIAGNOSTIC_MOLECULES)
    batch_ok = tuple(emb_b.shape) == (b, cfg["hidden_dim"])
    if not batch_ok:
        checks["graph_emb_shape_batch"] = False
        failures.append(f"batch graph shape {tuple(emb_b.shape)} != {(b, cfg['hidden_dim'])}")
    if not bool(torch.isfinite(emb_b).all() and torch.isfinite(node_b).all()):
        checks["all_finite"] = False
        failures.append("batch embeddings contain non-finite values")

    # Compare batched vs single (order-preserving)
    single_stack = np.stack(graph_vectors, axis=0)
    batch_np = emb_b.cpu().numpy().astype(np.float64)
    max_batch_vs_single = float(np.max(np.abs(single_stack - batch_np)))

    # --- Cross-molecule diagnostics ---
    dist = pairwise_l2(single_stack)
    ids = [m["id"] for m in DIAGNOSTIC_MOLECULES]
    pairwise = []
    near_dups = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = float(dist[i, j])
            pairwise.append({"a": ids[i], "b": ids[j], "l2_distance": d})
            if d < NEAR_DUP_THRESHOLD:
                near_dups.append({"a": ids[i], "b": ids[j], "l2_distance": d})

    off_diag = dist[np.triu_indices(len(ids), k=1)]
    dim_std = single_stack.std(axis=0)
    frac_near_zero_dim = float(np.mean(dim_std < CONSTANT_STD_THRESHOLD))
    across_mol_std = float(single_stack.std())
    min_off = float(off_diag.min()) if off_diag.size else float("nan")
    max_off = float(off_diag.max()) if off_diag.size else float("nan")
    mean_off = float(off_diag.mean()) if off_diag.size else float("nan")

    if math.isfinite(min_off) and min_off < NEAR_DUP_THRESHOLD:
        checks["not_constant_across_molecules"] = False
        failures.append(
            f"near-duplicate graph embeddings detected (min pairwise L2={min_off})"
        )
    if across_mol_std < CONSTANT_STD_THRESHOLD or frac_near_zero_dim > 0.95:
        checks["not_numerically_degenerate"] = False
        failures.append(
            f"degenerate embeddings: across_mol_std={across_mol_std}, "
            f"frac_near_zero_dims={frac_near_zero_dim}"
        )

    all_pass = all(checks.values()) and not failures

    report = {
        "report_id": "gin_embedding_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase2_gin_embedding",
        "interpretation_limits": {
            "implementation_checks_only": True,
            "nonzero_distance_is_not_chemical_proof": True,
            "latent_dims_not_interpreted_as_chemical_properties": True,
            "edge_attr_unused_in_gin_forward": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": ckpt_hash,
            "encoder_config": cfg,
            "n_parameters": n_params,
            "expected_n_parameters": 184197,
            "param_count_matches": n_params == 184197,
        },
        "checks": checks,
        "failures": failures,
        "overall_pass": all_pass,
        "molecules": molecule_rows,
        "batch": {
            "batch_size": b,
            "graph_emb_shape": list(emb_b.shape),
            "node_emb_shape": list(node_b.shape),
            "projection_shape": list(proj_b.shape),
            "max_abs_delta_vs_single_forward": max_batch_vs_single,
        },
        "cross_molecule": {
            "ids": ids,
            "pairwise_l2_distances": pairwise,
            "distance_matrix": dist.tolist(),
            "min_pairwise_l2": min_off,
            "max_pairwise_l2": max_off,
            "mean_pairwise_l2": mean_off,
            "near_duplicates": near_dups,
            "near_dup_threshold": NEAR_DUP_THRESHOLD,
            "embedding_matrix_std": across_mol_std,
            "per_dim_std_mean": float(dim_std.mean()),
            "per_dim_std_min": float(dim_std.min()),
            "frac_dims_near_zero_std": frac_near_zero_dim,
            "constant_std_threshold": CONSTANT_STD_THRESHOLD,
        },
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    ck = report["checkpoint"]
    xm = report["cross_molecule"]
    lines = [
        "# Phase 2 — GIN Embedding Validation",
        "",
        f"**Report ID:** `{report['report_id']}`  ",
        f"**Created (UTC):** {report['created_utc']}  ",
        f"**Overall pass:** {report['overall_pass']}",
        "",
        "## Interpretation limits",
        "",
        "- These are **implementation diagnostics**, not evidence that embeddings encode specific chemistry.",
        "- A nonzero distance between different molecules only shows the encoder is not constant.",
        "- Individual latent dimensions are **not** interpreted as chemical properties here.",
        "- GIN forward uses `x` and `edge_index` only; `edge_attr` is unused.",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{ck['path']}`",
        f"- SHA-256: `{ck['sha256']}`",
        f"- Config: `{ck['encoder_config']}`",
        f"- Parameters: {ck['n_parameters']} (expected {ck['expected_n_parameters']})",
        "",
        "## Checks",
        "",
    ]
    for k, v in report["checks"].items():
        lines.append(f"- `{k}`: **{v}**")
    if report["failures"]:
        lines += ["", "## Failures", ""]
        for f in report["failures"]:
            lines.append(f"- {f}")
    lines += [
        "",
        "## Per-molecule graph embeddings",
        "",
        "| ID | SMILES | N | ‖emb‖₂ | mean | std | finite | repeat Δ |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in report["molecules"]:
        lines.append(
            f"| {row['id']} | `{row['smiles']}` | {row['n_atoms']} | "
            f"{row['graph_emb_l2_norm']:.4f} | {row['graph_emb_mean']:.4f} | "
            f"{row['graph_emb_std']:.4f} | {row['finite']} | "
            f"{row['eval_repeat_max_abs_delta_graph']:.2e} |"
        )
    lines += [
        "",
        "## Cross-molecule L2 distances",
        "",
        f"- min / mean / max pairwise L2: "
        f"{xm['min_pairwise_l2']:.6f} / {xm['mean_pairwise_l2']:.6f} / {xm['max_pairwise_l2']:.6f}",
        f"- near-duplicates (threshold {xm['near_dup_threshold']}): {len(xm['near_duplicates'])}",
        f"- embedding-matrix std: {xm['embedding_matrix_std']:.6f}",
        f"- fraction of dims with near-zero std: {xm['frac_dims_near_zero_std']:.4f}",
        "",
        "## Batch check",
        "",
        f"- batch graph emb shape: `{report['batch']['graph_emb_shape']}`",
        f"- max |batch − single|: {report['batch']['max_abs_delta_vs_single_forward']:.2e}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to GraphCL GIN checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for JSON/MD reports (new experiments only)",
    )
    args = parser.parse_args(argv)

    report = run_validation(args.checkpoint)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gin_embedding_validation.json"
    md_path = out_dir / "GIN_EMBEDDING_VALIDATION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"overall_pass={report['overall_pass']}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
