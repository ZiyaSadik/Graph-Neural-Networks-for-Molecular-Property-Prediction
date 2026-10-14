#!/usr/bin/env python3
"""
Phase 3 — Rigorous fixed-support BACE few-shot evaluation (Protocol B family).

Preserves the original locked report; writes new results only under
outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/.

Protocol notes:
- Support: stratified K-per-class from official OGB scaffold TRAIN only.
- Query: full VALIDATION set for multi-seed / multi-K development stats.
- Locked TEST: reproduced only for the canonical K=5, seed=42 episode
  (consistency check vs apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json).
  Test is NOT used to select K, seed, or hyperparameters.
- Distances / (d0-d1) scores are NOT calibrated probabilities.
- Not interchangeable with Protocol A (episodic ProtoNet on the test pool).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
ORIGINAL_REPORT = (
    REPO_ROOT / "apps" / "mol_cli" / "evaluation_reports" / "bace_fixed_support_eval_v1.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "phase_strengthen_2026-10-11" / "phase3_fewshot_bace"
)

# Predeclared before looking at results — do not edit after seeing outcomes.
K_SHOTS = (1, 3, 5, 10)
SUPPORT_SEEDS = (42, 7, 13, 21, 99)
CANONICAL_K = 5
CANONICAL_SEED = 42
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity_class_1",
    "specificity_class_0",
    "auroc",
)


def _mean_std(vals: list[float | None]) -> dict[str, float | None | int]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n_valid": 0, "mean": None, "std": None}
    arr = np.asarray(clean, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"n_valid": len(clean), "mean": float(arr.mean()), "std": std}


def build_model_for_support(
    encoder,
    inventory,
    support_rows: list[dict[str, Any]],
    checkpoint_sha: str,
    manifest_id: str,
):
    """Build a lightweight scorer from selected support rows (train only)."""
    from dataclasses import replace

    from apps.mol_cli.bace_infer import (
        classify_query_embedding,
        embed_graphs,
        mean_prototypes_by_original_label,
    )
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    graphs = [
        smiles_to_pyg_data(row.get("smiles_canonical") or row["smiles_original"])
        for row in support_rows
    ]
    labels = [int(row["label"]) for row in support_rows]
    indices = [int(row["dataset_index"]) for row in support_rows]
    emb = embed_graphs(encoder, graphs, device="cpu")
    prototypes = mean_prototypes_by_original_label(emb, labels)

    class _Model:
        def __init__(self):
            self.prototypes = prototypes
            self.support_dataset_indices = tuple(indices)
            self.checkpoint_sha256 = checkpoint_sha
            self.support_manifest_id = manifest_id
            self.calibrated_probability = False

        def score_query_smiles(self, smiles: str):
            q = embed_graphs(encoder, [smiles_to_pyg_data(smiles)], device="cpu")
            result = classify_query_embedding(q[0], self.prototypes)
            return replace(
                result,
                checkpoint_sha256=self.checkpoint_sha256,
                support_manifest_id=self.support_manifest_id,
                support_dataset_indices=self.support_dataset_indices,
                calibrated_probability=False,
            )

    return _Model()

def evaluate_split(model, inventory, query_indices: list[int], split_name: str) -> dict[str, Any]:
    from apps.mol_cli.bace_metrics import compute_binary_metrics
    from apps.mol_cli.evaluate_bace_support import assert_support_query_disjoint
    from apps.mol_cli.ogb_graph import smiles_to_pyg_data

    assert_support_query_disjoint(set(model.support_dataset_indices), query_indices, split_name)

    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    failures: list[dict[str, Any]] = []

    for idx in query_indices:
        label = int(inventory.labels[idx])
        smiles = inventory.smiles[idx]
        try:
            smiles_to_pyg_data(smiles)
            result = model.score_query_smiles(smiles)
        except Exception as exc:  # record and continue for transparency
            failures.append({"dataset_index": idx, "error": str(exc)})
            continue
        y_true.append(label)
        y_pred.append(int(result.predicted_class))
        scores.append(result.distance_to_class_0 - result.distance_to_class_1)

    metrics = compute_binary_metrics(y_true, y_pred, scores_class_1=scores) if y_true else None
    return {
        "split": split_name,
        "n_query_indices": len(query_indices),
        "n_evaluated": len(y_true),
        "n_failures": len(failures),
        "failures": failures,
        "metrics": metrics,
        "support_dataset_indices": list(model.support_dataset_indices),
        "n_support": len(model.support_dataset_indices),
    }


def reproduce_canonical(encoder, inventory, checkpoint_sha: str) -> dict[str, Any]:
    """
    Reproduce using the same official evaluation path as the locked report
    (pinned manifest + BacePrototypeModel), without overwriting that report.
    """
    del encoder, inventory  # unused; official path reloads encoder/manifest

    from apps.mol_cli.evaluate_bace_support import build_evaluation_report

    reproduced = build_evaluation_report(
        checkpoint_path=DEFAULT_CHECKPOINT,
        manifest_path=REPO_ROOT
        / "apps"
        / "mol_cli"
        / "support_manifests"
        / "bace_train_k5_v1.json",
        device="cpu",
    )
    original = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
    comparisons = {}
    for split_name in ("validation", "test"):
        orig_m = original[split_name]["metrics"]
        got_m = reproduced[split_name]["metrics"]
        diffs = {}
        match = True
        for key in METRIC_KEYS:
            o = orig_m.get(key)
            g = got_m.get(key)
            if o is None and g is None:
                diffs[key] = {"original": None, "reproduced": None, "abs_diff": 0.0}
                continue
            if o is None or g is None:
                match = False
                diffs[key] = {"original": o, "reproduced": g, "abs_diff": None}
                continue
            ad = abs(float(o) - float(g))
            diffs[key] = {"original": float(o), "reproduced": float(g), "abs_diff": ad}
            # Allow tiny float noise; require exact support indices separately.
            if ad > 1e-12:
                match = False
        comparisons[split_name] = {
            "metrics_match": match,
            "support_indices_match": list(original["support_dataset_indices"])
            == list(reproduced["support_dataset_indices"]),
            "metric_diffs": diffs,
        }

    return {
        "protocol": "canonical_fixed_support_k5_seed42",
        "k_shot": CANONICAL_K,
        "seed": CANONICAL_SEED,
        "reproduction_path": "apps.mol_cli.evaluate_bace_support.build_evaluation_report",
        "original_report": str(ORIGINAL_REPORT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "original_checkpoint_sha256": original.get("checkpoint_sha256"),
        "current_checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha_match": original.get("checkpoint_sha256") == checkpoint_sha,
        "validation": {
            "metrics": reproduced["validation"]["metrics"],
            "n_evaluated": reproduced["validation"]["n_evaluated"],
            "support_dataset_indices": list(reproduced["support_dataset_indices"]),
        },
        "test": {
            "metrics": reproduced["test"]["metrics"],
            "n_evaluated": reproduced["test"]["n_evaluated"],
            "support_dataset_indices": list(reproduced["support_dataset_indices"]),
        },
        "consistency": comparisons,
        "overall_consistent": all(
            c["metrics_match"] and c["support_indices_match"] for c in comparisons.values()
        )
        and (original.get("checkpoint_sha256") == checkpoint_sha),
        "note_on_multishot_path": (
            "Multi-K episodes embed support via smiles_original from select_train_support; "
            "canonical reproduction uses the pinned manifest smiles_canonical path."
        ),
    }

def run_multishot_validation(encoder, inventory, checkpoint_sha: str) -> dict[str, Any]:
    from apps.mol_cli.bace_support import select_train_support

    episodes = []
    for k in K_SHOTS:
        for seed in SUPPORT_SEEDS:
            try:
                support = select_train_support(inventory, k_shot=k, seed=seed)
            except Exception as exc:
                episodes.append(
                    {
                        "k_shot": k,
                        "seed": seed,
                        "status": "support_selection_failed",
                        "error": str(exc),
                    }
                )
                continue
            model = build_model_for_support(
                encoder,
                inventory,
                support,
                checkpoint_sha,
                manifest_id=f"bace_train_k{k}_seed{seed}",
            )
            # Disjointness vs support
            overlap = set(model.support_dataset_indices) & set(inventory.valid_indices)
            if overlap:
                raise RuntimeError(f"Support/valid overlap for k={k} seed={seed}: {overlap}")
            result = evaluate_split(model, inventory, inventory.valid_indices, "valid")
            episodes.append(
                {
                    "k_shot": k,
                    "seed": seed,
                    "status": "ok",
                    "support_dataset_indices": list(model.support_dataset_indices),
                    "n_support": len(model.support_dataset_indices),
                    "n_query": result["n_evaluated"],
                    "n_query_failures": result["n_failures"],
                    "metrics": result["metrics"],
                }
            )

    # Aggregate by K (validation only)
    by_k = {}
    for k in K_SHOTS:
        rows = [e for e in episodes if e.get("k_shot") == k and e.get("status") == "ok"]
        agg = {"n_episodes": len(rows), "metrics": {}}
        for key in METRIC_KEYS:
            agg["metrics"][key] = _mean_std(
                [(r["metrics"] or {}).get(key) for r in rows]
            )
        # invalid AUROC count
        auroc_none = sum(
            1 for r in rows if (r["metrics"] or {}).get("auroc") is None
        )
        agg["n_auroc_undefined"] = auroc_none
        by_k[str(k)] = agg

    return {
        "query_split": "valid",
        "k_shots": list(K_SHOTS),
        "support_seeds": list(SUPPORT_SEEDS),
        "n_episodes_planned": len(K_SHOTS) * len(SUPPORT_SEEDS),
        "prototype_rule": (
            "For each class c in {0,1}, prototype_c = mean(GIN graph_embedding of "
            "support examples with original label c). Query prediction = "
            "argmin_c ||emb(q) - prototype_c||_2. Score for AUROC = d0 - d1 "
            "(higher => more class-1-like). Scores are not calibrated probabilities."
        ),
        "episodes": episodes,
        "summary_by_k": by_k,
    }


def make_plots(summary: dict[str, Any], out_dir: Path) -> list[str]:
    paths = []
    ks = [int(k) for k in summary["summary_by_k"]]
    for metric in ("balanced_accuracy", "auroc", "sensitivity_class_1", "specificity_class_0"):
        means, stds, labels = [], [], []
        for k in ks:
            cell = summary["summary_by_k"][str(k)]["metrics"][metric]
            if cell["mean"] is None:
                continue
            labels.append(str(k))
            means.append(cell["mean"])
            stds.append(cell["std"] or 0.0)
        if not means:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color="#2E75B6", ecolor="#333")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}-shot" for k in labels])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"BACE valid — {metric}\n(mean±std over {len(SUPPORT_SEEDS)} seeds)")
        ax.axhline(0.5, color="#999", ls="--", lw=0.8)
        fig.tight_layout()
        p = out_dir / f"plot_valid_{metric}.png"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return paths


def write_markdown(report: dict[str, Any], path: Path) -> None:
    repro = report["canonical_reproduction"]
    multi = report["multishot_validation"]
    lines = [
        "# Phase 3 — Rigorous Few-Shot BACE Evaluation (Protocol B family)",
        "",
        f"**Created (UTC):** {report['created_utc']}",
        "",
        "## Protocol boundaries",
        "",
        "- Support from **train only**; queries on **validation** for multi-K/multi-seed stats.",
        "- Locked **test** used only to reproduce the original K=5 seed=42 report.",
        "- Distances / (d0−d1) are **not** calibrated probabilities.",
        "- **Not interchangeable** with Protocol A (episodic ProtoNet on the test pool).",
        "- No-GraphCL checkpoint was **not available** on disk for Protocol B comparison; "
        "episodic Protocol A ablation remains the recorded GraphCL vs No-GraphCL comparison.",
        "",
        "## Canonical reproduction (K=5, seed=42)",
        "",
        f"- Checkpoint SHA match: **{repro['checkpoint_sha_match']}**",
        f"- Overall consistent with saved report: **{repro['overall_consistent']}**",
        "",
    ]
    for split in ("validation", "test"):
        c = repro["consistency"][split]
        lines.append(f"### {split}")
        lines.append(f"- support indices match: {c['support_indices_match']}")
        lines.append(f"- metrics match: {c['metrics_match']}")
        lines.append("")
        lines.append("| metric | original | reproduced | |Δ| |")
        lines.append("|---|---:|---:|---:|")
        for key, d in c["metric_diffs"].items():
            lines.append(
                f"| {key} | {d['original']} | {d['reproduced']} | {d['abs_diff']} |"
            )
        lines.append("")

    lines += [
        "## Multi-shot validation summary (predeclared seeds)",
        "",
        f"- K shots: {multi['k_shots']}",
        f"- Seeds: {multi['support_seeds']}",
        f"- Query split: `{multi['query_split']}` (n={report['n_valid_queries']})",
        f"- Episodes planned: {multi['n_episodes_planned']}",
        "",
        multi["prototype_rule"],
        "",
        "| K | n_eps | bal_acc mean±std | sens mean±std | spec mean±std | AUROC mean±std | AUROC undef |",
        "|---:|---:|---|---|---|---|---:|",
    ]
    for k in multi["k_shots"]:
        cell = multi["summary_by_k"][str(k)]
        m = cell["metrics"]

        def fmt(key: str) -> str:
            x = m[key]
            if x["mean"] is None:
                return "n/a"
            return f"{x['mean']:.4f}±{x['std']:.4f}"

        lines.append(
            f"| {k} | {cell['n_episodes']} | {fmt('balanced_accuracy')} | "
            f"{fmt('sensitivity_class_1')} | {fmt('specificity_class_0')} | "
            f"{fmt('auroc')} | {cell['n_auroc_undefined']} |"
        )
    lines += [
        "",
        "## Figures",
        "",
    ]
    for p in report.get("figure_paths", []):
        lines.append(f"- `{p}`")
    lines += [
        "",
        "## Limitations",
        "",
        "- Validation is class-imbalanced (more actives); specificity estimates have high variance at small n_neg.",
        "- Multi-K results are **not** tuned on the locked test set.",
        "- GraphCL vs No-GraphCL under this Protocol B family was not re-run (no No-GraphCL weights).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
    from diagnostics.gin_embedding_validation import load_encoder, sha256_file
    from apps.mol_cli.bace_support import load_bace_split_inventory

    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, _ = load_encoder(checkpoint_path)
    ckpt_sha = sha256_file(checkpoint_path)
    inventory = load_bace_split_inventory()

    repro = reproduce_canonical(encoder, inventory, ckpt_sha)
    multi = run_multishot_validation(encoder, inventory, ckpt_sha)
    figures = make_plots(multi, out_dir)

    report = {
        "report_id": "fewshot_bace_multishot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase3_fewshot_bace",
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": ckpt_sha,
        "n_train": len(inventory.train_indices),
        "n_valid_queries": len(inventory.valid_indices),
        "n_test_queries": len(inventory.test_indices),
        "class_counts": {
            "train": inventory.train_class_counts,
            "valid": inventory.valid_class_counts,
            "test": inventory.test_class_counts,
        },
        "canonical_reproduction": repro,
        "multishot_validation": multi,
        "figure_paths": figures,
        "original_artifacts_untouched": True,
        "no_graphcl_protocol_b_available": True,
    }

    json_path = out_dir / "fewshot_bace_multishot.json"
    md_path = out_dir / "FEWSHOT_BACE_MULTISHOT.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"canonical_consistent={repro['overall_consistent']}")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)
    report = run(args.out_dir, args.checkpoint)
    return 0 if report["canonical_reproduction"]["overall_consistent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
