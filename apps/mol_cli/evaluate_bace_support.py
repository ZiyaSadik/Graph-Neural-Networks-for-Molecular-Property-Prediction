"""
Leakage-safe evaluation of fixed-support BACE inference.

Prototypes are built only from the train-only support manifest.
Queries are official validation and test molecules, evaluated separately.
Does not train, tune support, or wire the CLI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bace_infer import (
    CANONICAL_CHECKPOINT,
    CANONICAL_MANIFEST,
    BaceInferenceError,
    BacePrototypeModel,
    resolve_checkpoint_path,
    sha256_file,
)
from .bace_metrics import MetricsError, compute_binary_metrics
from .bace_support import load_bace_split_inventory, load_manifest
from .ogb_graph import GraphConversionError, smiles_to_pyg_data

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent / "evaluation_reports"
DEFAULT_REPORT_PATH = REPORT_DIR / "bace_fixed_support_eval_v1.json"

PROTOCOL = (
    "Fixed K=5-per-class support from official OGB scaffold train only "
    "(manifest bace_train_k5_v1, seed 42). Prototypes = mean GraphCL GIN "
    "embeddings per original label 0/1. Query prediction = argmin Euclidean "
    "distance to prototypes. Validation and test are scored separately; "
    "neither is used for support or prototype construction. Distances are "
    "not calibrated probabilities. This protocol is not interchangeable with "
    "the research paper's test-episode ProtoNet evaluation."
)


class EvaluationError(RuntimeError):
    """Evaluation integrity or I/O failure."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_support_query_disjoint(
    support_indices: set[int],
    query_indices: list[int],
    split_name: str,
) -> None:
    overlap = sorted(support_indices & set(query_indices))
    if overlap:
        raise EvaluationError(
            f"Data-integrity failure: {len(overlap)} {split_name} query "
            f"indices overlap the fixed train support (e.g. {overlap[:5]})."
        )


def evaluate_split_predictions(
    model: BacePrototypeModel,
    inventory,
    query_indices: list[int],
    split_name: str,
) -> dict[str, Any]:
    """Score every query index; fail on support overlap or conversion errors."""
    support_set = set(model.support_dataset_indices)
    assert_support_query_disjoint(support_set, query_indices, split_name)

    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    rows: list[dict[str, Any]] = []

    for idx in query_indices:
        if idx < 0 or idx >= len(inventory.labels):
            raise EvaluationError(f"Out-of-range index {idx} in {split_name}.")
        label = int(inventory.labels[idx])
        if label not in (0, 1):
            raise EvaluationError(f"Non-binary label at index {idx}.")
        smiles = inventory.smiles[idx]
        try:
            # Verify graph builds; score via adapter SMILES path
            smiles_to_pyg_data(smiles)
            result = model.score_query_smiles(smiles)
        except (GraphConversionError, BaceInferenceError) as exc:
            raise EvaluationError(
                f"{split_name} index {idx} failed: {exc}"
            ) from None

        # score for class 1: larger => closer to active prototype relative to inactive
        score_c1 = result.distance_to_class_0 - result.distance_to_class_1
        y_true.append(label)
        y_pred.append(int(result.predicted_class))
        scores.append(float(score_c1))
        rows.append(
            {
                "dataset_index": int(idx),
                "true_label": label,
                "predicted_class": int(result.predicted_class),
                "distance_to_class_0": result.distance_to_class_0,
                "distance_to_class_1": result.distance_to_class_1,
                "score_class_1_d0_minus_d1": score_c1,
            }
        )

    try:
        metrics = compute_binary_metrics(y_true, y_pred, scores_class_1=scores)
    except MetricsError as exc:
        raise EvaluationError(str(exc)) from None

    return {
        "split": split_name,
        "n_official": len(query_indices),
        "n_evaluated": len(rows),
        "class_counts_true": {
            "0": sum(1 for y in y_true if y == 0),
            "1": sum(1 for y in y_true if y == 1),
        },
        "metrics": metrics,
        "predictions": rows,
    }


def build_evaluation_report(
    checkpoint_path: Path | None = None,
    manifest_path: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    ckpt = resolve_checkpoint_path(checkpoint_path)
    man_path = Path(manifest_path) if manifest_path else CANONICAL_MANIFEST
    if not man_path.is_file():
        raise EvaluationError(f"Manifest not found: {man_path}")

    manifest_raw = man_path.read_text(encoding="utf-8")
    manifest = load_manifest(man_path)
    inventory = load_bace_split_inventory()

    # Offline evaluation always requires the local OGB inventory (strict path).
    model = BacePrototypeModel.from_checkpoint_and_manifest(
        checkpoint_path=ckpt,
        manifest_path=man_path,
        device=device,
        dataset_inventory="required",
    )

    support_set = set(model.support_dataset_indices)
    # Integrity: support must be subset of train
    train_set = set(inventory.train_indices)
    if not support_set.issubset(train_set):
        raise EvaluationError("Support indices are not a subset of official train.")

    valid_result = evaluate_split_predictions(
        model, inventory, list(inventory.valid_indices), "valid"
    )
    test_result = evaluate_split_predictions(
        model, inventory, list(inventory.test_indices), "test"
    )

    # Drop per-query rows from the saved summary? Keep them for reproducibility
    # but the report can be large (~300 rows) — OK for v1.

    return {
        "report_id": "bace_fixed_support_eval_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "protocol": PROTOCOL,
        "checkpoint_path": str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_relative_to(REPO_ROOT) else str(ckpt),
        "checkpoint_sha256": model.checkpoint_sha256,
        "support_manifest_id": model.support_manifest_id,
        "support_manifest_path": str(
            man_path.relative_to(REPO_ROOT) if man_path.is_relative_to(REPO_ROOT) else man_path
        ),
        "support_manifest_sha256": sha256_text(manifest_raw),
        "support_seed": manifest.get("seed"),
        "support_dataset_indices": list(model.support_dataset_indices),
        "label_semantics": {
            "0": "inactive",
            "1": "active",
        },
        "prediction_rule": "argmin Euclidean distance to class prototypes; calibrated_probability=false",
        "auroc_score_definition": "distance_to_class_0 - distance_to_class_1",
        "not_interchangeable_with": (
            "Research paper test-episode ProtoNet accuracy (episodes sampled "
            "from test pool only)."
        ),
        "validation": valid_result,
        "test": test_result,
        "scope_limitations": [
            "Fixed train support is not tuned on validation or test.",
            "Metrics describe this protocol only; not clinical utility.",
            "Distances and AUROC scores are not calibrated probabilities.",
            "Scaffold split molecules may be distributionally related within BACE.",
        ],
    }


def write_report(report: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> Path:
    path = Path(path)
    if path.exists():
        raise EvaluationError(f"Report already exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    try:
        print("Building fixed-support BACE evaluation report...")
        print(f"Checkpoint: {CANONICAL_CHECKPOINT}")
        print(f"Manifest:   {CANONICAL_MANIFEST}")
        report = build_evaluation_report()
        path = write_report(report)
        v = report["validation"]
        t = report["test"]
        print(f"Wrote {path}")
        print(
            f"VALID n={v['n_evaluated']} acc={v['metrics']['accuracy']:.4f} "
            f"bal={v['metrics']['balanced_accuracy']} "
            f"auroc={v['metrics']['auroc']}"
        )
        print(
            f"TEST  n={t['n_evaluated']} acc={t['metrics']['accuracy']:.4f} "
            f"bal={t['metrics']['balanced_accuracy']} "
            f"auroc={t['metrics']['auroc']}"
        )
        return 0
    except (EvaluationError, BaceInferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
