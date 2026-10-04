"""Binary classification metrics for fixed-support BACE evaluation."""

from __future__ import annotations

from typing import Any


class MetricsError(ValueError):
    """Invalid inputs for metric computation."""


def confusion_matrix_binary(
    y_true: list[int],
    y_pred: list[int],
    class_order: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """
    Confusion matrix with explicit class order.

    matrix[i][j] = count of true class class_order[i] predicted as class_order[j].
    For (0, 1): [[TN, FP], [FN, TP]].
    """
    if len(y_true) != len(y_pred):
        raise MetricsError("y_true and y_pred length mismatch.")
    if class_order != (0, 1):
        raise MetricsError("Only class_order (0, 1) is supported.")

    tn = fp = fn = tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt not in (0, 1) or yp not in (0, 1):
            raise MetricsError(f"Non-binary label pair true={yt} pred={yp}.")
        if yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tp += 1

    return {
        "class_order": [0, 1],
        "class_names": ["inactive", "active"],
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def compute_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    scores_class_1: list[float] | None = None,
) -> dict[str, Any]:
    """
    Accuracy, balanced accuracy, sensitivity, specificity, optional AUROC.

    Sensitivity = recall of class 1 = TP / (TP + FN).
    Specificity = TN / (TN + FP).
    AUROC uses scores_class_1 (higher => more class-1-like), e.g. d0 - d1.
    Undefined metrics are reported as null with a reason.
    """
    if len(y_true) == 0:
        raise MetricsError("Cannot compute metrics on an empty prediction list.")

    cm = confusion_matrix_binary(y_true, y_pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = len(y_true)
    accuracy = (tp + tn) / n

    n_pos = tp + fn
    n_neg = tn + fp
    sens = (tp / n_pos) if n_pos > 0 else None
    spec = (tn / n_neg) if n_neg > 0 else None

    if n_pos > 0 and n_neg > 0:
        balanced_accuracy = 0.5 * ((tp / n_pos) + (tn / n_neg))
    else:
        balanced_accuracy = None

    auroc = None
    auroc_undefined_reason = None
    if scores_class_1 is None:
        auroc_undefined_reason = "no_scores_provided"
    elif len(scores_class_1) != len(y_true):
        raise MetricsError("scores_class_1 length mismatch.")
    elif n_pos == 0 or n_neg == 0:
        auroc_undefined_reason = "only_one_class_present_in_y_true"
    else:
        auroc = _auroc_binary(y_true, scores_class_1)
        if auroc is None:
            auroc_undefined_reason = "undefined_ranking_or_constant_scores"

    return {
        "n_evaluated": n,
        "n_true_class_0": n_neg,
        "n_true_class_1": n_pos,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity_class_1": sens,
        "specificity_class_0": spec,
        "auroc": auroc,
        "auroc_undefined_reason": auroc_undefined_reason,
        "confusion_matrix": cm,
        "metric_notes": {
            "confusion_matrix_layout": "rows=true class [0,1], cols=predicted class [0,1]",
            "sensitivity": "TP/(TP+FN) for class 1 (active)",
            "specificity": "TN/(TN+FP) for class 0 (inactive)",
            "score_for_auroc": "higher means more class-1-like (e.g. d0 - d1)",
            "calibrated_probability": False,
        },
    }


def _auroc_binary(y_true: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney / trapezoid AUROC; None if undefined."""
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return None
    # All equal scores => AUROC conventionally 0.5 but ranking undefined; treat as undefined
    if max(scores) == min(scores):
        return None

    correct = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                correct += 1.0
            elif p == n:
                correct += 0.5
    return correct / (len(pos) * len(neg))
