#!/usr/bin/env python3
"""
Phase 5 — Calibration assessment for BACE Protocol B distance scores.

Fits a logistic (Platt-style) map from score=(d0-d1) using TRAIN molecules
excluding the fixed support set. Evaluates on VALIDATION only.
Never touches the locked test set for fitting or model selection.

Conclusion is evidence-based: do not call outputs "calibrated probabilities"
unless metrics and reliability support that claim.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_MANIFEST = (
    REPO_ROOT / "apps" / "mol_cli" / "support_manifests" / "bace_train_k5_v1.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "phase_strengthen_2026-10-11" / "phase5_calibration"
)


def collect_scores(model, inventory, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    y = []
    s = []
    for idx in indices:
        r = model.score_query_smiles(inventory.smiles[idx])
        y.append(int(inventory.labels[idx]))
        s.append(r.distance_to_class_0 - r.distance_to_class_1)
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        w = float(mask.sum()) / n
        ece += w * abs(acc - conf)
        rows.append({"bin": i, "lo": lo, "hi": hi, "n": int(mask.sum()), "conf": conf, "acc": acc})
    return {"ece": float(ece), "n_bins": n_bins, "bins": rows}


def reliability_diagram(y: np.ndarray, p: np.ndarray, path: Path, title: str) -> None:
    info = expected_calibration_error(y, p, n_bins=10)
    xs, ys = [], []
    for b in info["bins"]:
        if b["n"] and b["conf"] is not None:
            xs.append(b["conf"])
            ys.append(b["acc"])
    fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    if xs:
        ax.scatter(xs, ys, c="#2E75B6", s=36, zorder=3, label="bin means")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency (class 1)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    from apps.mol_cli.bace_infer import BacePrototypeModel
    from apps.mol_cli.bace_support import load_bace_split_inventory
    from diagnostics.gin_embedding_validation import sha256_file

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_bace_split_inventory()
    model = BacePrototypeModel.from_checkpoint_and_manifest(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        device="cpu",
        dataset_inventory="required",
    )
    support = set(model.support_dataset_indices)
    fit_idx = [i for i in inventory.train_indices if i not in support]
    valid_idx = list(inventory.valid_indices)

    y_fit, s_fit = collect_scores(model, inventory, fit_idx)
    y_va, s_va = collect_scores(model, inventory, valid_idx)

    # Uncalibrated: logistic sigmoid of raw score is NOT justified; use
    # rank-based pseudo via empirical CDF of fit scores as a weak baseline,
    # plus a fitted Platt map for comparison.
    # Primary "uncalibrated" probability proxy: sigmoid(score) for diagnostic only.
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    p_uncal = sigmoid(s_va)  # diagnostic proxy, not a claimed probability
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(s_fit.reshape(-1, 1), y_fit)
    p_cal = clf.predict_proba(s_va.reshape(-1, 1))[:, 1]

    def pack(name: str, y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
        ece = expected_calibration_error(y, prob, n_bins=10)
        return {
            "name": name,
            "n": int(len(y)),
            "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
            "brier": float(brier_score_loss(y, prob)),
            "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7))),
            "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
            "ece": ece["ece"],
            "ece_bins": ece["bins"],
        }

    uncal = pack("sigmoid_of_raw_score_diagnostic", y_va, p_uncal)
    cal = pack("platt_logistic_on_d0_minus_d1", y_va, p_cal)

    reliability_diagram(
        y_va, p_uncal, out_dir / "reliability_uncalibrated_sigmoid.png",
        "Valid — sigmoid(d0−d1) [diagnostic]",
    )
    reliability_diagram(
        y_va, p_cal, out_dir / "reliability_platt.png",
        "Valid — Platt(d0−d1)",
    )

    # Decision: calibrated probabilities only if ECE improves AND reliability
    # is not dominated by empty/extreme bins AND AUROC remains meaningful.
    claim_ok = (
        cal["ece"] + 1e-6 < uncal["ece"]
        and cal["brier"] <= uncal["brier"] + 1e-6
        and cal["class_counts"]["0"] >= 15
        and cal["class_counts"]["1"] >= 15
    )
    # Still refuse strong claim: validation has only 21 negatives historically.
    claim_ok = False  # explicit: evidence insufficient for "calibrated probabilities"
    rationale = (
        "Prototype distances are not generative logits. Prior gap analysis showed "
        "only modest discrimination (AUROC≈0.589 on episodic gaps). Validation is "
        "severely imbalanced (≈21 inactive / 130 active), so ECE/reliability bins "
        "are underpowered. Platt on (d0−d1) is reported as a diagnostic comparison "
        "only; outputs are NOT claimed as calibrated probabilities."
    )

    report = {
        "report_id": "bace_calibration_assessment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase5_calibration",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_split": "train_minus_support",
        "eval_split": "valid",
        "n_fit": len(fit_idx),
        "n_eval": len(valid_idx),
        "locked_test_used": False,
        "score_definition": "d0 - d1 (higher => more class-1-like)",
        "uncalibrated_diagnostic": uncal,
        "platt_calibration": cal,
        "platt_coef": {"coef": clf.coef_.ravel().tolist(), "intercept": clf.intercept_.tolist()},
        "claim_calibrated_probabilities": claim_ok,
        "rationale": rationale,
        "figures": [
            "reliability_uncalibrated_sigmoid.png",
            "reliability_platt.png",
        ],
    }
    (out_dir / "bace_calibration_assessment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# Phase 5 — BACE Calibration Assessment",
        "",
        f"**Claim calibrated probabilities:** **{claim_ok}**",
        "",
        rationale,
        "",
        f"- Fit: train∖support (n={len(fit_idx)})",
        f"- Eval: validation (n={len(valid_idx)}; counts {cal['class_counts']})",
        f"- Locked test used: False",
        "",
        "| method | Brier | ECE | AUROC | log_loss |",
        "|---|---:|---:|---:|---:|",
        f"| sigmoid(score) diagnostic | {uncal['brier']:.4f} | {uncal['ece']:.4f} | {uncal['auroc']} | {uncal['log_loss']:.4f} |",
        f"| Platt(score) | {cal['brier']:.4f} | {cal['ece']:.4f} | {cal['auroc']} | {cal['log_loss']:.4f} |",
        "",
        "Figures: `reliability_uncalibrated_sigmoid.png`, `reliability_platt.png`.",
        "",
    ]
    (out_dir / "BACE_CALIBRATION_ASSESSMENT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")
    print(f"claim_calibrated_probabilities={claim_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
