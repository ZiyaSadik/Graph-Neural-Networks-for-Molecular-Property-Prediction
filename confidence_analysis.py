"""
Distance-gap usefulness analysis for prototype ProtoNet.

Uses the canonical GraphCL checkpoint only. Does not retrain GraphCL.
Does not treat |d0-d1| as a probability or calibrated confidence.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

from prototype_explain import load_bace_mapping, molecule_record
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
SEED = 42
NUM_EPISODES = 100
K_SHOT = 5
Q_QUERY = 5
N_BINS = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_original_label_episode(sampler, k_shot: int, q_query: int):
    needed = k_shot + q_query
    support_indices = []
    query_indices = []
    support_labels = []
    query_labels = []
    for original_label in (0, 1):
        candidates = sampler.label_to_indices[original_label]
        sampled = random.sample(candidates, needed)
        supp = sampled[:k_shot]
        qry = sampled[k_shot:]
        if set(supp) & set(qry):
            raise RuntimeError("Support/query overlap")
        support_indices.extend(supp)
        query_indices.extend(qry)
        support_labels.extend([original_label] * k_shot)
        query_labels.extend([original_label] * q_query)
    overlap = sorted(set(support_indices) & set(query_indices))
    if overlap:
        raise RuntimeError(overlap)
    return {
        "support_indices": support_indices,
        "query_indices": query_indices,
        "support_labels": torch.tensor(support_labels, dtype=torch.long),
        "query_labels": torch.tensor(query_labels, dtype=torch.long),
        "support_batch": Batch.from_data_list(
            [sampler.dataset[i] for i in support_indices]
        ),
        "query_batch": Batch.from_data_list(
            [sampler.dataset[i] for i in query_indices]
        ),
        "support_query_overlap": overlap,
    }


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
    encoder = encoder.to(device)
    encoder.eval()
    cfg = encoder.get_config()
    if cfg["num_layers"] != 5:
        raise RuntimeError(cfg)

    dataset = PygGraphPropPredDataset(name="ogbg-molbace", root="dataset")
    protocol = get_bace_held_out_protocol(dataset)
    mapping = load_bace_mapping(dataset)

    set_seed(SEED)
    sampler = EpisodeSampler(
        dataset,
        device="cpu",
        allowed_indices=protocol["fewshot_pool"],
        forbidden_indices=protocol["train_reserved"] + protocol["valid_reserved"],
        verbose=True,
    )
    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    explainer = PrototypeExplainer(encoder, protonet, device=device)

    records = []
    for ep in range(NUM_EPISODES):
        episode = sample_original_label_episode(sampler, K_SHOT, Q_QUERY)
        q_recs = [molecule_record(mapping, i) for i in episode["query_indices"]]
        explained = explainer.explain_episode(
            episode["support_batch"],
            episode["support_labels"],
            episode["query_batch"],
            episode["query_labels"],
            n_way=2,
            query_indices=episode["query_indices"],
            support_indices=episode["support_indices"],
            query_ids=[r["molecule_id"] for r in q_recs],
        )
        for row in explained["explanations"]:
            records.append(
                {
                    "episode": ep,
                    "query_molecule_id": row["query_molecule_id"],
                    "query_index": row["query_index"],
                    "true_label": row["ground_truth_label"],
                    "predicted_label": row["predicted_class"],
                    "correct": int(row["correct"]),
                    "distance_to_class_0_prototype": row[
                        "distance_to_class_0_prototype"
                    ],
                    "distance_to_class_1_prototype": row[
                        "distance_to_class_1_prototype"
                    ],
                    "distance_gap": row["distance_gap_confidence"],
                }
            )

    df = pd.DataFrame(records)
    if df["distance_gap"].isna().any():
        raise RuntimeError("Missing distance_gap")

    correct = df["correct"] == 1
    gap = df["distance_gap"].to_numpy()
    y = df["correct"].to_numpy()
    mean_gap_correct = float(df.loc[correct, "distance_gap"].mean())
    mean_gap_incorrect = float(df.loc[~correct, "distance_gap"].mean())
    n_correct = int(correct.sum())
    n_incorrect = int((~correct).sum())
    overall_acc = float(y.mean())
    r_pb = point_biserial(gap, y.astype(float))
    try:
        auroc = float(roc_auc_score(y, gap))
    except ValueError:
        auroc = float("nan")

    df = df.copy()
    df["gap_bin"] = pd.qcut(
        df["distance_gap"], q=N_BINS, duplicates="drop", labels=False
    )
    bin_rows = []
    for b, g in df.groupby("gap_bin", sort=True):
        bin_rows.append(
            {
                "bin": int(b),
                "n": int(len(g)),
                "gap_min": float(g["distance_gap"].min()),
                "gap_max": float(g["distance_gap"].max()),
                "gap_mean": float(g["distance_gap"].mean()),
                "accuracy": float(g["correct"].mean()),
            }
        )
    bin_accs = [r["accuracy"] for r in bin_rows]
    bin_means = [r["gap_mean"] for r in bin_rows]
    acc_increases_with_gap = all(
        bin_accs[i] <= bin_accs[i + 1] + 1e-12 for i in range(len(bin_accs) - 1)
    )
    spearman_bin = (
        float(np.corrcoef(bin_means, bin_accs)[0, 1]) if len(bin_rows) > 1 else float("nan")
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "confidence_analysis.csv"
    df.drop(columns=["gap_bin"]).to_csv(csv_path, index=False)

    larger_gap_better = (
        mean_gap_correct > mean_gap_incorrect and r_pb > 0 and auroc > 0.5
    )
    summary = {
        "checkpoint": str(CHECKPOINT),
        "encoder_config": cfg,
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "n_way": 2,
        "k_shot": K_SHOT,
        "q_query": Q_QUERY,
        "n_query_predictions": int(len(df)),
        "overall_accuracy": overall_acc,
        "mean_distance_gap_correct": mean_gap_correct,
        "mean_distance_gap_incorrect": mean_gap_incorrect,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "point_biserial_gap_vs_correct": r_pb,
        "auroc_gap_as_score_for_correctness": auroc,
        "quantile_bins": bin_rows,
        "quantile_bin_accuracy_monotone_nondecreasing": acc_increases_with_gap,
        "pearson_bin_mean_gap_vs_bin_accuracy": spearman_bin,
        "distance_gap_is_probability": False,
        "distance_gap_is_calibrated_confidence": False,
        "larger_gaps_correspond_to_higher_accuracy": bool(larger_gap_better),
        "interpretation_note": (
            "Distance-gap is |d0-d1| in embedding space, not a class probability. "
            "It is not a calibrated confidence score."
        ),
    }
    (OUT_DIR / "confidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot(
        [
            df.loc[~correct, "distance_gap"].to_numpy(),
            df.loc[correct, "distance_gap"].to_numpy(),
        ],
        tick_labels=["incorrect", "correct"],
    )
    axes[0].set_ylabel("Distance-gap |d0 − d1|")
    axes[0].set_title("Gap by correctness")
    axes[0].grid(axis="y", alpha=0.3)

    xs = [r["gap_mean"] for r in bin_rows]
    ys = [r["accuracy"] for r in bin_rows]
    ns = [r["n"] for r in bin_rows]
    axes[1].plot(xs, ys, "o-", color="steelblue")
    for x, yv, n in zip(xs, ys, ns):
        axes[1].annotate(f"n={n}", (x, yv), textcoords="offset points", xytext=(4, 6), fontsize=8)
    axes[1].set_xlabel("Bin mean distance-gap")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy vs gap quantile bins")
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Prototype distance-gap analysis (seed={SEED}, {NUM_EPISODES} episodes)\n"
        "Gap is not a probability",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confidence_vs_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if larger_gap_better:
        gap_verdict = (
            "On this run, correct queries had a larger mean gap than incorrect "
            "queries, and gap ranked correctness better than chance (AUROC>0.5). "
            "That is an empirical association only; the gap is still not a probability."
        )
    else:
        gap_verdict = (
            "Larger distance-gaps do **not** reliably correspond to higher "
            "correctness on this protocol. Do not treat |d0-d1| as useful "
            "calibrated confidence."
        )

    bin_md = [
        "| bin | n | gap min | gap max | gap mean | accuracy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in bin_rows:
        bin_md.append(
            f"| {r['bin']} | {r['n']} | {r['gap_min']:.4f} | {r['gap_max']:.4f} | "
            f"{r['gap_mean']:.4f} | {r['accuracy']:.4f} |"
        )

    md = f"""# Distance-gap analysis (not calibrated confidence)

Checkpoint: `{CHECKPOINT.as_posix()}` (canonical 5-layer GraphCL GINEncoder).  
Protocol: OGB BACE scaffold test, frozen ProtoNet, 2-way 5-shot, {NUM_EPISODES} episodes, seed **{SEED}**.  
Class 0/1 prototypes use original BACE labels. Support and query are disjoint.  
GraphCL was **not** retrained. `|d0-d1|` is **not** a probability and **not** calibrated confidence.

## Query-level counts

- Predictions: {len(df)}
- Overall accuracy: {overall_acc:.6f}
- Correct: {n_correct}; incorrect: {n_incorrect}

## Mean distance-gap

- Correct: **{mean_gap_correct:.6f}**
- Incorrect: **{mean_gap_incorrect:.6f}**
- Difference (correct − incorrect): **{mean_gap_correct - mean_gap_incorrect:+.6f}**

## Association with correctness

- Point-biserial correlation (gap vs correct): **{r_pb:.6f}**
- AUROC treating gap as a score for being correct: **{auroc:.6f}**

## Accuracy by gap quantile bins (q={N_BINS})

{chr(10).join(bin_md)}

Monotone nondecreasing accuracy across bins: {acc_increases_with_gap}  
Pearson(bin mean gap, bin accuracy): {spearman_bin:.6f}

## Verdict

{gap_verdict}

## Calibration limitations

A calibrated confidence would be a predicted probability whose bin-wise frequency matches the value. Distance-gap is an unnormalized embedding-space margin. It has no mapping to [0, 1] in this experiment, so expected calibration error is not defined for the raw gap. Sum-pooling yields large Euclidean distances; gap magnitude is scale-dependent and not comparable to a likelihood.
"""
    (OUT_DIR / "CONFIDENCE_ANALYSIS.md").write_text(md, encoding="utf-8")
    print("=" * 50)
    print(f"n={len(df)} acc={overall_acc:.4f}")
    print(f"mean gap correct={mean_gap_correct:.4f} incorrect={mean_gap_incorrect:.4f}")
    print(f"point-biserial={r_pb:.4f} AUROC={auroc:.4f}")
    print(f"Wrote {OUT_DIR}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
