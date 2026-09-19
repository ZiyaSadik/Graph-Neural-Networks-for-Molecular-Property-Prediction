"""
No-GraphCL ablation.

The encoder is the same GINEncoder architecture, randomly initialized
with seed 42, and is never trained. Few-shot evaluation replays the
exact support/query index sets from the completed GraphCL run.

This does not load or overwrite GraphCL checkpoints or core results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset

from src.data_utils import allow_pyg_torch_load, get_bace_held_out_protocol
from src.models import GINEncoder, PrototypicalNetwork
from src.training import ProtoNetEvaluator

allow_pyg_torch_load()

CORE_DIR = Path("outputs/core_pipeline")
OUT_DIR = CORE_DIR / "ablation"
GRAPHCL_EPISODES = CORE_DIR / "fewshot_results.csv"
GRAPHCL_SUMMARY = CORE_DIR / "fewshot_summary.json"
GRAPHCL_CONFIG = CORE_DIR / "experiment_config.json"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_replay_episodes(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            support = json.loads(row["support_indices"])
            query = json.loads(row["query_indices"])
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "support_indices": support,
                    "query_indices": query,
                    "n_way": int(row["n_way"]),
                    "k_shot": int(row["k_shot"]),
                    "q_query": int(row["q_query"]),
                }
            )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    core_cfg = json.loads(GRAPHCL_CONFIG.read_text(encoding="utf-8"))
    graphcl_summary = json.loads(GRAPHCL_SUMMARY.read_text(encoding="utf-8"))
    episodes = load_replay_episodes(GRAPHCL_EPISODES)
    if len(episodes) != 100:
        raise RuntimeError(f"Expected 100 GraphCL episodes, found {len(episodes)}")

    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    enc_cfg = core_cfg["encoder"]
    encoder = GINEncoder(
        in_dim=enc_cfg["in_dim"],
        hidden_dim=enc_cfg["hidden_dim"],
        num_layers=enc_cfg["num_layers"],
        dropout=enc_cfg["dropout"],
        pool_type=enc_cfg["pool_type"],
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(
        "No-GraphCL encoder: random init, 0 optimization steps, "
        f"{n_params:,} parameters, device={device}"
    )

    bace = PygGraphPropPredDataset(
        name="ogbg-molbace", root=core_cfg["fewshot"]["root"]
    )
    protocol = get_bace_held_out_protocol(bace)
    allowed = set(protocol["fewshot_pool"])
    forbidden = set(protocol["train_reserved"]) | set(protocol["valid_reserved"])

    for spec in episodes:
        idxs = set(spec["support_indices"]) | set(spec["query_indices"])
        if idxs - allowed:
            raise RuntimeError("Replay episode uses indices outside BACE test pool.")
        if idxs & forbidden:
            raise RuntimeError("Replay episode leaks into reserved BACE train/valid.")

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    evaluator = ProtoNetEvaluator(encoder=encoder, protonet=protonet, device=device)
    results = evaluator.evaluate_fixed_episodes(
        dataset=bace,
        episode_specs=episodes,
        n_way=2,
        k_shot=5,
        q_query=5,
    )

    accs = [row["accuracy"] for row in results["episodes"]]
    no_graphcl_csv = OUT_DIR / "no_graphcl_results.csv"
    with no_graphcl_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode",
            "accuracy",
            "loss",
            "n_way",
            "k_shot",
            "q_query",
            "support_indices",
            "query_indices",
            "support_query_overlap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["episodes"]:
            writer.writerow(
                {
                    **row,
                    "support_indices": json.dumps(row["support_indices"]),
                    "query_indices": json.dumps(row["query_indices"]),
                    "support_query_overlap": json.dumps(row["support_query_overlap"]),
                }
            )

    graphcl_accs = []
    with GRAPHCL_EPISODES.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graphcl_accs.append(float(row["accuracy"]))

    no_summary = {
        "condition": "no_graphcl_random_init",
        "encoder_training": "none",
        "initialization": "pytorch_default_with_seed_42",
        "optimization_steps": 0,
        "graphcl_weights_used": False,
        "mse_to_noise_used": False,
        "bace_labels_used_for_encoder_training": False,
        "episodes_replayed_from": str(GRAPHCL_EPISODES),
        "seed": seed,
        "mean_acc": results["mean_acc"],
        "std_acc": results["std_acc"],
        "min_acc": float(min(accs)),
        "max_acc": float(max(accs)),
        "mean_loss": results["mean_loss"],
        "num_episodes": results["num_episodes"],
        "split": "ogb_scaffold_test_only",
        "n_way": 2,
        "k_shot": 5,
        "q_query": 5,
        "distance": "euclidean",
    }
    write_json(OUT_DIR / "no_graphcl_summary.json", no_summary)

    ablation_config = {
        "comparison": "GraphCL pretraining vs no GraphCL (random init)",
        "fairness_rule": (
            "Conditions differ only in encoder weights: GraphCL MolPCBA "
            "pretraining vs random initialization. Downstream ProtoNet "
            "protocol, architecture, distance, seed, and episode index sets "
            "are identical. No encoder training in the No-GraphCL condition."
        ),
        "seed": seed,
        "device": str(device),
        "encoder": enc_cfg,
        "no_graphcl": {
            "initialization": "torch default, seed 42",
            "pretraining": None,
            "fine_tuning": None,
            "frozen_at_eval": True,
        },
        "graphcl_condition": {
            "results_read_from": str(GRAPHCL_SUMMARY),
            "checkpoint_not_modified": str(core_cfg["paths"]["checkpoint"]),
            "mean_acc": graphcl_summary["mean_acc"],
            "std_acc": graphcl_summary["std_acc"],
        },
        "fewshot": core_cfg["fewshot"],
        "replay_episode_count": len(episodes),
    }
    write_json(OUT_DIR / "ablation_config.json", ablation_config)

    graphcl_min = float(min(graphcl_accs))
    graphcl_max = float(max(graphcl_accs))
    compare_path = OUT_DIR / "graphcl_vs_no_graphcl.csv"
    with compare_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "mean_accuracy",
                "std_accuracy",
                "min_accuracy",
                "max_accuracy",
                "num_episodes",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition": "graphcl",
                "mean_accuracy": graphcl_summary["mean_acc"],
                "std_accuracy": graphcl_summary["std_acc"],
                "min_accuracy": graphcl_min,
                "max_accuracy": graphcl_max,
                "num_episodes": graphcl_summary["num_episodes"],
                "seed": graphcl_summary["seed"],
            }
        )
        writer.writerow(
            {
                "condition": "no_graphcl",
                "mean_accuracy": results["mean_acc"],
                "std_accuracy": results["std_acc"],
                "min_accuracy": float(min(accs)),
                "max_accuracy": float(max(accs)),
                "num_episodes": results["num_episodes"],
                "seed": seed,
            }
        )

    delta = results["mean_acc"] - graphcl_summary["mean_acc"]
    if results["mean_acc"] > graphcl_summary["mean_acc"]:
        claim = (
            "On this protocol, the No-GraphCL encoder mean accuracy is higher. "
            "Do not claim that GraphCL improves few-shot accuracy."
        )
    elif results["mean_acc"] < graphcl_summary["mean_acc"]:
        claim = (
            "On this protocol, GraphCL mean accuracy is higher than random-init "
            "No-GraphCL. This is a single seed (42) and does not by itself "
            "establish a general improvement."
        )
    else:
        claim = "Mean accuracies are identical on this run."

    md = f"""# Ablation: GraphCL vs No-GraphCL

## Method

**No-GraphCL encoder training/initialization**

- Same canonical `GINEncoder` (5 layers, hidden 128, dropout 0.1, sum pool, `in_dim=9`).
- Weights drawn from PyTorch default initialization after `seed=42`.
- **Zero optimization steps.** No GraphCL, no MSE-to-noise, no BACE supervised training, no ProtoNet meta-training.
- Encoder is frozen (`eval`, `no_grad`) during few-shot evaluation, matching the GraphCL condition.

This is a fair comparison of *whether GraphCL pretraining changes few-shot ProtoNet accuracy* because the downstream protocol is held fixed: OGB BACE scaffold test, 2-way 5-shot, 5-query, 100 episodes, Euclidean ProtoNet, same code path. Episode support/query **index sets are replayed** from `outputs/core_pipeline/fewshot_results.csv` so the two encoders see the same molecules.

GraphCL results are **read from the completed core experiment**, not recomputed. The GraphCL checkpoint was not modified and GraphCL was not rerun.

## Results

| condition | mean_accuracy | std_accuracy | min | max | episodes | seed |
|---|---:|---:|---:|---:|---:|---:|
| graphcl | {graphcl_summary["mean_acc"]} | {graphcl_summary["std_acc"]} | {graphcl_min} | {graphcl_max} | {graphcl_summary["num_episodes"]} | {graphcl_summary["seed"]} |
| no_graphcl | {results["mean_acc"]} | {results["std_acc"]} | {min(accs)} | {max(accs)} | {results["num_episodes"]} | {seed} |

No-GraphCL minus GraphCL mean accuracy: **{delta:+.6f}**.

{claim}

## Limitations

- Single seed (42).
- No-GraphCL BatchNorm running statistics are uninitialized defaults; GraphCL has statistics from MolPCBA. That is part of pretraining, not a protocol change.
- Replay uses the GraphCL run's episode list (those episodes were sampled after GraphCL RNG use). That is more controlled than resampling new episodes.
- ProtoNet does not update the encoder in either condition; this ablation does not test BACE fine-tuning.
- Historical 51.8% and 9/10 diagnostics are unused.
"""
    (OUT_DIR / "ABLATION_SUMMARY.md").write_text(md, encoding="utf-8")

    print("=" * 50)
    print("NO-GRAPHCL ABLATION")
    print("=" * 50)
    print(
        f"Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f} "
        f"over {results['num_episodes']} episodes"
    )
    print(
        f"GraphCL (stored): {graphcl_summary['mean_acc']:.4f} ± "
        f"{graphcl_summary['std_acc']:.4f}"
    )
    print(f"Delta (no_graphcl - graphcl): {delta:+.4f}")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
