"""
Multi-seed paired robustness check: GraphCL checkpoint vs random-init encoder.

Does not retrain GraphCL or write to graphcl_checkpoint.pt.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset

from src.data_utils import (
    EpisodeSampler,
    allow_pyg_torch_load,
    get_bace_held_out_protocol,
)
from src.models import GINEncoder, PrototypicalNetwork
from src.training import ProtoNetEvaluator

allow_pyg_torch_load()

CORE_DIR = Path("outputs/core_pipeline")
OUT_DIR = CORE_DIR / "ablation"
CHECKPOINT = CORE_DIR / "graphcl_checkpoint.pt"
SEEDS = [42, 43, 44, 45, 46]
NUM_EPISODES = 100
N_WAY = 2
K_SHOT = 5
Q_QUERY = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_key(spec: dict) -> tuple:
    return (
        tuple(spec["support_indices"]),
        tuple(spec["query_indices"]),
    )


def sample_episode_specs(dataset, protocol, seed: int, num_episodes: int):
    set_seed(seed)
    sampler = EpisodeSampler(
        dataset,
        device="cpu",
        allowed_indices=protocol["fewshot_pool"],
        forbidden_indices=protocol["train_reserved"] + protocol["valid_reserved"],
        verbose=False,
    )
    specs = []
    for ep in range(num_episodes):
        _, _, _, _, meta = sampler.sample_episode(N_WAY, K_SHOT, Q_QUERY)
        if meta["support_query_overlap"]:
            raise RuntimeError(f"Support/query overlap at episode {ep}: {meta}")
        specs.append(
            {
                "episode": ep,
                "support_indices": list(meta["support_indices"]),
                "query_indices": list(meta["query_indices"]),
            }
        )
    return specs


def verify_paired_generation(dataset, protocol) -> dict:
    a = sample_episode_specs(dataset, protocol, 42, NUM_EPISODES)
    b = sample_episode_specs(dataset, protocol, 42, NUM_EPISODES)
    c = sample_episode_specs(dataset, protocol, 43, NUM_EPISODES)
    same_42 = [episode_key(x) for x in a] == [episode_key(x) for x in b]
    diff_43 = [episode_key(x) for x in a] != [episode_key(x) for x in c]
    if not same_42:
        raise RuntimeError("Paired episode generation failed: seed 42 resample mismatch.")
    if not diff_43:
        raise RuntimeError("Seed 42 and 43 produced identical episode lists.")
    n_changed = sum(
        episode_key(x) != episode_key(y) for x, y in zip(a, c)
    )
    report = {
        "seed_42_resample_identical": True,
        "seed_42_vs_43_lists_differ": True,
        "episodes_compared": NUM_EPISODES,
        "n_episodes_different_42_vs_43": n_changed,
    }
    print("Paired-episode check passed:")
    print(f"  seed 42 resample identical: {same_42}")
    print(f"  seed 42 vs 43 differ: {diff_43} ({n_changed}/{NUM_EPISODES} episodes differ)")
    return report


def make_no_graphcl_encoder(enc_cfg, seed: int, device):
    set_seed(seed)
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
    return encoder


def summarize(results):
    accs = [row["accuracy"] for row in results["episodes"]]
    return {
        "mean_accuracy": results["mean_acc"],
        "std_accuracy": results["std_acc"],
        "min_accuracy": float(min(accs)),
        "max_accuracy": float(max(accs)),
        "num_episodes": results["num_episodes"],
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_cfg = json.loads((CORE_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    enc_cfg = core_cfg["encoder"]

    bace = PygGraphPropPredDataset(
        name="ogbg-molbace", root=core_cfg["fewshot"]["root"]
    )
    protocol = get_bace_held_out_protocol(bace)

    pairing_report = verify_paired_generation(bace, protocol)

    print(f"Loading GraphCL checkpoint (read-only): {CHECKPOINT}")
    graphcl_encoder = GINEncoder.from_checkpoint(str(CHECKPOINT), map_location=device)
    graphcl_encoder = graphcl_encoder.to(device)
    graphcl_encoder.eval()
    for p in graphcl_encoder.parameters():
        p.requires_grad_(False)

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    graphcl_eval = ProtoNetEvaluator(graphcl_encoder, protonet, device=device)

    rows = []
    per_seed = []

    for seed in SEEDS:
        specs = sample_episode_specs(bace, protocol, seed, NUM_EPISODES)
        specs_again = sample_episode_specs(bace, protocol, seed, NUM_EPISODES)
        if [episode_key(x) for x in specs] != [episode_key(x) for x in specs_again]:
            raise RuntimeError(f"Episode pairing failed for seed {seed}")

        g_res = graphcl_eval.evaluate_fixed_episodes(
            bace, specs, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY
        )
        no_encoder = make_no_graphcl_encoder(enc_cfg, seed, device)
        no_eval = ProtoNetEvaluator(no_encoder, protonet, device=device)
        n_res = no_eval.evaluate_fixed_episodes(
            bace, specs, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY
        )

        g_sum = summarize(g_res)
        n_sum = summarize(n_res)
        diff_g_minus_n = g_sum["mean_accuracy"] - n_sum["mean_accuracy"]
        print(
            f"seed {seed}: GraphCL {g_sum['mean_accuracy']:.4f} ± {g_sum['std_accuracy']:.4f} | "
            f"No-GraphCL {n_sum['mean_accuracy']:.4f} ± {n_sum['std_accuracy']:.4f} | "
            f"G-N {diff_g_minus_n:+.4f}"
        )

        rows.append(
            {
                "condition": "graphcl",
                "seed": seed,
                "mean_accuracy": g_sum["mean_accuracy"],
                "std_accuracy": g_sum["std_accuracy"],
                "num_episodes": g_sum["num_episodes"],
            }
        )
        rows.append(
            {
                "condition": "no_graphcl",
                "seed": seed,
                "mean_accuracy": n_sum["mean_accuracy"],
                "std_accuracy": n_sum["std_accuracy"],
                "num_episodes": n_sum["num_episodes"],
            }
        )
        per_seed.append(
            {
                "seed": seed,
                "graphcl_mean_accuracy": g_sum["mean_accuracy"],
                "graphcl_std_accuracy": g_sum["std_accuracy"],
                "no_graphcl_mean_accuracy": n_sum["mean_accuracy"],
                "no_graphcl_std_accuracy": n_sum["std_accuracy"],
                "difference_graphcl_minus_no_graphcl": diff_g_minus_n,
                "difference_no_graphcl_minus_graphcl": -diff_g_minus_n,
                "n_episodes_identical_between_conditions": NUM_EPISODES,
            }
        )

    g_means = [x["graphcl_mean_accuracy"] for x in per_seed]
    n_means = [x["no_graphcl_mean_accuracy"] for x in per_seed]
    diffs = [x["difference_graphcl_minus_no_graphcl"] for x in per_seed]

    across = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "graphcl_mean_across_seeds": float(np.mean(g_means)),
        "graphcl_std_across_seeds": float(np.std(g_means, ddof=1)),
        "no_graphcl_mean_across_seeds": float(np.mean(n_means)),
        "no_graphcl_std_across_seeds": float(np.std(n_means, ddof=1)),
        "mean_difference_graphcl_minus_no_graphcl": float(np.mean(diffs)),
        "std_difference_graphcl_minus_no_graphcl": float(np.std(diffs, ddof=1)),
        "across_seed_std_is_sample_ddof1": True,
        "n_seeds_graphcl_higher": int(sum(d > 0 for d in diffs)),
        "n_seeds_no_graphcl_higher": int(sum(d < 0 for d in diffs)),
        "n_seeds_tied": int(sum(d == 0 for d in diffs)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "multi_seed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "mean_accuracy",
                "std_accuracy",
                "num_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_modified": False,
        "graphcl_retrained": False,
        "pairing_check": pairing_report,
        "protocol": {
            "dataset": "ogbg-molbace",
            "split": "ogb_scaffold_test_only",
            "n_way": N_WAY,
            "k_shot": K_SHOT,
            "q_query": Q_QUERY,
            "num_episodes_per_seed": NUM_EPISODES,
            "encoder_frozen": True,
            "distance": "euclidean",
        },
        "per_seed": per_seed,
        "across_seeds": across,
        "note": (
            "This is a robustness analysis over episode seeds. "
            "Do not treat the original single-seed ablation as the only evidence. "
            "GraphCL weights are fixed from the completed core checkpoint."
        ),
    }
    write_json(OUT_DIR / "multi_seed_summary.json", summary)

    table_lines = [
        "| seed | GraphCL mean | GraphCL std | No-GraphCL mean | No-GraphCL std | GraphCL − No-GraphCL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_seed:
        table_lines.append(
            f"| {item['seed']} | {item['graphcl_mean_accuracy']:.6f} | "
            f"{item['graphcl_std_accuracy']:.6f} | {item['no_graphcl_mean_accuracy']:.6f} | "
            f"{item['no_graphcl_std_accuracy']:.6f} | "
            f"{item['difference_graphcl_minus_no_graphcl']:+.6f} |"
        )

    md = f"""# Multi-seed robustness: GraphCL vs No-GraphCL

GraphCL was **not** retrained. Checkpoint `{CHECKPOINT.as_posix()}` was loaded read-only.

## Pairing check (before evaluation)

- Resampling seed 42 twice produced **identical** support/query index sets ({NUM_EPISODES} episodes).
- Seed 42 vs 43 differed in **{pairing_report['n_episodes_different_42_vs_43']}/{NUM_EPISODES}** episodes.
- For every reported seed, the same index list was used for both encoders.

## Protocol

OGB BACE scaffold **test** only; 2-way 5-shot 5-query; frozen encoder; Euclidean ProtoNet; {NUM_EPISODES} episodes per seed; seeds {SEEDS}.

GraphCL weights: core MolPCBA checkpoint (fixed).  
No-GraphCL: random `GINEncoder` init with the same seed as episode sampling (seed reset before init, independent of the episode RNG stream after sampling).

## Per-seed results

{chr(10).join(table_lines)}

## Across seeds (n={across['n_seeds']}; sample std, ddof=1)

| condition | mean of seed-means | std of seed-means |
|---|---:|---:|
| GraphCL | {across['graphcl_mean_across_seeds']:.6f} | {across['graphcl_std_across_seeds']:.6f} |
| No-GraphCL | {across['no_graphcl_mean_across_seeds']:.6f} | {across['no_graphcl_std_across_seeds']:.6f} |

Mean difference (GraphCL − No-GraphCL): **{across['mean_difference_graphcl_minus_no_graphcl']:+.6f}** ± {across['std_difference_graphcl_minus_no_graphcl']:.6f}

Seeds with GraphCL higher: {across['n_seeds_graphcl_higher']}/{across['n_seeds']}.  
Seeds with No-GraphCL higher: {across['n_seeds_no_graphcl_higher']}/{across['n_seeds']}.

This is a **robustness analysis**. It does not by itself establish a general claim that GraphCL helps or hurts few-shot BACE accuracy.

## Limitations

- Five episode seeds only; one GraphCL pretraining run.
- No-GraphCL BatchNorm running stats remain at defaults.
- Original core-run episodes (sampled after GraphCL training RNG) are a different draw than seed-42 here.
"""
    (OUT_DIR / "MULTI_SEED_SUMMARY.md").write_text(md, encoding="utf-8")
    print("=" * 50)
    print(
        f"Across seeds: GraphCL {across['graphcl_mean_across_seeds']:.4f} ± "
        f"{across['graphcl_std_across_seeds']:.4f} | No-GraphCL "
        f"{across['no_graphcl_mean_across_seeds']:.4f} ± "
        f"{across['no_graphcl_std_across_seeds']:.4f}"
    )
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
