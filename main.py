"""
Canonical pipeline:

    ogbg-molpcba graphs (labels unused)
            -> GraphCL (NT-Xent) on GINEncoder
            -> graphcl_checkpoint.pt
            -> reload GINEncoder
            -> 2-way 5-shot ProtoNet on ogbg-molbace TEST split only
            -> prototype-based episode explanation

BACE is never used for encoder pretraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import Subset
from torch_geometric.data import Batch as PyGBatch

from src.data_utils import (
    EpisodeSampler,
    GraphAugmentation,
    allow_pyg_torch_load,
    get_bace_held_out_protocol,
)
from src.explainability import PrototypeExplainer
from src.models import GINEncoder, PrototypicalNetwork
from src.training import GraphCLTrainer, ProtoNetEvaluator

allow_pyg_torch_load()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="GraphCL + ProtoNet core pipeline")
    parser.add_argument("--smoke", action="store_true", help="Tiny run; not a paper result")
    parser.add_argument("--output-dir", type=str, default="outputs/core_pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool-type", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-pretrain-graphs", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=2)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument(
        "--pretrain-root", type=str, default="data/ogb", help="OGB root for molpcba"
    )
    parser.add_argument(
        "--bace-root", type=str, default="dataset", help="OGB root for molbace"
    )
    parser.add_argument("--skip-graphcl", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def load_molpcba(root: str, max_graphs=None):
    print("Loading unlabeled pretraining graphs: ogbg-molpcba")
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=root)
    if dataset[0].x is None:
        raise RuntimeError("ogbg-molpcba is missing node features.")
    n_full = len(dataset)
    if max_graphs is not None:
        n = min(int(max_graphs), n_full)
        dataset_used = Subset(dataset, list(range(n)))
    else:
        n = n_full
        dataset_used = dataset
    print(f"  molpcba graphs available={n_full}, used_for_graphcl={n}")
    print("  labels are ignored (unlabeled GraphCL).")
    return dataset, dataset_used, n_full, n


def verify_no_bace_in_pretrain(pretrain_name: str) -> None:
    if "bace" in pretrain_name.lower():
        raise RuntimeError("Refusing to pretrain on BACE.")


def reload_and_check(checkpoint_path: Path, probe_graph, device):
    trained = GINEncoder.from_checkpoint(str(checkpoint_path), map_location=device)
    trained = trained.to(device)
    trained.eval()
    probe = probe_graph.clone().to(device)
    if getattr(probe, "batch", None) is None:
        probe.batch = torch.zeros(probe.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        _, emb_a, _ = trained(probe)

    reloaded = GINEncoder.from_checkpoint(str(checkpoint_path), map_location=device)
    reloaded = reloaded.to(device)
    reloaded.eval()
    with torch.no_grad():
        _, emb_b, _ = reloaded(probe)

    max_diff = float((emb_a - emb_b).abs().max().item())
    if max_diff > 1e-5:
        raise RuntimeError(f"Checkpoint reload mismatch: max abs diff={max_diff}")
    print(f"Checkpoint reload verified (max abs embedding diff={max_diff:.2e})")
    return reloaded


def run(args):
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.max_pretrain_graphs = args.max_pretrain_graphs or 64
        args.num_episodes = 1
        if args.output_dir == "outputs/core_pipeline":
            args.output_dir = "outputs/core_pipeline/smoke"
        print("SMOKE TEST: not a paper result. Tiny GraphCL subset + 1 episode.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "graphcl_checkpoint.pt"
    history_csv = output_dir / "graphcl_training_history.csv"
    fewshot_csv = output_dir / "fewshot_results.csv"
    fewshot_summary_path = output_dir / "fewshot_summary.json"
    config_path = output_dir / "experiment_config.json"

    verify_no_bace_in_pretrain("ogbg-molpcba")

    config = {
        "smoke": bool(args.smoke),
        "is_paper_result": False if args.smoke else True,
        "seed": args.seed,
        "device": str(device),
        "encoder": {
            "class": "GINEncoder",
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pool_type": args.pool_type,
        },
        "graphcl": {
            "method": "GraphCL",
            "loss": "NT-Xent",
            "dataset": "ogbg-molpcba",
            "uses_labels": False,
            "excludes_bace": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "max_pretrain_graphs": args.max_pretrain_graphs,
            "root": args.pretrain_root,
        },
        "fewshot": {
            "dataset": "ogbg-molbace",
            "split": "ogb_scaffold_test_only",
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "q_query": args.q_query,
            "num_episodes": args.num_episodes,
            "distance": "euclidean",
            "root": args.bace_root,
        },
        "historical_results_not_used": [
            "outputs/final_results.txt mean 0.5180 (MSE-to-noise diagnostic)",
            "explainability 9/10 single episode (diagnostic)",
        ],
        "paths": {
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint_path),
        },
    }
    write_json(config_path, config)

    if args.skip_graphcl:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        encoder = GINEncoder.from_checkpoint(str(checkpoint_path), map_location=device)
        encoder = encoder.to(device)
        print(f"Loaded existing GraphCL checkpoint: {checkpoint_path}")
    else:
        full_pcba, pcba_for_train, n_full, n_used = load_molpcba(
            args.pretrain_root, max_graphs=args.max_pretrain_graphs
        )
        config["graphcl"]["n_molpcba_available"] = n_full
        config["graphcl"]["n_molpcba_used"] = n_used
        in_dim = int(full_pcba[0].x.size(1))
        config["encoder"]["in_dim"] = in_dim

        encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pool_type=args.pool_type,
        ).to(device)
        print(f"Canonical GINEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

        trainer = GraphCLTrainer(
            encoder=encoder,
            augmentation_fn=GraphAugmentation.random_augment,
            device=device,
            temperature=args.temperature,
        )
        trainer.pretrain(
            dataset=pcba_for_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=str(checkpoint_path),
            save_interval=max(args.epochs, 1),
            history_csv_path=str(history_csv),
            extra_checkpoint_meta={
                "seed": args.seed,
                "pretrain_dataset": "ogbg-molpcba",
                "pretrain_uses_labels": False,
                "bace_used_in_pretrain": False,
                "n_pretrain_graphs": n_used,
                "smoke": bool(args.smoke),
            },
        )
        encoder = reload_and_check(checkpoint_path, full_pcba[0], device)

    print("Loading held-out few-shot dataset: ogbg-molbace")
    bace = PygGraphPropPredDataset(name="ogbg-molbace", root=args.bace_root)
    if bace[0].x is None:
        raise RuntimeError("ogbg-molbace is missing node features.")
    if int(bace[0].x.size(1)) != encoder.in_dim:
        raise RuntimeError(
            f"Feature dim mismatch: encoder in_dim={encoder.in_dim}, "
            f"BACE x={bace[0].x.size(1)}"
        )

    protocol = get_bace_held_out_protocol(bace)
    forbidden = protocol["train_reserved"] + protocol["valid_reserved"]
    config["fewshot"]["protocol"] = {
        k: protocol[k]
        for k in protocol
        if k not in ("train_reserved", "valid_reserved", "fewshot_pool")
    }
    config["fewshot"]["n_train_reserved"] = protocol["n_train_reserved"]
    config["fewshot"]["n_valid_reserved"] = protocol["n_valid_reserved"]
    config["fewshot"]["n_fewshot_pool"] = protocol["n_fewshot_pool"]
    write_json(config_path, config)

    sampler = EpisodeSampler(
        bace,
        device=device,
        allowed_indices=protocol["fewshot_pool"],
        forbidden_indices=forbidden,
    )

    protonet = PrototypicalNetwork(distance_metric="euclidean", temperature=1.0)
    evaluator = ProtoNetEvaluator(encoder=encoder, protonet=protonet, device=device)
    results = evaluator.evaluate(
        episode_sampler=sampler,
        num_episodes=args.num_episodes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
    )

    with fewshot_csv.open("w", newline="") as f:
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

    explainer = PrototypeExplainer(encoder, protonet, device=device)
    first = results["episodes"][0]

    support_graphs = [bace[i] for i in first["support_indices"]]
    query_graphs = [bace[i] for i in first["query_indices"]]
    n_support = len(first["support_indices"])
    n_query = len(first["query_indices"])
    k = args.k_shot
    q = args.q_query
    support_labels = torch.tensor(
        [i // k for i in range(n_support)], dtype=torch.long
    )
    query_labels = torch.tensor(
        [i // q for i in range(n_query)], dtype=torch.long
    )
    explanation = explainer.explain_episode(
        PyGBatch.from_data_list(support_graphs),
        support_labels,
        PyGBatch.from_data_list(query_graphs),
        query_labels,
        n_way=args.n_way,
        query_indices=first["query_indices"],
        support_indices=first["support_indices"],
    )

    summary = {
        "smoke": bool(args.smoke),
        "is_paper_result": False if args.smoke else True,
        "note": (
            "Smoke metrics are pipeline checks only."
            if args.smoke
            else "Few-shot evaluation of GraphCL GINEncoder on BACE test split."
        ),
        "seed": args.seed,
        "mean_acc": results["mean_acc"],
        "std_acc": results["std_acc"],
        "mean_loss": results["mean_loss"],
        "num_episodes": results["num_episodes"],
        "split": "ogb_scaffold_test_only",
        "do_not_use_historical_0_5180": True,
        "do_not_use_historical_9_of_10_episode": True,
        "first_episode_prototype_explanation": explanation,
        "checkpoint": str(checkpoint_path),
    }
    write_json(fewshot_summary_path, summary)
    write_json(config_path, config)

    print("=" * 50)
    print("FEW-SHOT RESULTS" + (" (SMOKE)" if args.smoke else ""))
    print("=" * 50)
    print(
        f"Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f} "
        f"over {results['num_episodes']} episodes"
    )
    print(f"Outputs: {output_dir}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
