"""Evaluate Phase 4E.3 frozen-GIN ESOL regression head on the validation split (read-only)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from apps.mol_cli.checkpoint_smoke import load_encoder_from_research_checkpoint
from apps.mol_cli.ogb_graph import smiles_to_pyg_data

ESOL = REPO / "data" / "moleculenet" / "esol"
SRC = ESOL / "delaney-processed.csv"
ASSIGN = ESOL / "esol_split_assignments_v1.csv"
CKPT = REPO / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
HEAD = HERE / "regression_head.pt"
RESULTS = HERE / "results.json"
TARGET = "measured log solubility in mols per litre"
EXPECTED_SRC = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"
EXPECTED_MAP = "90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac"
EXPECTED_CKPT = "d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f"


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> int:
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))

    if hashlib.sha256(SRC.read_bytes()).hexdigest() != EXPECTED_SRC:
        print("FAIL source checksum", file=sys.stderr)
        return 1
    if hashlib.sha256(CKPT.read_bytes()).hexdigest() != EXPECTED_CKPT:
        print("FAIL checkpoint checksum", file=sys.stderr)
        return 1

    asn = list(csv.DictReader(ASSIGN.read_text(encoding="utf-8").splitlines()))
    pairs = sorted(((int(r["row_index"]), r["split"]) for r in asn), key=lambda x: x[0])
    map_fp = hashlib.sha256(
        "".join(f"{i}\t{s}\n" for i, s in pairs).encode("utf-8")
    ).hexdigest()
    if map_fp != EXPECTED_MAP:
        print("FAIL mapping fingerprint", map_fp, file=sys.stderr)
        return 1

    src_rows = list(csv.DictReader(SRC.read_text(encoding="utf-8").splitlines()))
    graphs = []
    y_list = []
    for src, a in zip(src_rows, asn):
        if a["split"] != "valid":
            continue
        graphs.append(smiles_to_pyg_data(src["smiles"].strip()).clone())
        y_list.append(float(src[TARGET]))

    enc, _ = load_encoder_from_research_checkpoint(CKPT, map_location="cpu")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        emb = enc(Batch.from_data_list(graphs))[1]

    # Head payload is a plain dict of tensors / primitives (no custom classes).
    payload = torch.load(HEAD, map_location="cpu", weights_only=True)
    head = RegressionHead(
        in_dim=payload["head_config"]["in_dim"],
        hidden=payload["head_config"]["hidden"],
    )
    head.load_state_dict(payload["head_state_dict"])
    head.eval()

    y_mean = payload["target_normalization"]["train_mean"]
    y_std = payload["target_normalization"]["train_std_population"]
    with torch.no_grad():
        pred = head(emb) * y_std + y_mean
        y = torch.tensor(y_list, dtype=torch.float32)
        err = pred - y
        mae = float(torch.mean(torch.abs(err)))
        rmse = float(torch.sqrt(torch.mean(err ** 2)))

    ok = (
        abs(mae - saved["evaluation"]["metrics"]["mae"]) < 1e-5
        and abs(rmse - saved["evaluation"]["metrics"]["rmse"]) < 1e-5
    )
    print("valid_mae", mae)
    print("valid_rmse", rmse)
    print("matches_results_json", ok)
    print("encoder_frozen", not any(p.requires_grad for p in enc.parameters()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
