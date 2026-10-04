"""Recompute Phase 4F.4 Tox21 validation baselines (read-only; no test metrics)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from apps.mol_cli.checkpoint_smoke import load_encoder_from_research_checkpoint
from apps.mol_cli.ogb_graph import smiles_to_pyg_data

DIR = REPO / "data" / "moleculenet" / "tox21"
SOURCE = DIR / "tox21.csv"
ASSIGN = DIR / "tox21_split_assignments_v1.csv"
CKPT = REPO / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
HEAD = HERE / "classification_head.pt"
RESULTS = HERE / "results.json"

EXPECTED_CSV = "7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761"
EXPECTED_MAP = "bffa1dbba4729aadc8bd5e19c052f0982c8b81a64b58e129175fc0d85d463ebd"
EXPECTED_CKPT = "d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f"
TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim=128, hidden=64, n_tasks=12, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x):
        return self.net(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_missing(v: str) -> bool:
    s = (v or "").strip()
    return s == "" or s.lower() in {"nan", "none", "na", "null"}


def parse_label(v: str):
    if is_missing(v):
        return None
    x = float(v)
    if abs(x - 0.0) < 1e-12:
        return 0
    if abs(x - 1.0) < 1e-12:
        return 1
    raise ValueError(v)


def main() -> int:
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))
    if sha256_file(SOURCE) != EXPECTED_CSV:
        print("FAIL source checksum", file=sys.stderr)
        return 1
    if sha256_file(CKPT) != EXPECTED_CKPT:
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

    src_rows = list(csv.DictReader(SOURCE.read_text(encoding="utf-8").splitlines()))
    valid = []
    for src, a in zip(src_rows, asn):
        if a["split"] != "valid":
            continue
        if str(a["graph_ineligible"]).lower() in {"true", "1"}:
            continue
        labels = [parse_label(src.get(t, "")) for t in TASKS]
        g = smiles_to_pyg_data(src["smiles"].strip()).clone()
        valid.append((g, labels))

    enc, _ = load_encoder_from_research_checkpoint(CKPT, map_location="cpu")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        emb = enc(Batch.from_data_list([g for g, _ in valid]))[1]

    payload = torch.load(HEAD, map_location="cpu", weights_only=True)
    cfg = payload["head_config"]
    head = MultiTaskHead(
        in_dim=cfg["in_dim"],
        hidden=cfg["hidden"],
        n_tasks=cfg["n_tasks"],
        dropout=cfg["dropout"],
    )
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(emb).cpu().numpy()

    y = np.zeros((len(valid), len(TASKS)), dtype=np.float64)
    m = np.zeros((len(valid), len(TASKS)), dtype=np.float64)
    for i, (_, labels) in enumerate(valid):
        for ti, lab in enumerate(labels):
            if lab is None:
                continue
            y[i, ti] = lab
            m[i, ti] = 1.0

    aucs = []
    for ti, t in enumerate(TASKS):
        mask = m[:, ti] > 0.5
        yt = y[mask, ti]
        sc = logits[mask, ti]
        prob = 1.0 / (1.0 + np.exp(-np.clip(sc, -50, 50)))
        saved_auc = saved["frozen_gin_multitask"]["evaluation"]["per_task"][t]["roc_auc"]
        if (yt == 1).sum() < 1 or (yt == 0).sum() < 1:
            auc = None
        else:
            auc = float(roc_auc_score(yt, prob))
            aucs.append(auc)
        if saved_auc is None:
            ok = auc is None
        else:
            ok = auc is not None and abs(auc - saved_auc) < 1e-5
        if not ok:
            print("FAIL metric mismatch", t, auc, saved_auc, file=sys.stderr)
            return 1

    macro = float(np.mean(aucs)) if aucs else None
    saved_macro = saved["frozen_gin_multitask"]["evaluation"]["macro"]["roc_auc"]
    if macro is None or abs(macro - saved_macro) >= 1e-5:
        print("FAIL macro mismatch", macro, saved_macro, file=sys.stderr)
        return 1

    print("valid_macro_roc_auc", macro)
    print("matches_results_json", True)
    print("encoder_frozen", not any(p.requires_grad for p in enc.parameters()))
    print("n_valid_graph_eligible", len(valid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
