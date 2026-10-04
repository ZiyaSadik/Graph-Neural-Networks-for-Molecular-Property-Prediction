"""Recompute Phase 4E.1 train-mean baseline metrics from frozen ESOL split (read-only)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ESOL = REPO / "data" / "moleculenet" / "esol"
SRC = ESOL / "delaney-processed.csv"
ASSIGN = ESOL / "esol_split_assignments_v1.csv"
RESULTS = Path(__file__).resolve().parent / "results.json"
TARGET = "measured log solubility in mols per litre"
EXPECTED_MAP = "90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac"
EXPECTED_SRC = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"


def main() -> int:
    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    if src_sha != EXPECTED_SRC:
        print("FAIL source checksum", src_sha, file=sys.stderr)
        return 1
    src_rows = list(csv.DictReader(SRC.read_text(encoding="utf-8").splitlines()))
    asn_rows = list(csv.DictReader(ASSIGN.read_text(encoding="utf-8").splitlines()))
    pairs = sorted(((int(r["row_index"]), r["split"]) for r in asn_rows), key=lambda x: x[0])
    map_fp = hashlib.sha256("".join(f"{i}\t{s}\n" for i, s in pairs).encode("utf-8")).hexdigest()
    if map_fp != EXPECTED_MAP:
        print("FAIL mapping fingerprint", map_fp, file=sys.stderr)
        return 1
    train_y = []
    valid_y = []
    for i, (src, asn) in enumerate(zip(src_rows, asn_rows)):
        y = float(src[TARGET])
        if asn["split"] == "train":
            train_y.append(y)
        elif asn["split"] == "valid":
            valid_y.append(y)
        # test ignored
    mean = sum(train_y) / len(train_y)
    mae = sum(abs(mean - y) for y in valid_y) / len(valid_y)
    rmse = math.sqrt(sum((mean - y) ** 2 for y in valid_y) / len(valid_y))
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))
    ok = (
        abs(mean - saved["baseline"]["train_mean_measured_logS"]) < 1e-12
        and abs(mae - saved["evaluation"]["metrics"]["mae"]) < 1e-12
        and abs(rmse - saved["evaluation"]["metrics"]["rmse"]) < 1e-12
    )
    print("train_mean", mean)
    print("valid_mae", mae)
    print("valid_rmse", rmse)
    print("matches_results_json", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
