"""Recompute Phase 4E.2 descriptor baseline metrics from the frozen ESOL split."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ESOL = REPO / "data" / "moleculenet" / "esol"
SRC = ESOL / "delaney-processed.csv"
ASSIGN = ESOL / "esol_split_assignments_v1.csv"
RESULTS = HERE / "results.json"
TARGET = "measured log solubility in mols per litre"
EXPECTED_SRC = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"
EXPECTED_MAP = "90f827da4c8622a85408b461b0e028a4cc882439c01c54d06d5bc08db92602ac"


def main() -> int:
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))
    names = saved["descriptors"]["names"]
    alpha = saved["model"]["parameters"]["Ridge"]["alpha"]
    seed = saved["model"]["parameters"]["Ridge"]["random_state"]

    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    if src_sha != EXPECTED_SRC:
        print("FAIL source checksum", file=sys.stderr)
        return 1
    src_rows = list(csv.DictReader(SRC.read_text(encoding="utf-8").splitlines()))
    asn_rows = list(csv.DictReader(ASSIGN.read_text(encoding="utf-8").splitlines()))
    pairs = sorted(((int(r["row_index"]), r["split"]) for r in asn_rows), key=lambda x: x[0])
    map_fp = hashlib.sha256("".join(f"{i}\t{s}\n" for i, s in pairs).encode("utf-8")).hexdigest()
    if map_fp != EXPECTED_MAP:
        print("FAIL mapping fingerprint", map_fp, file=sys.stderr)
        return 1

    def featurize(smi: str):
        mol = Chem.MolFromSmiles(smi)
        vals = []
        for name in names:
            v = float(getattr(Descriptors, name)(mol))
            if not math.isfinite(v):
                raise RuntimeError(f"nonfinite {name}")
            vals.append(v)
        return vals

    Xtr, ytr, Xva, yva = [], [], [], []
    for i, (src, asn) in enumerate(zip(src_rows, asn_rows)):
        if asn["split"] == "test":
            continue
        x = featurize(src["smiles"].strip())
        y = float(src[TARGET])
        if asn["split"] == "train":
            Xtr.append(x); ytr.append(y)
        elif asn["split"] == "valid":
            Xva.append(x); yva.append(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=seed)),
    ])
    model.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64))
    pred = model.predict(np.asarray(Xva, dtype=np.float64))
    yva = np.asarray(yva, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - yva)))
    rmse = float(np.sqrt(np.mean((pred - yva) ** 2)))
    ok = (
        abs(mae - saved["evaluation"]["metrics"]["mae"]) < 1e-10
        and abs(rmse - saved["evaluation"]["metrics"]["rmse"]) < 1e-10
    )
    print("valid_mae", mae)
    print("valid_rmse", rmse)
    print("matches_results_json", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
