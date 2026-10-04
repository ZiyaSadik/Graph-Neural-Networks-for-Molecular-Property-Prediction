"""
Leakage-safe BACE support-manifest utilities (train split only).

Reads official OGB scaffold split CSVs and mol/label tables under
dataset/ogbg_molbace. Does not load the encoder, build prototypes, or
score queries. Does not use validation or test indices for selection.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACE_ROOT = REPO_ROOT / "dataset" / "ogbg_molbace"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "support_manifests" / "bace_train_k5_v1.json"
)

MANIFEST_ID = "bace_train_k5_v1"
DATASET_NAME = "ogbg-molbace"
SPLIT_NAME = "ogb_scaffold"
SUPPORT_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_K_SHOT = 5

LABEL_SEMANTICS = {
    "0": "inactive (non-inhibitor)",
    "1": "active (BACE-1 inhibitor)",
}

SELECTION_RULE = (
    "stratified_random_sample_without_replacement_from_official_OGB_"
    "scaffold_train_split_only; fixed seed; K examples per binary class; "
    "validation and test indices are never eligible"
)


class SupportManifestError(ValueError):
    """Invalid support selection or manifest."""


@dataclass(frozen=True)
class SplitInventory:
    train_indices: list[int]
    valid_indices: list[int]
    test_indices: list[int]
    labels: list[int]
    smiles: list[str]
    mol_ids: list[str]
    train_class_counts: dict[int, int]
    valid_class_counts: dict[int, int]
    test_class_counts: dict[int, int]


def _read_split_indices(path: Path) -> list[int]:
    import pandas as pd

    df = pd.read_csv(path, compression="gzip", header=None)
    return [int(v) for v in df.iloc[:, 0].tolist()]


def bace_dataset_inventory_available(
    bace_root: Path | str = DEFAULT_BACE_ROOT,
) -> bool:
    """True when the local OGB BACE raw/split/mapping files needed for inventory exist."""
    root = Path(bace_root)
    split_dir = root / "split" / "scaffold"
    required = [
        split_dir / "train.csv.gz",
        split_dir / "valid.csv.gz",
        split_dir / "test.csv.gz",
        root / "raw" / "graph-label.csv.gz",
        root / "mapping" / "mol.csv.gz",
    ]
    return all(p.is_file() for p in required)


def load_bace_split_inventory(bace_root: Path | str = DEFAULT_BACE_ROOT) -> SplitInventory:
    """
    Load official OGB scaffold splits + labels + SMILES from cached files.

    Does not download. Does not unpickle geometric_data_processed.pt.
    """
    import pandas as pd

    root = Path(bace_root)
    split_dir = root / "split" / "scaffold"
    required = [
        split_dir / "train.csv.gz",
        split_dir / "valid.csv.gz",
        split_dir / "test.csv.gz",
        root / "raw" / "graph-label.csv.gz",
        root / "mapping" / "mol.csv.gz",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SupportManifestError(
            "Local ogbg-molbace cache incomplete; refusing to download. "
            f"Missing: {missing}"
        )

    train = _read_split_indices(split_dir / "train.csv.gz")
    valid = _read_split_indices(split_dir / "valid.csv.gz")
    test = _read_split_indices(split_dir / "test.csv.gz")

    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise SupportManifestError("OGB BACE splits are not disjoint.")

    labels_df = pd.read_csv(
        root / "raw" / "graph-label.csv.gz", compression="gzip", header=None
    )
    mol_df = pd.read_csv(root / "mapping" / "mol.csv.gz", compression="gzip")
    if len(labels_df) != len(mol_df):
        raise SupportManifestError(
            f"Label/SMILES length mismatch: {len(labels_df)} vs {len(mol_df)}"
        )

    labels = [int(v) for v in labels_df.iloc[:, 0].tolist()]
    classes = [int(v) for v in mol_df["Class"].tolist()]
    if labels != classes:
        raise SupportManifestError(
            "mol.csv.gz Class does not match raw/graph-label.csv.gz."
        )
    if any(y not in (0, 1) for y in labels):
        raise SupportManifestError("Non-binary BACE labels found.")

    smiles = [str(s) for s in mol_df["smiles"].tolist()]
    mol_ids = [str(m) for m in mol_df["mol_id"].tolist()]

    def _counts(indices: list[int]) -> dict[int, int]:
        out = {0: 0, 1: 0}
        for i in indices:
            out[labels[i]] += 1
        return out

    return SplitInventory(
        train_indices=train,
        valid_indices=valid,
        test_indices=test,
        labels=labels,
        smiles=smiles,
        mol_ids=mol_ids,
        train_class_counts=_counts(train),
        valid_class_counts=_counts(valid),
        test_class_counts=_counts(test),
    )


def _canonical_smiles_or_fail(smiles: str, dataset_index: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SupportManifestError(
            f"Dataset index {dataset_index}: SMILES does not parse: {smiles!r}"
        )
    if mol.GetNumAtoms() <= 0:
        raise SupportManifestError(
            f"Dataset index {dataset_index}: parsed SMILES has zero atoms."
        )
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:
        raise SupportManifestError(
            f"Dataset index {dataset_index}: cannot canonicalize SMILES ({exc})."
        ) from None


def select_train_support(
    inventory: SplitInventory,
    k_shot: int = DEFAULT_K_SHOT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """
    Deterministic stratified sample: K train indices per class {0,1}.
    """
    if k_shot <= 0:
        raise SupportManifestError("k_shot must be positive.")

    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx in inventory.train_indices:
        by_class[inventory.labels[idx]].append(int(idx))

    for cls in (0, 1):
        by_class[cls] = sorted(by_class[cls])
        if len(by_class[cls]) < k_shot:
            raise SupportManifestError(
                f"Train class {cls} has only {len(by_class[cls])} molecules; "
                f"need {k_shot}."
            )

    forbidden = set(inventory.valid_indices) | set(inventory.test_indices)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for cls in (0, 1):
        chosen = rng.sample(by_class[cls], k_shot)
        for idx in sorted(chosen):
            if idx in forbidden:
                raise SupportManifestError(
                    f"Selection leaked non-train index {idx}."
                )
            if inventory.labels[idx] != cls:
                raise SupportManifestError(
                    f"Index {idx} expected label {cls}, found {inventory.labels[idx]}."
                )
            original = inventory.smiles[idx]
            canonical = _canonical_smiles_or_fail(original, idx)
            selected.append(
                {
                    "dataset_index": int(idx),
                    "label": int(cls),
                    "smiles_original": original,
                    "smiles_canonical": canonical,
                    "mol_id": inventory.mol_ids[idx],
                }
            )

    indices = [row["dataset_index"] for row in selected]
    if len(indices) != len(set(indices)):
        raise SupportManifestError("Duplicate support indices after selection.")
    if len(selected) != 2 * k_shot:
        raise SupportManifestError("Unexpected support size.")

    return selected


def build_manifest_dict(
    inventory: SplitInventory,
    k_shot: int = DEFAULT_K_SHOT,
    seed: int = DEFAULT_SEED,
    created_utc: str | None = None,
) -> dict[str, Any]:
    support = select_train_support(inventory, k_shot=k_shot, seed=seed)
    return {
        "manifest_id": MANIFEST_ID,
        "dataset": DATASET_NAME,
        "split": SUPPORT_SPLIT,
        "split_name": SPLIT_NAME,
        "k_shot": int(k_shot),
        "n_way": 2,
        "seed": int(seed),
        "selection_rule": SELECTION_RULE,
        "label_semantics": LABEL_SEMANTICS,
        "label_source": "dataset/ogbg_molbace/raw/graph-label.csv.gz and mapping/mol.csv.gz Class",
        "created_utc": created_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_root_relative": "dataset/ogbg_molbace",
        "n_train": len(inventory.train_indices),
        "n_valid": len(inventory.valid_indices),
        "n_test": len(inventory.test_indices),
        "train_class_counts": {
            "0": inventory.train_class_counts[0],
            "1": inventory.train_class_counts[1],
        },
        "support": support,
    }


def validate_manifest_for_inference(manifest: dict[str, Any]) -> None:
    """
    Self-contained manifest checks for interactive BACE inference.

    Validates schema, K-shot balance, labels, SMILES parseability, and
    canonical-SMILES consistency from the committed manifest alone.

    Does **not** independently verify that support indices belong to the
    official OGB training split (that requires ``validate_manifest`` with a
    local dataset inventory). Does not download data.
    """
    required_top = [
        "manifest_id",
        "dataset",
        "split",
        "split_name",
        "k_shot",
        "seed",
        "selection_rule",
        "support",
    ]
    for key in required_top:
        if key not in manifest:
            raise SupportManifestError(f"Manifest missing field {key!r}.")

    if manifest["dataset"] != DATASET_NAME:
        raise SupportManifestError("Unexpected dataset name.")
    if manifest["split"] != SUPPORT_SPLIT:
        raise SupportManifestError("Support split must be 'train'.")

    support = manifest["support"]
    if not isinstance(support, list) or not support:
        raise SupportManifestError("support must be a non-empty list.")

    k_shot = int(manifest["k_shot"])
    if k_shot <= 0:
        raise SupportManifestError("k_shot must be positive.")
    if len(support) != 2 * k_shot:
        raise SupportManifestError(
            f"Expected {2 * k_shot} support rows, found {len(support)}."
        )

    seen: set[int] = set()
    class_counts = {0: 0, 1: 0}

    for row in support:
        for field in (
            "dataset_index",
            "label",
            "smiles_original",
            "smiles_canonical",
        ):
            if field not in row:
                raise SupportManifestError(f"Support row missing {field!r}.")

        idx = int(row["dataset_index"])
        label = int(row["label"])
        if label not in (0, 1):
            raise SupportManifestError(f"Invalid label {label} at index {idx}.")
        if idx in seen:
            raise SupportManifestError(f"Duplicate dataset_index {idx}.")
        seen.add(idx)

        canonical = _canonical_smiles_or_fail(row["smiles_original"], idx)
        if canonical != row["smiles_canonical"]:
            raise SupportManifestError(
                f"Index {idx}: smiles_canonical mismatch "
                f"(got {row['smiles_canonical']!r}, expected {canonical!r})."
            )
        # Canonical of stored canonical must still parse
        _canonical_smiles_or_fail(row["smiles_canonical"], idx)

        class_counts[label] += 1

    if class_counts[0] != k_shot or class_counts[1] != k_shot:
        raise SupportManifestError(
            f"Expected {k_shot} per class, found {class_counts}."
        )


def validate_manifest(
    manifest: dict[str, Any],
    inventory: SplitInventory | None = None,
) -> None:
    """
    Full validation including local OGB inventory leakage / label / SMILES checks.

    Always requires a dataset inventory (loads the local cache when ``inventory``
    is omitted). Use ``validate_manifest_for_inference`` for interactive scoring
    without that cache.
    """
    validate_manifest_for_inference(manifest)

    if inventory is None:
        inventory = load_bace_split_inventory()

    support = manifest["support"]
    train_set = set(inventory.train_indices)
    valid_set = set(inventory.valid_indices)
    test_set = set(inventory.test_indices)

    for row in support:
        idx = int(row["dataset_index"])
        label = int(row["label"])

        if idx not in train_set:
            raise SupportManifestError(
                f"Support index {idx} is not in the official training split."
            )
        if idx in valid_set or idx in test_set:
            raise SupportManifestError(
                f"Support index {idx} belongs to validation/test (leakage)."
            )

        if inventory.labels[idx] != label:
            raise SupportManifestError(
                f"Index {idx}: manifest label {label} != dataset label "
                f"{inventory.labels[idx]}."
            )
        if inventory.smiles[idx] != row["smiles_original"]:
            raise SupportManifestError(
                f"Index {idx}: smiles_original does not match mol.csv.gz."
            )


def write_manifest(
    manifest: dict[str, Any],
    path: Path | str = DEFAULT_MANIFEST_PATH,
    inventory: SplitInventory | None = None,
) -> Path:
    path = Path(path)
    if path.exists():
        raise SupportManifestError(
            f"Manifest already exists; refusing to overwrite: {path}"
        )
    validate_manifest(manifest, inventory=inventory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise SupportManifestError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
