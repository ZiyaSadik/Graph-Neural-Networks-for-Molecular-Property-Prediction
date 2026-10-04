"""Tests for leakage-safe BACE train-only support manifests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from apps.mol_cli.bace_support import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SEED,
    SupportManifestError,
    build_manifest_dict,
    load_bace_split_inventory,
    load_manifest,
    select_train_support,
    validate_manifest,
    validate_manifest_for_inference,
    write_manifest,
)

BACE_ROOT = Path(__file__).resolve().parents[3] / "dataset" / "ogbg_molbace"


def _inventory_available() -> bool:
    return (BACE_ROOT / "split" / "scaffold" / "train.csv.gz").is_file()


@unittest.skipUnless(_inventory_available(), "local ogbg-molbace cache not found")
class TestBaceSupportManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load_bace_split_inventory(BACE_ROOT)
        cls.train_set = set(cls.inventory.train_indices)
        cls.valid_set = set(cls.inventory.valid_indices)
        cls.test_set = set(cls.inventory.test_indices)

    def test_split_sizes_and_class_counts(self):
        self.assertEqual(len(self.inventory.train_indices), 1210)
        self.assertEqual(len(self.inventory.valid_indices), 151)
        self.assertEqual(len(self.inventory.test_indices), 152)
        self.assertEqual(self.inventory.train_class_counts[0], 730)
        self.assertEqual(self.inventory.train_class_counts[1], 480)
        self.assertFalse(self.train_set & self.valid_set)
        self.assertFalse(self.train_set & self.test_set)
        self.assertFalse(self.valid_set & self.test_set)

    def test_selector_is_deterministic(self):
        a = select_train_support(self.inventory, k_shot=5, seed=DEFAULT_SEED)
        b = select_train_support(self.inventory, k_shot=5, seed=DEFAULT_SEED)
        self.assertEqual(a, b)
        m1 = build_manifest_dict(
            self.inventory, created_utc="2026-01-01T00:00:00+00:00"
        )
        m2 = build_manifest_dict(
            self.inventory, created_utc="2026-01-01T00:00:00+00:00"
        )
        self.assertEqual(m1["support"], m2["support"])

    def test_selection_train_only_balanced_original_labels(self):
        support = select_train_support(self.inventory, k_shot=5, seed=DEFAULT_SEED)
        indices = [row["dataset_index"] for row in support]
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(set(indices)), 10)
        for row in support:
            idx = row["dataset_index"]
            self.assertIn(idx, self.train_set)
            self.assertNotIn(idx, self.valid_set)
            self.assertNotIn(idx, self.test_set)
            self.assertEqual(row["label"], self.inventory.labels[idx])
            self.assertIn(row["label"], (0, 1))
        self.assertEqual(sum(1 for r in support if r["label"] == 0), 5)
        self.assertEqual(sum(1 for r in support if r["label"] == 1), 5)

    def test_written_manifest_on_disk_if_present(self):
        if not DEFAULT_MANIFEST_PATH.is_file():
            self.skipTest("manifest not generated yet")
        manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        validate_manifest(manifest, inventory=self.inventory)
        for row in manifest["support"]:
            self.assertIn(row["dataset_index"], self.train_set)
            self.assertNotIn(row["dataset_index"], self.valid_set)
            self.assertNotIn(row["dataset_index"], self.test_set)

    def test_rejects_validation_or_test_index_injection(self):
        manifest = build_manifest_dict(
            self.inventory, created_utc="2026-01-01T00:00:00+00:00"
        )
        leaked = copy.deepcopy(manifest)
        leaked["support"][0]["dataset_index"] = self.inventory.valid_indices[0]
        leaked["support"][0]["label"] = self.inventory.labels[
            self.inventory.valid_indices[0]
        ]
        leaked["support"][0]["smiles_original"] = self.inventory.smiles[
            self.inventory.valid_indices[0]
        ]
        from apps.mol_cli.bace_support import _canonical_smiles_or_fail

        leaked["support"][0]["smiles_canonical"] = _canonical_smiles_or_fail(
            leaked["support"][0]["smiles_original"],
            leaked["support"][0]["dataset_index"],
        )
        with self.assertRaises(SupportManifestError):
            validate_manifest(leaked, inventory=self.inventory)

        leaked_test = copy.deepcopy(manifest)
        ti = self.inventory.test_indices[0]
        leaked_test["support"][1]["dataset_index"] = ti
        leaked_test["support"][1]["label"] = self.inventory.labels[ti]
        leaked_test["support"][1]["smiles_original"] = self.inventory.smiles[ti]
        leaked_test["support"][1]["smiles_canonical"] = _canonical_smiles_or_fail(
            self.inventory.smiles[ti], ti
        )
        with self.assertRaises(SupportManifestError):
            validate_manifest(leaked_test, inventory=self.inventory)

    def test_rejects_missing_fields_bad_labels_duplicates(self):
        manifest = build_manifest_dict(
            self.inventory, created_utc="2026-01-01T00:00:00+00:00"
        )
        missing = copy.deepcopy(manifest)
        del missing["support"]
        with self.assertRaises(SupportManifestError):
            validate_manifest(missing, inventory=self.inventory)

        bad_label = copy.deepcopy(manifest)
        bad_label["support"][0]["label"] = 2
        with self.assertRaises(SupportManifestError):
            validate_manifest(bad_label, inventory=self.inventory)

        dup = copy.deepcopy(manifest)
        dup["support"][1]["dataset_index"] = dup["support"][0]["dataset_index"]
        dup["support"][1]["label"] = dup["support"][0]["label"]
        dup["support"][1]["smiles_original"] = dup["support"][0]["smiles_original"]
        dup["support"][1]["smiles_canonical"] = dup["support"][0]["smiles_canonical"]
        with self.assertRaises(SupportManifestError):
            validate_manifest(dup, inventory=self.inventory)

    def test_write_refuses_overwrite(self):
        manifest = build_manifest_dict(
            self.inventory, created_utc="2026-01-01T00:00:00+00:00"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bace_train_k5_v1.json"
            write_manifest(manifest, path=path, inventory=self.inventory)
            with self.assertRaises(SupportManifestError):
                write_manifest(manifest, path=path, inventory=self.inventory)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["support"], manifest["support"])


class TestInferenceOnlyManifestValidation(unittest.TestCase):
    """Does not require dataset/ogbg_molbace."""

    @unittest.skipUnless(DEFAULT_MANIFEST_PATH.is_file(), "manifest missing")
    def test_committed_manifest_passes_inference_validation(self):
        manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        validate_manifest_for_inference(manifest)
        self.assertEqual(int(manifest["k_shot"]), 5)
        self.assertEqual(len(manifest["support"]), 10)

    @unittest.skipUnless(DEFAULT_MANIFEST_PATH.is_file(), "manifest missing")
    def test_inference_validation_rejects_bad_label_and_smiles(self):
        manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        bad_label = copy.deepcopy(manifest)
        bad_label["support"][0]["label"] = 3
        with self.assertRaises(SupportManifestError):
            validate_manifest_for_inference(bad_label)

        bad_smi = copy.deepcopy(manifest)
        bad_smi["support"][0]["smiles_original"] = "not_a_smiles"
        with self.assertRaises(SupportManifestError):
            validate_manifest_for_inference(bad_smi)

        mismatch = copy.deepcopy(manifest)
        mismatch["support"][0]["smiles_canonical"] = "C"
        with self.assertRaises(SupportManifestError):
            validate_manifest_for_inference(mismatch)

    @unittest.skipUnless(DEFAULT_MANIFEST_PATH.is_file(), "manifest missing")
    def test_inference_validation_rejects_unbalanced_support(self):
        manifest = copy.deepcopy(load_manifest(DEFAULT_MANIFEST_PATH))
        flipped = False
        for row in manifest["support"]:
            if int(row["label"]) == 1:
                row["label"] = 0
                flipped = True
                break
        self.assertTrue(flipped)
        with self.assertRaises(SupportManifestError) as ctx:
            validate_manifest_for_inference(manifest)
        self.assertIn("per class", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
