"""Tests for Phase 3 multi-shot BACE evaluation helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
ORIGINAL_REPORT = (
    REPO_ROOT / "apps" / "mol_cli" / "evaluation_reports" / "bace_fixed_support_eval_v1.json"
)


def _ready() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401
        from apps.mol_cli.bace_support import bace_dataset_inventory_available

        return CHECKPOINT.is_file() and bace_dataset_inventory_available()
    except ImportError:
        return False


@unittest.skipUnless(_ready(), "checkpoint / BACE inventory / deps unavailable")
class TestFewshotBaceMultishot(unittest.TestCase):
    def test_support_query_disjoint_and_seed_deterministic(self):
        from apps.mol_cli.bace_support import load_bace_split_inventory, select_train_support

        inv = load_bace_split_inventory()
        a = select_train_support(inv, k_shot=5, seed=42)
        b = select_train_support(inv, k_shot=5, seed=42)
        c = select_train_support(inv, k_shot=5, seed=7)
        idx_a = [r["dataset_index"] for r in a]
        idx_b = [r["dataset_index"] for r in b]
        idx_c = [r["dataset_index"] for r in c]
        self.assertEqual(idx_a, idx_b)
        self.assertNotEqual(idx_a, idx_c)
        support = set(idx_a)
        self.assertFalse(support & set(inv.valid_indices))
        self.assertFalse(support & set(inv.test_indices))
        self.assertTrue(support.issubset(set(inv.train_indices)))

    def test_metrics_balanced_accuracy(self):
        from apps.mol_cli.bace_metrics import compute_binary_metrics

        m = compute_binary_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, -0.2, 0.3, 0.9])
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(m["sensitivity_class_1"], 0.5)
        self.assertAlmostEqual(m["specificity_class_0"], 0.5)

    def test_canonical_support_matches_manifest(self):
        from apps.mol_cli.bace_support import load_bace_split_inventory, select_train_support

        inv = load_bace_split_inventory()
        support = select_train_support(inv, k_shot=5, seed=42)
        got = [r["dataset_index"] for r in support]
        original = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(got, original["support_dataset_indices"])


if __name__ == "__main__":
    unittest.main()
