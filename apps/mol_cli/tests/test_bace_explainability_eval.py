"""Tests for Phase 4 explainability index mapping and predeclaration."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ready() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401
        from apps.mol_cli.bace_support import bace_dataset_inventory_available

        return bace_dataset_inventory_available()
    except ImportError:
        return False


@unittest.skipUnless(_ready(), "BACE inventory / deps unavailable")
class TestBaceExplainabilityEval(unittest.TestCase):
    def test_atom_index_mapping_ethanol(self):
        from diagnostics.bace_explainability_eval import verify_atom_index_mapping

        info = verify_atom_index_mapping("CCO", 3)
        self.assertTrue(info["index_mapping_ok"])
        self.assertEqual(info["rdkit_n_atoms"], 3)
        self.assertEqual(info["graph_n_nodes"], 3)

    def test_predeclared_indices_stratified_and_in_valid(self):
        from apps.mol_cli.bace_support import load_bace_split_inventory
        from diagnostics.bace_explainability_eval import predeclare_validation_indices

        inv = load_bace_split_inventory()
        idxs = predeclare_validation_indices(inv)
        self.assertTrue(set(idxs).issubset(set(inv.valid_indices)))
        labels = [inv.labels[i] for i in idxs]
        self.assertIn(0, labels)
        self.assertIn(1, labels)
        # deterministic
        self.assertEqual(idxs, predeclare_validation_indices(inv))


if __name__ == "__main__":
    unittest.main()
