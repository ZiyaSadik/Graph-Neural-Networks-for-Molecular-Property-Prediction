"""Tests for chemist-facing analysis service (reuses mol_cli models)."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"


def _ready() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401

        return CKPT.is_file()
    except ImportError:
        return False


@unittest.skipUnless(_ready(), "deps / checkpoint unavailable")
class TestChemistAppService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apps.chemist_app.service import ChemistAnalysisService

        cls.svc = ChemistAnalysisService()

    def test_invalid_smiles(self):
        r = self.svc.analyze("not_a_smiles!!!", run_explanation=False)
        self.assertFalse(r.valid)
        self.assertIsNotNone(r.error)

    def test_empty_smiles(self):
        r = self.svc.analyze("   ", run_explanation=False)
        self.assertFalse(r.valid)

    def test_valid_ethanol_report_structure(self):
        r = self.svc.analyze("CCO", run_explanation=False)
        self.assertTrue(r.valid)
        self.assertEqual(r.canonical_smiles, "CCO")
        self.assertTrue(r.descriptors_are_calculated_not_learned)
        self.assertIn("molecular_weight", r.descriptors or {})
        self.assertIsNotNone(r.structure_png)
        self.assertTrue(r.reliability.get("no_untrained_substitutes"))

        if r.bace and r.bace.get("available"):
            self.assertFalse(r.bace["calibrated_probability"])
            self.assertIn("distance_to_class_0_inactive", r.bace)
            self.assertNotIn("probability", r.bace)

        if r.esol and r.esol.get("available"):
            self.assertIn("predicted_logS", r.esol)
            self.assertIn("mols per litre", r.esol["target_scale"])

        if r.tox21 and r.tox21.get("available"):
            self.assertEqual(len(r.tox21["tasks"]), 12)
            self.assertFalse(r.tox21["calibrated_probability"])

    def test_status_probe_lists_checkpoints(self):
        self.assertIn("graphcl_encoder", self.svc.status)
        self.assertTrue(self.svc.status["graphcl_encoder"].available)


@unittest.skipUnless(_ready(), "deps / checkpoint unavailable")
class TestChemistDrawingAndExplain(unittest.TestCase):
    def test_atom_index_png(self):
        from apps.chemist_app.drawing import mol_to_png_with_atom_indices
        from apps.mol_cli.smiles_input import parse_smiles

        mol = parse_smiles("CCO").mol
        png = mol_to_png_with_atom_indices(mol)
        self.assertGreater(len(png), 100)
        self.assertTrue(png[:8] == b"\x89PNG\r\n\x1a\n")

    def test_importance_rejects_misaligned_mask(self):
        from apps.chemist_app.drawing import mol_to_png_with_atom_importance
        from apps.mol_cli.smiles_input import parse_smiles
        import numpy as np

        mol = parse_smiles("CCO").mol
        with self.assertRaises(ValueError):
            mol_to_png_with_atom_importance(mol, np.array([0.1, 0.2]))

    def test_explain_index_mapping_helper(self):
        from apps.chemist_app.explain import verify_index_mapping

        self.assertTrue(verify_index_mapping("CCO", 3))


if __name__ == "__main__":
    unittest.main()
