"""Tests for Phase 2 GIN embedding validation diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"


def _deps_ok() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(CHECKPOINT.is_file(), f"checkpoint missing: {CHECKPOINT}")
@unittest.skipUnless(_deps_ok(), "torch / torch_geometric / ogb unavailable")
class TestGinEmbeddingValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diagnostics.gin_embedding_validation import run_validation

        cls.report = run_validation(CHECKPOINT)

    def test_overall_pass(self):
        self.assertTrue(self.report["overall_pass"], self.report.get("failures"))

    def test_checkpoint_hash_recorded(self):
        self.assertEqual(len(self.report["checkpoint"]["sha256"]), 64)
        self.assertTrue(self.report["checkpoint"]["param_count_matches"])

    def test_node_and_graph_shapes(self):
        for row in self.report["molecules"]:
            self.assertEqual(row["node_emb_shape"][1], 128)
            self.assertEqual(row["node_emb_shape"][0], row["n_atoms"])
            self.assertEqual(row["graph_emb_shape"], [1, 128])

    def test_batch_shape(self):
        n = len(self.report["molecules"])
        self.assertEqual(self.report["batch"]["graph_emb_shape"], [n, 128])

    def test_finite_and_repeatable(self):
        self.assertTrue(self.report["checks"]["all_finite"])
        self.assertTrue(self.report["checks"]["eval_repeatable"])
        for row in self.report["molecules"]:
            self.assertEqual(row["eval_repeat_max_abs_delta_graph"], 0.0)

    def test_not_constant_across_molecules(self):
        self.assertTrue(self.report["checks"]["not_constant_across_molecules"])
        self.assertGreater(self.report["cross_molecule"]["min_pairwise_l2"], 1e-4)

    def test_interpretation_limits_documented(self):
        lim = self.report["interpretation_limits"]
        self.assertTrue(lim["implementation_checks_only"])
        self.assertTrue(lim["nonzero_distance_is_not_chemical_proof"])


if __name__ == "__main__":
    unittest.main()
