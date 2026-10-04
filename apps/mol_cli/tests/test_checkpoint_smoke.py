"""
Checkpoint compatibility smoke tests.

Loads the local GraphCL checkpoint with weights_only=True, builds the
research GINEncoder from the resulting dict, and embeds one CLI-built graph.

Does not train, does not evaluate BACE accuracy, does not enable CLI prediction.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(CHECKPOINT.is_file(), f"checkpoint missing: {CHECKPOINT}")
@unittest.skipUnless(_torch_available(), "torch / torch_geometric / ogb unavailable")
class TestCheckpointCompatibilitySmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apps.mol_cli.checkpoint_smoke import (
            DOCUMENTED_PARAM_COUNT,
            assert_documented_config,
            checkpoint_file_info,
            load_encoder_from_research_checkpoint,
        )

        cls.DOCUMENTED_PARAM_COUNT = DOCUMENTED_PARAM_COUNT
        cls.assert_documented_config = staticmethod(assert_documented_config)
        info = checkpoint_file_info(CHECKPOINT)
        cls.checkpoint_size = info["size_bytes"]
        cls.encoder, cls.ckpt = load_encoder_from_research_checkpoint(
            CHECKPOINT, map_location="cpu"
        )

    def test_checkpoint_file_nonempty(self):
        self.assertGreater(self.checkpoint_size, 0)

    def test_required_keys_present(self):
        self.assertIn("encoder_config", self.ckpt)
        self.assertIn("encoder_state_dict", self.ckpt)

    def test_config_matches_documentation(self):
        self.assert_documented_config(self.ckpt["encoder_config"])
        self.assertEqual(self.encoder.in_dim, 9)
        self.assertEqual(self.encoder.hidden_dim, 128)
        self.assertEqual(self.encoder.num_layers, 5)
        self.assertEqual(self.encoder.pool_type, "sum")

    def test_encoder_eval_mode(self):
        self.assertFalse(self.encoder.training)

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.encoder.parameters())
        self.assertEqual(n, self.DOCUMENTED_PARAM_COUNT)

    def test_embed_ethanol_graph_finite_no_grad(self):
        import torch

        from apps.mol_cli.ogb_graph import smiles_to_pyg_data

        data = smiles_to_pyg_data("CCO")
        self.assertEqual(tuple(data.x.shape), (3, 9))

        self.encoder.eval()
        with torch.no_grad():
            projection, graph_embedding, _ = self.encoder(data)

        self.assertEqual(tuple(graph_embedding.shape), (1, 128))
        self.assertTrue(torch.isfinite(graph_embedding).all())
        self.assertFalse(graph_embedding.requires_grad)
        # Projection exists but is not used for ProtoNet; still must be finite.
        self.assertEqual(tuple(projection.shape), (1, 128))
        self.assertTrue(torch.isfinite(projection).all())

    def test_safe_load_uses_weights_only_true(self):
        """Helper must deserialize with weights_only=True."""
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "checkpoint_smoke.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name != "load":
                continue
            for kw in node.keywords:
                if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is True:
                        found = True
        self.assertTrue(found, "expected torch.load(..., weights_only=True)")


class TestBaceStillUnavailable(unittest.TestCase):
    def test_report_still_marks_bace_unavailable(self):
        from apps.mol_cli.report import UNAVAILABLE_STATUS, learned_endpoint_statuses

        statuses = learned_endpoint_statuses()
        self.assertEqual(statuses["BACE"], UNAVAILABLE_STATUS)


if __name__ == "__main__":
    unittest.main()
