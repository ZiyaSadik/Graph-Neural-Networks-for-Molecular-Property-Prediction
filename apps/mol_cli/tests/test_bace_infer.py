"""Tests for isolated BACE inference adapter (mechanics only)."""

from __future__ import annotations

import copy
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from apps.mol_cli.bace_infer import (
    CANONICAL_CHECKPOINT,
    CANONICAL_MANIFEST,
    CLASS_NAME,
    BaceInferenceError,
    BacePrototypeModel,
    classify_query_embedding,
    documented_class_name,
    load_validated_support_manifest,
    mean_prototypes_by_original_label,
    sha256_file,
)
from apps.mol_cli.bace_support import (
    SupportManifestError,
    bace_dataset_inventory_available,
    load_bace_split_inventory,
    load_manifest,
    validate_manifest,
    validate_manifest_for_inference,
)
from apps.mol_cli.checkpoint_smoke import load_checkpoint_dict_safe
from apps.mol_cli.report import UNAVAILABLE_STATUS, learned_endpoint_statuses

REPO_ROOT = Path(__file__).resolve().parents[3]


class TinyFixedEncoder(nn.Module):
    """
    Maps each graph to a fixed embedding based on data.bace_support_index
    or data.query_code (for synthetic tests). Returns (proj, graph_emb, None).
    """

    def __init__(self, table: dict[int, torch.Tensor], dim: int = 4):
        super().__init__()
        self.dim = dim
        self.table = {int(k): v.float() for k, v in table.items()}
        # unused parameter so .parameters() is non-empty if needed
        self._bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, data: Data):
        if hasattr(data, "batch") and data.batch is not None:
            num_graphs = int(data.batch.max().item()) + 1
        else:
            num_graphs = 1
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        outs = []
        for g in range(num_graphs):
            # Prefer per-graph lists stored on batch via custom attrs from DataList
            key = None
            if hasattr(data, "query_code"):
                qc = data.query_code
                if torch.is_tensor(qc):
                    key = int(qc.view(-1)[g].item()) if qc.numel() > 1 else int(qc.item())
                else:
                    key = int(qc)
            elif hasattr(data, "bace_support_index"):
                si = data.bace_support_index
                if torch.is_tensor(si):
                    # Batch may stack differently; fall back to order table keys
                    key = int(si.view(-1)[min(g, si.numel() - 1)].item())
                elif isinstance(si, (list, tuple)):
                    key = int(si[g])
                else:
                    key = int(si)
            if key is None or key not in self.table:
                # Use graph order index as key for batched support built sequentially
                key = g
            if key not in self.table:
                raise RuntimeError(f"TinyFixedEncoder missing key {key}")
            outs.append(self.table[key])
        emb = torch.stack(outs, dim=0)
        return emb, emb, None


def _manual_batch_encode(encoder, graphs):
    """Encode graphs one-by-one to avoid PyG batching quirks with custom attrs."""
    emb = []
    encoder.eval()
    with torch.no_grad():
        for g in graphs:
            # ensure batch vector
            if getattr(g, "batch", None) is None:
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            _proj, ge, _ = encoder(g)
            emb.append(ge.view(-1))
    return torch.stack(emb, dim=0)


class TestPrototypeMath(unittest.TestCase):
    def test_prototypes_are_class_means(self):
        emb = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        )
        labels = [0, 0, 1, 1]
        proto = mean_prototypes_by_original_label(emb, labels)
        self.assertEqual(tuple(proto.shape), (2, 2))
        self.assertTrue(torch.allclose(proto[0], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.allclose(proto[1], torch.tensor([0.0, 3.0])))

    def test_distance_prediction_controlled(self):
        prototypes = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        # Closer to class 0
        r0 = classify_query_embedding(torch.tensor([1.0, 0.0]), prototypes)
        self.assertEqual(r0.predicted_class, 0)
        self.assertEqual(r0.predicted_class_name, "inactive")
        self.assertFalse(r0.calibrated_probability)
        self.assertAlmostEqual(r0.distance_to_class_0, 1.0, places=5)
        self.assertAlmostEqual(r0.distance_to_class_1, 9.0, places=5)
        self.assertAlmostEqual(r0.distance_gap, 8.0, places=5)
        # Closer to class 1
        r1 = classify_query_embedding(torch.tensor([9.0, 0.0]), prototypes)
        self.assertEqual(r1.predicted_class, 1)
        self.assertEqual(r1.predicted_class_name, "active")

    def test_label_meanings_documented(self):
        self.assertEqual(documented_class_name(0), "inactive")
        self.assertEqual(documented_class_name(1), "active")
        self.assertEqual(CLASS_NAME[0], "inactive")
        self.assertEqual(CLASS_NAME[1], "active")

    def test_rejects_nonfinite(self):
        bad = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])
        with self.assertRaises(BaceInferenceError):
            mean_prototypes_by_original_label(bad, [0, 1])
        proto = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        with self.assertRaises(BaceInferenceError):
            classify_query_embedding(torch.tensor([float("inf"), 0.0]), proto)


class TestManifestGuards(unittest.TestCase):
    @unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
    def test_corrupt_manifest_rejected(self):
        manifest = load_manifest(CANONICAL_MANIFEST)
        bad = copy.deepcopy(manifest)
        bad["support"][0]["label"] = 3
        with self.assertRaises(SupportManifestError):
            validate_manifest_for_inference(bad)
        if bace_dataset_inventory_available():
            with self.assertRaises(SupportManifestError):
                validate_manifest(bad, inventory=load_bace_split_inventory())

    @unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
    def test_inference_loader_works_without_inventory_mode(self):
        manifest = load_validated_support_manifest(
            CANONICAL_MANIFEST,
            dataset_inventory="inference",
        )
        self.assertEqual(manifest["manifest_id"], "bace_train_k5_v1")
        self.assertEqual(len(manifest["support"]), 10)

    @unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
    def test_required_inventory_mode_fails_clearly_when_cache_missing(self):
        if bace_dataset_inventory_available():
            self.skipTest("local ogbg-molbace cache is present")
        with self.assertRaises(BaceInferenceError) as ctx:
            load_validated_support_manifest(
                CANONICAL_MANIFEST,
                dataset_inventory="required",
            )
        self.assertIn("incomplete", str(ctx.exception).lower())

    @unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
    def test_model_rejects_invalid_label_in_support_build(self):
        manifest = load_manifest(CANONICAL_MANIFEST)
        bad = copy.deepcopy(manifest)
        bad["support"][0]["label"] = 5
        # Bypass validate_manifest; support_graphs_from_manifest / from_encoder should fail
        table = {i: torch.zeros(2) for i in range(10)}
        enc = TinyFixedEncoder(table, dim=2)
        with self.assertRaises(BaceInferenceError):
            # Build graphs path checks labels
            from apps.mol_cli.bace_infer import support_graphs_from_manifest

            support_graphs_from_manifest(bad)


class TestFakeEncoderIntegration(unittest.TestCase):
    @unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
    def test_query_not_added_to_support_and_prediction(self):
        manifest = load_manifest(CANONICAL_MANIFEST)
        # Assign distinct embeddings per support row order
        table = {}
        labels = []
        for i, row in enumerate(manifest["support"]):
            labels.append(int(row["label"]))
            if row["label"] == 0:
                table[i] = torch.tensor([0.0, 0.0, 0.0, 0.0])
            else:
                table[i] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        enc = TinyFixedEncoder(table, dim=4)

        # Manually embed support in order (avoid Batch attr issues)
        from apps.mol_cli.bace_infer import (
            mean_prototypes_by_original_label,
            support_graphs_from_manifest,
        )

        graphs, labs, indices = support_graphs_from_manifest(manifest)
        support_size_before = len(graphs)
        emb = _manual_batch_encode(enc, graphs)
        # Override embeddings to match label means for clarity
        emb = torch.stack(
            [
                torch.tensor([0.0, 0.0, 0.0, 0.0])
                if lab == 0
                else torch.tensor([10.0, 0.0, 0.0, 0.0])
                for lab in labs
            ]
        )
        prototypes = mean_prototypes_by_original_label(emb, labs)
        model = BacePrototypeModel(
            encoder=enc,
            prototypes=prototypes,
            support_labels=tuple(labs),
            support_dataset_indices=tuple(indices),
            support_manifest_id=manifest["manifest_id"],
            checkpoint_sha256="fake",
            _support_size=support_size_before,
        )
        # Query closer to active prototype
        q = torch.tensor([[9.0, 0.0, 0.0, 0.0]])
        result = model.score_query_embedding(q)
        self.assertEqual(result.predicted_class, 1)
        self.assertFalse(result.calibrated_probability)
        self.assertIn("distance_to_class_0", result.to_dict())
        self.assertEqual(len(model.support_dataset_indices), support_size_before)
        self.assertEqual(model._support_size, support_size_before)


class TestCheckpointLoadingPath(unittest.TestCase):
    def test_safe_loader_uses_weights_only_true(self):
        from apps.mol_cli import checkpoint_smoke as cs

        src = inspect.getsource(cs.load_checkpoint_dict_safe)
        self.assertIn("weights_only=True", src)

    @unittest.skipUnless(CANONICAL_CHECKPOINT.is_file(), "checkpoint missing")
    def test_infer_module_does_not_call_string_path_loader(self):
        import apps.mol_cli.bace_infer as bi

        src = inspect.getsource(bi)
        # Must not call torch.load with weights_only=False
        self.assertNotIn("weights_only=False", src)
        # Uses helper that loads with weights_only=True
        self.assertIn("load_encoder_from_research_checkpoint", src)


@unittest.skipUnless(CANONICAL_CHECKPOINT.is_file(), "checkpoint missing")
@unittest.skipUnless(CANONICAL_MANIFEST.is_file(), "manifest missing")
class TestEndToEndSmoke(unittest.TestCase):
    def test_real_checkpoint_support_and_query_ethanol(self):
        digest = sha256_file(CANONICAL_CHECKPOINT)
        self.assertEqual(len(digest), 64)
        # Confirm safe dict load works before adapter
        ckpt = load_checkpoint_dict_safe(CANONICAL_CHECKPOINT)
        self.assertIn("encoder_state_dict", ckpt)

        model = BacePrototypeModel.from_checkpoint_and_manifest(
            checkpoint_path=CANONICAL_CHECKPOINT,
            manifest_path=CANONICAL_MANIFEST,
            device="cpu",
            dataset_inventory="inference",
        )
        self.assertEqual(model.checkpoint_sha256, digest)
        self.assertEqual(model.support_manifest_id, "bace_train_k5_v1")
        self.assertEqual(len(model.support_dataset_indices), 10)
        self.assertEqual(tuple(model.prototypes.shape), (2, 128))
        self.assertTrue(torch.isfinite(model.prototypes).all())

        result = model.score_query_smiles("CCO")
        self.assertIn(result.predicted_class, (0, 1))
        self.assertEqual(
            result.predicted_class_name, CLASS_NAME[result.predicted_class]
        )
        self.assertFalse(result.calibrated_probability)
        self.assertTrue(result.distance_to_class_0 >= 0.0)
        self.assertTrue(result.distance_to_class_1 >= 0.0)
        self.assertEqual(result.checkpoint_sha256, digest)
        # No accuracy claim — just mechanics
        self.assertIsNotNone(result.query_smiles_canonical)


class TestCliStillUnavailable(unittest.TestCase):
    def test_bace_still_unavailable_in_report(self):
        """Default report API keeps BACE unavailable unless a result is supplied."""
        self.assertEqual(
            learned_endpoint_statuses()["BACE"], UNAVAILABLE_STATUS
        )

    def test_predict_does_not_top_level_import_bace_infer(self):
        """bace_infer may be lazy-imported only on the explicit --bace path."""
        import ast

        predict_path = REPO_ROOT / "apps" / "mol_cli" / "predict.py"
        tree = ast.parse(
            predict_path.read_text(encoding="utf-8"), filename=str(predict_path)
        )
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn("bace_infer", node.module.split("."))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("bace_infer", alias.name.split("."))


if __name__ == "__main__":
    unittest.main()
