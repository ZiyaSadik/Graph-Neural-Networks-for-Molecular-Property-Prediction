"""Tests for fixed-support BACE evaluation utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apps.mol_cli.bace_metrics import (
    MetricsError,
    compute_binary_metrics,
    confusion_matrix_binary,
)
from apps.mol_cli.bace_support import load_bace_split_inventory, load_manifest
from apps.mol_cli.evaluate_bace_support import (
    DEFAULT_REPORT_PATH,
    EvaluationError,
    assert_support_query_disjoint,
    write_report,
)
from apps.mol_cli.bace_infer import CANONICAL_MANIFEST
from apps.mol_cli.report import UNAVAILABLE_STATUS, learned_endpoint_statuses

REPO_ROOT = Path(__file__).resolve().parents[3]
BACE_ROOT = REPO_ROOT / "dataset" / "ogbg_molbace"


def _data_ok() -> bool:
    return (BACE_ROOT / "split" / "scaffold" / "train.csv.gz").is_file()


class TestBinaryMetrics(unittest.TestCase):
    def test_confusion_and_metrics_synthetic(self):
        # true: 0,0,1,1  pred: 0,1,0,1  => TN=1 FP=1 FN=1 TP=1
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        cm = confusion_matrix_binary(y_true, y_pred)
        self.assertEqual(cm["matrix"], [[1, 1], [1, 1]])
        self.assertEqual(cm["class_order"], [0, 1])
        m = compute_binary_metrics(y_true, y_pred, scores_class_1=[0.0, 0.5, 0.2, 0.9])
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(m["sensitivity_class_1"], 0.5)
        self.assertAlmostEqual(m["specificity_class_0"], 0.5)
        self.assertIsNotNone(m["auroc"])
        self.assertFalse(m["metric_notes"]["calibrated_probability"])

    def test_missing_class_undefined_metrics(self):
        y_true = [0, 0, 0]
        y_pred = [0, 1, 0]
        m = compute_binary_metrics(y_true, y_pred, scores_class_1=[0.1, 0.2, 0.3])
        self.assertIsNone(m["sensitivity_class_1"])
        self.assertIsNone(m["balanced_accuracy"])
        self.assertIsNone(m["auroc"])
        self.assertEqual(m["auroc_undefined_reason"], "only_one_class_present_in_y_true")

    def test_empty_rejected(self):
        with self.assertRaises(MetricsError):
            compute_binary_metrics([], [])


@unittest.skipUnless(_data_ok() and CANONICAL_MANIFEST.is_file(), "BACE cache/manifest missing")
class TestEvaluationIntegrity(unittest.TestCase):
    def test_official_split_membership_and_disjoint_support(self):
        inventory = load_bace_split_inventory(BACE_ROOT)
        manifest = load_manifest(CANONICAL_MANIFEST)
        support = {int(r["dataset_index"]) for r in manifest["support"]}
        train = set(inventory.train_indices)
        valid = set(inventory.valid_indices)
        test = set(inventory.test_indices)
        self.assertTrue(support.issubset(train))
        self.assertFalse(support & valid)
        self.assertFalse(support & test)
        assert_support_query_disjoint(support, list(inventory.valid_indices), "valid")
        assert_support_query_disjoint(support, list(inventory.test_indices), "test")

    def test_overlap_fails_loudly(self):
        with self.assertRaises(EvaluationError):
            assert_support_query_disjoint({1, 2, 3}, [3, 4, 5], "valid")

    def test_labels_are_original_binary(self):
        inventory = load_bace_split_inventory(BACE_ROOT)
        for idx in inventory.valid_indices + inventory.test_indices:
            self.assertIn(inventory.labels[idx], (0, 1))

    def test_write_report_refuses_overwrite(self):
        report = {"report_id": "tmp", "ok": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bace_fixed_support_eval_v1.json"
            write_report(report, path=path)
            with self.assertRaises(EvaluationError):
                write_report(report, path=path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["report_id"], "tmp")

    def test_deterministic_metrics_on_fixed_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        scores = [0.1, 0.8, 0.4, 0.2, 0.9]
        a = compute_binary_metrics(y_true, y_pred, scores)
        b = compute_binary_metrics(y_true, y_pred, scores)
        self.assertEqual(a, b)


class TestScopeGuards(unittest.TestCase):
    def test_cli_bace_still_unavailable(self):
        self.assertEqual(learned_endpoint_statuses()["BACE"], UNAVAILABLE_STATUS)

    def test_predict_not_wired_to_evaluator(self):
        import ast

        predict_path = REPO_ROOT / "apps" / "mol_cli" / "predict.py"
        text = predict_path.read_text(encoding="utf-8")
        self.assertNotIn("evaluate_bace_support", text)
        # Default path remains isolated; bace_infer may appear only as a
        # lazy import inside the explicit --bace branch (not at module top level).
        tree = ast.parse(text, filename=str(predict_path))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn("bace_infer", node.module.split("."))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("bace_infer", alias.name.split("."))

    def test_default_report_path_under_apps(self):
        self.assertIn("apps", DEFAULT_REPORT_PATH.parts)
        self.assertIn("mol_cli", DEFAULT_REPORT_PATH.parts)
        self.assertIn("evaluation_reports", DEFAULT_REPORT_PATH.parts)


if __name__ == "__main__":
    unittest.main()
