"""Tests for explicitly gated BACE CLI integration (--bace)."""

from __future__ import annotations

import ast
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from apps.mol_cli.descriptors import compute_descriptors
from apps.mol_cli.report import (
    BACE_EVAL_REPORT_ID,
    BACE_EXPERIMENTAL_STATUS,
    UNAVAILABLE_STATUS,
    format_bace_section,
    format_report,
    learned_endpoint_statuses,
)
from apps.mol_cli.smiles_input import parse_smiles

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_ROOT = Path(__file__).resolve().parents[1]
PREDICT_PATH = CLI_ROOT / "predict.py"


def _fake_bace_result(**overrides):
    base = {
        "predicted_class": 0,
        "predicted_class_name": "inactive",
        "distance_to_class_0": 1.25,
        "distance_to_class_1": 3.5,
        "distance_gap": 2.25,
        "checkpoint_sha256": "d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f",
        "support_manifest_id": "bace_train_k5_v1",
        "calibrated_probability": False,
        "support_dataset_indices": (182, 897),
        "query_smiles_canonical": "CCO",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestDefaultPathIsolation(unittest.TestCase):
    def test_learned_endpoint_statuses_still_unavailable_by_default(self):
        statuses = learned_endpoint_statuses()
        self.assertEqual(statuses["BACE"], UNAVAILABLE_STATUS)
        for name in ("solubility", "toxicity", "ADMET"):
            self.assertEqual(statuses[name], UNAVAILABLE_STATUS)

    def test_format_report_without_bace_result_keeps_unavailable(self):
        parsed = parse_smiles("CCO")
        text = format_report(parsed, compute_descriptors(parsed.mol))
        self.assertIn(UNAVAILABLE_STATUS, text)
        self.assertNotIn("=== Experimental BACE", text)
        self.assertNotIn(BACE_EXPERIMENTAL_STATUS, text)

    def test_predict_has_no_top_level_bace_infer_import(self):
        tree = ast.parse(PREDICT_PATH.read_text(encoding="utf-8"), filename=str(PREDICT_PATH))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                self.assertNotIn(
                    "bace_infer",
                    parts,
                    "predict.py must not import bace_infer at module top level",
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("bace_infer", alias.name.split("."))

    def test_default_main_does_not_initialize_bace_model(self):
        from apps.mol_cli import predict as predict_mod

        with (
            patch.object(predict_mod, "_initialize_bace_model") as init_mock,
            patch.object(predict_mod, "input", side_effect=["quit"]),
            redirect_stdout(io.StringIO()),
        ):
            code = predict_mod.main([])
        self.assertEqual(code, 0)
        init_mock.assert_not_called()


class TestBaceReportFormatting(unittest.TestCase):
    def test_bace_section_contains_required_disclaimers(self):
        result = _fake_bace_result()
        text = "\n".join(format_bace_section(result))
        self.assertIn("inactive (0)", text)
        self.assertIn("1.250000", text)
        self.assertIn("3.500000", text)
        self.assertIn("calibrated_probability: false", text)
        self.assertIn("not probabilities", text.lower())
        self.assertIn("bace_train_k5_v1", text)
        self.assertIn(BACE_EVAL_REPORT_ID, text)
        self.assertIn("episodic", text.lower())
        self.assertIn("0.66", text)
        self.assertIn("0.43", text)
        self.assertIn("approx.", text)
        self.assertNotIn("\u2248", text)  # ≈ — not cp1252-safe
        self.assertIn("classified inactive", text.lower())
        self.assertIn("d6951b12", text)
        self.assertNotIn("clinical utility", text.lower())
        self.assertNotIn("confidence %", text.lower())
        # Windows default console encoding must be able to print the report.
        text.encode("cp1252")

    def test_full_bace_report_is_cp1252_encodable(self):
        parsed = parse_smiles("CCO")
        text = format_report(
            parsed,
            compute_descriptors(parsed.mol),
            bace_result=_fake_bace_result(),
        )
        self.assertIn("approx. 0.66", text)
        self.assertIn("approx. 0.43", text)
        self.assertIn("calibrated_probability: false", text)
        self.assertIn("episodic", text.lower())
        self.assertNotIn("\u2248", text)
        encoded = text.encode("cp1252")
        self.assertEqual(encoded.decode("cp1252"), text)

    def test_format_report_with_bace_result(self):
        parsed = parse_smiles("CCO")
        text = format_report(
            parsed,
            compute_descriptors(parsed.mol),
            bace_result=_fake_bace_result(),
        )
        self.assertIn(BACE_EXPERIMENTAL_STATUS, text)
        self.assertIn("=== Experimental BACE", text)
        self.assertIn("solubility", text)
        self.assertIn(UNAVAILABLE_STATUS, text)
        self.assertIn("Disabled", text)


class TestBaceCliOptIn(unittest.TestCase):
    def _mock_model(self):
        model = MagicMock()
        model.support_manifest_id = "bace_train_k5_v1"
        model.checkpoint_sha256 = (
            "d6951b12412cbab90e7a6a0ef04004c99d6b44898b65d13424a4cb17fbe1247f"
        )
        model.score_query_smiles.return_value = _fake_bace_result()
        return model

    def test_bace_flag_initializes_once_and_reuses_model(self):
        from apps.mol_cli import predict as predict_mod

        model = self._mock_model()
        with (
            patch.object(predict_mod, "_initialize_bace_model", return_value=model) as init_mock,
            patch.object(
                predict_mod,
                "input",
                side_effect=["CCO", "c1ccccc1", "quit"],
            ),
            redirect_stdout(io.StringIO()) as out,
        ):
            code = predict_mod.main(["--bace"])
        self.assertEqual(code, 0)
        init_mock.assert_called_once()
        self.assertEqual(model.score_query_smiles.call_count, 2)
        text = out.getvalue()
        self.assertIn("Experimental BACE mode ENABLED", text)
        self.assertIn("inactive (0)", text)
        self.assertIn(BACE_EVAL_REPORT_ID, text)

    def test_invalid_smiles_does_not_call_bace_scorer(self):
        from apps.mol_cli import predict as predict_mod

        model = self._mock_model()
        with (
            patch.object(predict_mod, "_initialize_bace_model", return_value=model),
            patch.object(
                predict_mod,
                "input",
                side_effect=["not-a-smiles", "quit"],
            ),
            redirect_stdout(io.StringIO()) as out,
        ):
            code = predict_mod.main(["--bace"])
        self.assertEqual(code, 0)
        model.score_query_smiles.assert_not_called()
        self.assertIn("Could not parse", out.getvalue())

    def test_init_failure_exits_before_repl(self):
        from apps.mol_cli import predict as predict_mod
        from apps.mol_cli.bace_infer import BaceInferenceError

        with (
            patch.object(
                predict_mod,
                "_initialize_bace_model",
                side_effect=BaceInferenceError("checkpoint missing"),
            ),
            patch.object(predict_mod, "input") as input_mock,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as err,
        ):
            code = predict_mod.main(["--bace"])
        self.assertEqual(code, 1)
        input_mock.assert_not_called()
        self.assertIn("BACE initialization failed", err.getvalue())
        self.assertIn("checkpoint missing", err.getvalue())

    def test_per_query_inference_error_continues_without_traceback(self):
        from apps.mol_cli import predict as predict_mod
        from apps.mol_cli.bace_infer import BaceInferenceError

        model = self._mock_model()
        model.score_query_smiles.side_effect = [
            BaceInferenceError("non-finite distances"),
            _fake_bace_result(predicted_class=1, predicted_class_name="active"),
        ]
        with (
            patch.object(predict_mod, "_initialize_bace_model", return_value=model),
            patch.object(
                predict_mod,
                "input",
                side_effect=["CCO", "CC", "quit"],
            ),
            redirect_stdout(io.StringIO()) as out,
            redirect_stderr(io.StringIO()),
        ):
            code = predict_mod.main(["--bace"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("BACE inference error: non-finite distances", text)
        self.assertNotIn("Traceback", text)
        self.assertIn("active (1)", text)
        self.assertEqual(model.score_query_smiles.call_count, 2)


if __name__ == "__main__":
    unittest.main()
