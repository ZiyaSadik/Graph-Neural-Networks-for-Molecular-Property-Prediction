"""Lightweight tests for CLI v0. Does not train models or load checkpoints."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from rdkit import Chem

from apps.mol_cli.descriptors import DESCRIPTOR_KEYS, compute_descriptors
from apps.mol_cli.report import (
    EXPLANATION_STATUS,
    LEARNED_ENDPOINTS,
    UNAVAILABLE_STATUS,
    format_report,
    learned_endpoint_statuses,
)
from apps.mol_cli.smiles_input import SmilesInputError, is_quit_command, parse_smiles

CLI_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORT_ROOTS = (
    "torch",
    "torch_geometric",
    "ogb",
    "src",
)


def _module_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestSmilesInput(unittest.TestCase):
    def test_valid_smiles(self):
        parsed = parse_smiles("CCO")
        self.assertEqual(parsed.num_atoms, 3)
        self.assertGreaterEqual(parsed.num_bonds, 2)
        self.assertEqual(parsed.original_smiles, "CCO")
        self.assertTrue(parsed.canonical_smiles)

    def test_invalid_smiles(self):
        with self.assertRaises(SmilesInputError) as ctx:
            parse_smiles("not-a-smiles")
        self.assertIn("Could not parse", str(ctx.exception))

    def test_empty_input(self):
        with self.assertRaises(SmilesInputError) as ctx:
            parse_smiles("")
        self.assertIn("No SMILES", str(ctx.exception))
        with self.assertRaises(SmilesInputError):
            parse_smiles("   ")
        with self.assertRaises(SmilesInputError):
            parse_smiles(None)

    def test_canonicalization(self):
        parsed = parse_smiles("C1=CC=CC=C1")
        self.assertEqual(parsed.original_smiles, "C1=CC=CC=C1")
        self.assertEqual(parsed.canonical_smiles, "c1ccccc1")

    def test_quit_commands(self):
        self.assertTrue(is_quit_command("quit"))
        self.assertTrue(is_quit_command("Q"))
        self.assertTrue(is_quit_command(" exit "))
        self.assertFalse(is_quit_command("CCO"))
        self.assertFalse(is_quit_command(""))

    def test_zero_atom_molecule_rejected(self):
        empty = Chem.Mol()
        self.assertEqual(empty.GetNumAtoms(), 0)
        with self.assertRaises(SmilesInputError):
            compute_descriptors(empty)


class TestDescriptors(unittest.TestCase):
    def test_descriptor_keys_and_types(self):
        parsed = parse_smiles("CCO")
        values = compute_descriptors(parsed.mol)
        self.assertEqual(tuple(values.keys()), DESCRIPTOR_KEYS)
        self.assertIsInstance(values["molecular_weight"], float)
        self.assertIsInstance(values["clogp"], float)
        self.assertIsInstance(values["tpsa"], float)
        self.assertIsInstance(values["hbond_donors"], int)
        self.assertIsInstance(values["hbond_acceptors"], int)
        self.assertIsInstance(values["rotatable_bonds"], int)
        self.assertIsInstance(values["ring_count"], int)
        self.assertGreater(values["molecular_weight"], 0.0)
        self.assertEqual(values["hbond_donors"], 1)
        self.assertEqual(values["hbond_acceptors"], 1)
        self.assertGreaterEqual(values["rotatable_bonds"], 0)
        self.assertEqual(values["ring_count"], 0)
        butane = parse_smiles("CCCC")
        self.assertGreaterEqual(compute_descriptors(butane.mol)["rotatable_bonds"], 1)

    def test_failed_parsing_has_no_descriptors(self):
        with self.assertRaises(SmilesInputError):
            parse_smiles("%%%%")
        with self.assertRaises(SmilesInputError):
            compute_descriptors(None)


class TestReportStatuses(unittest.TestCase):
    def test_all_learned_endpoints_unavailable(self):
        statuses = learned_endpoint_statuses()
        self.assertEqual(set(statuses), set(LEARNED_ENDPOINTS))
        for name in LEARNED_ENDPOINTS:
            self.assertEqual(statuses[name], UNAVAILABLE_STATUS)

    def test_report_text_contains_disclaimer_and_statuses(self):
        parsed = parse_smiles("C")
        text = format_report(parsed, compute_descriptors(parsed.mol))
        self.assertIn("not learned predictions", text.lower())
        self.assertIn("Not available — model integration and validation pending", text)
        for name in LEARNED_ENDPOINTS:
            self.assertIn(name, text)
        self.assertIn("Disabled", text)
        self.assertIn(EXPLANATION_STATUS.split("—")[0].strip(), text)
        self.assertNotIn("predicted probability", text.lower())


class TestNoResearchImports(unittest.TestCase):
    def test_rdkit_only_modules_do_not_import_research_stack(self):
        """Descriptor/report path stays free of torch/PyG/ogb/src at import time."""
        py_files = [
            CLI_ROOT / "__init__.py",
            CLI_ROOT / "smiles_input.py",
            CLI_ROOT / "descriptors.py",
            CLI_ROOT / "report.py",
        ]
        for path in py_files:
            roots = _module_import_roots(path)
            forbidden = roots & set(FORBIDDEN_IMPORT_ROOTS)
            self.assertFalse(
                forbidden,
                f"{path.name} imports forbidden modules: {sorted(forbidden)}",
            )

    def test_predict_does_not_top_level_import_torch_stack(self):
        """predict.py may lazy-import ogb_graph inside functions only."""
        path = CLI_ROOT / "predict.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level_roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_roots.add(node.module.split(".")[0])
        forbidden = top_level_roots & set(FORBIDDEN_IMPORT_ROOTS)
        self.assertFalse(
            forbidden,
            f"predict.py top-level imports forbidden modules: {sorted(forbidden)}",
        )
        # Relative imports of local packages are fine; ogb_graph is allowed as a
        # top-level relative import only if it is deferred. Ensure no absolute
        # torch/ogb at module scope (already checked). Lazy path uses relative import.


if __name__ == "__main__":
    unittest.main()
