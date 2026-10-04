"""Tests for OGB SMILES-to-graph conversion. Does not load checkpoints."""

from __future__ import annotations

import unittest

import numpy as np
from rdkit import Chem

from apps.mol_cli.ogb_graph import (
    EXPECTED_EDGE_FEAT_DIM,
    EXPECTED_NODE_FEAT_DIM,
    GraphConversionError,
    ogb_graph_deps_available,
    smiles_to_ogb_dict,
    smiles_to_pyg_data,
)


def _deps_ok() -> bool:
    ok, _ = ogb_graph_deps_available()
    return ok


@unittest.skipUnless(_deps_ok(), "torch / torch_geometric / ogb not available")
class TestOgbGraphConversion(unittest.TestCase):
    def test_deps_available_in_this_environment(self):
        ok, detail = ogb_graph_deps_available()
        self.assertTrue(ok, detail)

    def test_match_official_smiles2graph_ethanol(self):
        from ogb.utils.mol import smiles2graph

        smiles = "CCO"
        ours = smiles_to_ogb_dict(smiles)
        ref = smiles2graph(smiles)

        np.testing.assert_array_equal(ours["node_feat"], ref["node_feat"])
        np.testing.assert_array_equal(ours["edge_index"], ref["edge_index"])
        np.testing.assert_array_equal(ours["edge_feat"], ref["edge_feat"])
        self.assertEqual(ours["num_nodes"], ref["num_nodes"])
        self.assertEqual(ours["node_feat"].shape, (3, EXPECTED_NODE_FEAT_DIM))
        self.assertEqual(ours["node_feat"].dtype, np.int64)
        self.assertEqual(ours["edge_index"].dtype, np.int64)
        self.assertEqual(ours["edge_feat"].dtype, np.int64)
        self.assertEqual(ours["edge_feat"].shape[1], EXPECTED_EDGE_FEAT_DIM)

    def test_pyg_data_dtypes_and_shapes(self):
        import torch

        data = smiles_to_pyg_data("CCO")
        self.assertEqual(tuple(data.x.shape), (3, 9))
        self.assertEqual(data.x.dtype, torch.int64)
        self.assertEqual(data.edge_index.dtype, torch.int64)
        self.assertEqual(data.edge_attr.dtype, torch.int64)
        self.assertEqual(data.edge_index.size(0), 2)
        self.assertEqual(data.edge_index.size(1), data.edge_attr.size(0))
        self.assertEqual(int(data.num_nodes), 3)
        # Bidirectional: 2 undirected bonds -> 4 directed edges
        self.assertEqual(data.edge_index.size(1), 4)

    def test_bidirectional_edge_ordering_matches_ogb(self):
        from ogb.utils.mol import smiles2graph

        smiles = "C=O"
        ours = smiles_to_ogb_dict(smiles)
        ref = smiles2graph(smiles)
        np.testing.assert_array_equal(ours["edge_index"], ref["edge_index"])
        # Each undirected bond appears as (i,j) then (j,i)
        ei = ours["edge_index"]
        self.assertEqual(ei.shape[1] % 2, 0)
        for k in range(0, ei.shape[1], 2):
            self.assertEqual(int(ei[0, k]), int(ei[1, k + 1]))
            self.assertEqual(int(ei[1, k]), int(ei[0, k + 1]))

    def test_single_atom_molecule(self):
        from ogb.utils.mol import smiles2graph

        smiles = "C"
        ours = smiles_to_ogb_dict(smiles)
        ref = smiles2graph(smiles)
        np.testing.assert_array_equal(ours["node_feat"], ref["node_feat"])
        np.testing.assert_array_equal(ours["edge_index"], ref["edge_index"])
        np.testing.assert_array_equal(ours["edge_feat"], ref["edge_feat"])
        self.assertEqual(ours["num_nodes"], 1)
        self.assertEqual(ours["edge_index"].shape, (2, 0))
        self.assertEqual(ours["edge_feat"].shape, (0, EXPECTED_EDGE_FEAT_DIM))

        data = smiles_to_pyg_data(smiles)
        self.assertEqual(tuple(data.x.shape), (1, 9))
        self.assertEqual(data.edge_index.numel(), 0)

    def test_bond_types_match_ogb(self):
        from ogb.utils.mol import smiles2graph

        cases = {
            "single": "CC",
            "double": "C=C",
            "triple": "C#C",
            "aromatic": "c1ccccc1",
        }
        for label, smiles in cases.items():
            with self.subTest(bond=label, smiles=smiles):
                ours = smiles_to_ogb_dict(smiles)
                ref = smiles2graph(smiles)
                np.testing.assert_array_equal(
                    ours["node_feat"], ref["node_feat"], err_msg=label
                )
                np.testing.assert_array_equal(
                    ours["edge_index"], ref["edge_index"], err_msg=label
                )
                np.testing.assert_array_equal(
                    ours["edge_feat"], ref["edge_feat"], err_msg=label
                )
                self.assertEqual(ours["node_feat"].shape[1], EXPECTED_NODE_FEAT_DIM)
                self.assertEqual(ours["edge_feat"].shape[1], EXPECTED_EDGE_FEAT_DIM)

    def test_invalid_smiles(self):
        with self.assertRaises(GraphConversionError) as ctx:
            smiles_to_ogb_dict("not-a-smiles")
        self.assertIn("invalid smiles", str(ctx.exception).lower())

    def test_empty_smiles(self):
        with self.assertRaises(GraphConversionError):
            smiles_to_ogb_dict("")
        with self.assertRaises(GraphConversionError):
            smiles_to_ogb_dict("   ")

    def test_zero_atom_mol_rejected_before_approx(self):
        empty = Chem.Mol()
        self.assertEqual(empty.GetNumAtoms(), 0)
        # Empty RDKit mol has no SMILES path through our API; empty string already covered.
        # Explicitly ensure we never invent features for zero atoms via smiles path.
        with self.assertRaises(GraphConversionError):
            smiles_to_pyg_data("")

    def test_pyg_matches_numpy_from_same_smiles(self):
        import torch

        smiles = "CC(=O)O"
        graph = smiles_to_ogb_dict(smiles)
        data = smiles_to_pyg_data(smiles)
        np.testing.assert_array_equal(data.x.cpu().numpy(), graph["node_feat"])
        np.testing.assert_array_equal(
            data.edge_index.cpu().numpy(), graph["edge_index"]
        )
        np.testing.assert_array_equal(
            data.edge_attr.cpu().numpy(), graph["edge_feat"]
        )
        self.assertEqual(data.x.dtype, torch.int64)


class TestOgbGraphModuleIsolation(unittest.TestCase):
    def test_rdkit_modules_still_import_without_ogb_graph(self):
        """Descriptor path must not require loading ogb_graph at import time."""
        import importlib

        import apps.mol_cli.descriptors as descriptors
        import apps.mol_cli.smiles_input as smiles_input
        import apps.mol_cli.report as report

        importlib.reload(smiles_input)
        importlib.reload(descriptors)
        importlib.reload(report)
        parsed = smiles_input.parse_smiles("CCO")
        values = descriptors.compute_descriptors(parsed.mol)
        text = report.format_report(parsed, values, graph_summary=None)
        self.assertIn("Calculated descriptors", text)
        self.assertIn("OGB molecular graph", text)


if __name__ == "__main__":
    unittest.main()
