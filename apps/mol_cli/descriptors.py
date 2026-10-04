"""RDKit 2D descriptors. These are calculated features, not model outputs."""

from __future__ import annotations

from typing import Any

from rdkit.Chem import Descriptors, rdMolDescriptors

from .smiles_input import SmilesInputError

DESCRIPTOR_KEYS = (
    "molecular_weight",
    "clogp",
    "tpsa",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
)

DESCRIPTOR_LABELS = {
    "molecular_weight": "Molecular weight",
    "clogp": "cLogP",
    "tpsa": "TPSA",
    "hbond_donors": "Hydrogen-bond donors",
    "hbond_acceptors": "Hydrogen-bond acceptors",
    "rotatable_bonds": "Rotatable bonds",
    "ring_count": "Ring count",
}


def compute_descriptors(mol: Any) -> dict[str, float | int]:
    """Return a fixed set of RDKit descriptors for a parsed molecule."""
    if mol is None:
        raise SmilesInputError("Cannot calculate descriptors: no molecule was parsed.")

    n_atoms = int(mol.GetNumAtoms())
    if n_atoms <= 0:
        raise SmilesInputError(
            "Cannot calculate descriptors: molecule has no atoms."
        )

    try:
        values: dict[str, float | int] = {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "clogp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "hbond_donors": int(rdMolDescriptors.CalcNumHBD(mol)),
            "hbond_acceptors": int(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        }
    except Exception as exc:
        raise SmilesInputError(
            f"RDKit could not calculate descriptors ({exc})."
        ) from None

    return values
