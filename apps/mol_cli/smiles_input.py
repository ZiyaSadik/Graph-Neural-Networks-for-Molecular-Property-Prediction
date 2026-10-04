"""Parse and canonicalize SMILES. RDKit only; no learned models."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


class SmilesInputError(ValueError):
    """User-facing SMILES or input error (no traceback needed)."""


@dataclass(frozen=True)
class ParsedMolecule:
    original_smiles: str
    canonical_smiles: str
    mol: object
    num_atoms: int
    num_bonds: int


def is_quit_command(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in QUIT_COMMANDS


def parse_smiles(raw: str | None) -> ParsedMolecule:
    """
    Parse a SMILES string into an RDKit molecule.

    Raises SmilesInputError for empty input, invalid SMILES, or zero atoms.
    """
    if raw is None or not str(raw).strip():
        raise SmilesInputError(
            "No SMILES was entered. Type a SMILES string, or quit to exit."
        )

    original = str(raw).strip()
    mol = Chem.MolFromSmiles(original)
    if mol is None:
        raise SmilesInputError(
            f"Could not parse {original!r} as SMILES. "
            "Check atom symbols, parentheses, and ring numbers."
        )

    n_atoms = int(mol.GetNumAtoms())
    if n_atoms <= 0:
        raise SmilesInputError(
            f"{original!r} parsed but has no atoms. Enter a molecule with at least one atom."
        )

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:
        raise SmilesInputError(
            f"Parsed {original!r} but could not write a canonical SMILES ({exc})."
        ) from None

    return ParsedMolecule(
        original_smiles=original,
        canonical_smiles=canonical,
        mol=mol,
        num_atoms=n_atoms,
        num_bonds=int(mol.GetNumBonds()),
    )
