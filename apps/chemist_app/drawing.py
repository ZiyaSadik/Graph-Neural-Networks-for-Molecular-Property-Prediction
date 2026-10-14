"""2D molecule drawing with atom indices (RDKit only)."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np


def mol_to_png_with_atom_indices(mol: Any, size: tuple[int, int] = (420, 320)) -> bytes:
    """Draw molecule with atom index labels. Raises if RDKit drawing fails."""
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.additionalAtomLabelPadding = 0.15
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_png_with_atom_importance(
    mol: Any,
    node_mask: np.ndarray,
    size: tuple[int, int] = (420, 320),
) -> bytes:
    """
    Highlight atoms by importance mask (length must equal num atoms).
    Higher mask → more red. Atom indices are shown.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    n = mol.GetNumAtoms()
    if len(node_mask) != n:
        raise ValueError(
            f"node_mask length {len(node_mask)} != num atoms {n}; "
            "refusing to draw a misaligned explanation."
        )
    m = np.asarray(node_mask, dtype=np.float64).reshape(-1)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    else:
        m = np.zeros_like(m)

    colors = {i: (1.0, float(1.0 - m[i]), float(1.0 - m[i])) for i in range(n)}
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(n)),
        highlightAtomColors=colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def png_bytes_to_pil(png: bytes):
    from PIL import Image

    return Image.open(BytesIO(png))
