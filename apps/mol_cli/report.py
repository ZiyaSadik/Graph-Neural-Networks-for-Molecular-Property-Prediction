"""Terminal report formatting. No learned scores are invented."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .descriptors import DESCRIPTOR_LABELS, DESCRIPTOR_KEYS
from .smiles_input import ParsedMolecule

LEARNED_ENDPOINTS = ("BACE", "solubility", "toxicity", "ADMET")

UNAVAILABLE_STATUS = (
    "Not available — model integration and validation pending"
)

BACE_EXPERIMENTAL_STATUS = (
    "experimental few-shot prototype (see section below)"
)

DESCRIPTOR_DISCLAIMER = (
    "These values are calculated chemical features (RDKit), "
    "not learned predictions."
)

EXPLANATION_STATUS = (
    "Disabled — atom- and bond-level explanations are not shown because "
    "their reliability for this tool has not been evaluated. "
    "A plot or mask is not treated as a valid explanation."
)

GRAPH_CONSTRUCTION_DISCLAIMER = (
    "OGB graph construction only. Successful conversion does not validate "
    "GIN or ProtoNet predictions."
)

# Pinned summary of apps/mol_cli/evaluation_reports/bace_fixed_support_eval_v1.json
# (OGB BACE scaffold test split under the fixed train-support protocol).
# Not live metrics; not interchangeable with the paper episodic protocol.
BACE_EVAL_REPORT_ID = "bace_fixed_support_eval_v1"
BACE_SUPPORT_MANIFEST_ID = "bace_train_k5_v1"
BACE_SUPPORT_K_SHOT = 5
BACE_SUPPORT_SEED = 42
BACE_EVAL_TEST_BALANCED_ACCURACY_APPROX = 0.66
BACE_EVAL_TEST_SENSITIVITY_APPROX = 0.43


class BaceResultLike(Protocol):
    """Minimal fields needed to render a BACE inference result."""

    predicted_class: int
    predicted_class_name: str
    distance_to_class_0: float
    distance_to_class_1: float
    distance_gap: float
    checkpoint_sha256: str
    support_manifest_id: str
    calibrated_probability: bool


def learned_endpoint_statuses() -> dict[str, str]:
    """Default statuses when no optional learned result is supplied."""
    return {name: UNAVAILABLE_STATUS for name in LEARNED_ENDPOINTS}


def format_graph_section(graph_summary: dict[str, object] | None) -> list[str]:
    """Format optional OGB graph metadata (plain dict; no tensors)."""
    lines = ["", "=== OGB molecular graph ==="]
    if graph_summary is None:
        lines.append(
            "Not built in this session (dependencies unavailable or conversion skipped)."
        )
        return lines

    lines.append(GRAPH_CONSTRUCTION_DISCLAIMER)
    lines.append(f"Converter:         {graph_summary.get('converter', 'unknown')}")
    lines.append(f"Nodes:             {graph_summary.get('num_nodes')}")
    lines.append(f"Directed edges:    {graph_summary.get('num_edges')}")
    lines.append(
        f"Node features:     {graph_summary.get('node_feat_dim')} "
        f"({graph_summary.get('x_dtype')})"
    )
    lines.append(
        f"Edge features:     {graph_summary.get('edge_feat_dim')} "
        f"({graph_summary.get('edge_attr_dtype')})"
    )
    lines.append(
        f"edge_index dtype:  {graph_summary.get('edge_index_dtype')}"
    )
    return lines


def _checkpoint_hash_prefix(digest: str, n: int = 8) -> str:
    digest = (digest or "").strip()
    if len(digest) <= n:
        return digest or "(unknown)"
    return digest[:n] + "..."


def format_bace_section(bace_result: BaceResultLike | Mapping[str, Any]) -> list[str]:
    """
    Format experimental BACE few-shot prototype output plus fixed disclaimers.

    Accepts BaceInferenceResult or a plain mapping with the same keys.
    """
    if isinstance(bace_result, Mapping):
        pred = int(bace_result["predicted_class"])
        name = str(bace_result["predicted_class_name"])
        d0 = float(bace_result["distance_to_class_0"])
        d1 = float(bace_result["distance_to_class_1"])
        gap = float(bace_result["distance_gap"])
        digest = str(bace_result.get("checkpoint_sha256", ""))
        manifest_id = str(
            bace_result.get("support_manifest_id", BACE_SUPPORT_MANIFEST_ID)
        )
        calibrated = bool(bace_result.get("calibrated_probability", False))
    else:
        pred = int(bace_result.predicted_class)
        name = str(bace_result.predicted_class_name)
        d0 = float(bace_result.distance_to_class_0)
        d1 = float(bace_result.distance_to_class_1)
        gap = float(bace_result.distance_gap)
        digest = str(bace_result.checkpoint_sha256)
        manifest_id = str(bace_result.support_manifest_id) or BACE_SUPPORT_MANIFEST_ID
        calibrated = bool(bace_result.calibrated_probability)

    lines = [
        "",
        "=== Experimental BACE (few-shot prototypes) ===",
        f"Predicted class:     {name} ({pred})",
        f"Distance to class 0 (inactive): {d0:.6f}",
        f"Distance to class 1 (active):   {d1:.6f}",
        f"Distance gap |d0-d1|:           {gap:.6f}",
        f"calibrated_probability: {str(calibrated).lower()}",
        (
            "Distances are Euclidean prototype distances, "
            "not probabilities or calibrated confidence percentages."
        ),
        "",
        f"Checkpoint SHA-256 (prefix): {_checkpoint_hash_prefix(digest)}",
        (
            f"Support manifest:            {manifest_id} "
            f"(train-only, K={BACE_SUPPORT_K_SHOT} per class, "
            f"seed {BACE_SUPPORT_SEED})"
        ),
        f"Eval report:                 {BACE_EVAL_REPORT_ID}",
        (
            "Protocol: fixed train-only support prototypes on OGB BACE scaffold "
            "splits; not the research paper's episodic test-pool ProtoNet protocol."
        ),
        (
            f"Pinned {BACE_EVAL_REPORT_ID} test summary (this protocol only): "
            f"balanced accuracy approx. {BACE_EVAL_TEST_BALANCED_ACCURACY_APPROX:.2f}; "
            f"sensitivity approx. {BACE_EVAL_TEST_SENSITIVITY_APPROX:.2f}."
        ),
        (
            "Many true active molecules were classified inactive under that "
            "evaluation; this is not a claim of correctness for any new molecule."
        ),
        (
            "Predictions may be unreliable for molecules outside the evaluated "
            "BACE chemical domain. Not clinical advice."
        ),
    ]
    return lines


def format_report(
    parsed: ParsedMolecule,
    descriptors: dict[str, float | int],
    graph_summary: dict[str, object] | None = None,
    bace_result: BaceResultLike | Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "",
        "=== Molecule ===",
        f"Original SMILES:   {parsed.original_smiles}",
        f"Canonical SMILES:  {parsed.canonical_smiles}",
        f"Atoms:             {parsed.num_atoms}",
        f"Bonds:             {parsed.num_bonds}",
        "",
        "=== Calculated descriptors ===",
        DESCRIPTOR_DISCLAIMER,
    ]
    for key in DESCRIPTOR_KEYS:
        label = DESCRIPTOR_LABELS[key]
        value = descriptors[key]
        if isinstance(value, float):
            lines.append(f"{label:24s} {value:.4f}")
        else:
            lines.append(f"{label:24s} {value}")

    lines.extend(format_graph_section(graph_summary))

    statuses = learned_endpoint_statuses()
    if bace_result is not None:
        statuses["BACE"] = BACE_EXPERIMENTAL_STATUS

    lines.extend(["", "=== Learned predictions ==="])
    for name, status in statuses.items():
        lines.append(f"{name:12s} {status}")

    if bace_result is not None:
        lines.extend(format_bace_section(bace_result))

    lines.extend(
        [
            "",
            "=== Atom / bond explanations ===",
            EXPLANATION_STATUS,
            "",
        ]
    )
    return "\n".join(lines)
