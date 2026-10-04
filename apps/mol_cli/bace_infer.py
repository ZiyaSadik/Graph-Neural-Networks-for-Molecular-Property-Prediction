"""
Isolated BACE few-shot inference adapter (mechanics only).

Uses the fixed train-only support manifest and the research GraphCL GINEncoder.
Not wired into the user-facing CLI. Does not claim calibrated probabilities or
validation/test accuracy.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .bace_support import (
    DEFAULT_MANIFEST_PATH,
    LABEL_SEMANTICS,
    SupportManifestError,
    bace_dataset_inventory_available,
    load_bace_split_inventory,
    load_manifest,
    validate_manifest,
    validate_manifest_for_inference,
)
from .checkpoint_smoke import (
    CheckpointCompatibilityError,
    DEFAULT_CHECKPOINT_PATH,
    load_encoder_from_research_checkpoint,
)
from .ogb_graph import GraphConversionError, smiles_to_pyg_data

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CHECKPOINT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
CANONICAL_MANIFEST = DEFAULT_MANIFEST_PATH

CLASS_NAME = {
    0: "inactive",
    1: "active",
}


class BaceInferenceError(RuntimeError):
    """Inference / prototype / scoring failure."""


class GraphEncoder(Protocol):
    """Minimal encoder protocol for tests (fake) and GINEncoder."""

    def eval(self): ...

    def __call__(self, data): ...


@dataclass(frozen=True)
class BaceInferenceResult:
    predicted_class: int
    predicted_class_name: str
    distance_to_class_0: float
    distance_to_class_1: float
    distance_gap: float
    checkpoint_sha256: str
    support_manifest_id: str
    calibrated_probability: bool
    support_dataset_indices: tuple[int, ...]
    query_smiles_canonical: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    if not path.is_file():
        raise BaceInferenceError(f"Checkpoint not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint_path(path: Path | str | None = None) -> Path:
    if path is None:
        # Prefer absolute repo path; fall back to CWD-relative default.
        if CANONICAL_CHECKPOINT.is_file():
            return CANONICAL_CHECKPOINT
        cand = Path(DEFAULT_CHECKPOINT_PATH)
        if cand.is_file():
            return cand.resolve()
        raise BaceInferenceError(
            f"Checkpoint not found at {CANONICAL_CHECKPOINT}"
        )
    path = Path(path)
    if not path.is_file():
        raise BaceInferenceError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_validated_support_manifest(
    manifest_path: Path | str = CANONICAL_MANIFEST,
    *,
    dataset_inventory: str = "auto",
):
    """
    Load and validate the support manifest for BACE scoring.

    dataset_inventory:
      - ``"auto"`` (default for interactive inference): use full OGB inventory
        validation when ``dataset/ogbg_molbace`` is present; otherwise use
        ``validate_manifest_for_inference`` (schema / K-shot / SMILES only).
      - ``"required"``: always require the local OGB inventory (offline eval).
      - ``"inference"``: always use inference-only validation (no inventory).

    Inference-only mode does **not** re-verify training-split membership.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise BaceInferenceError(f"Support manifest not found: {path}")
    manifest = load_manifest(path)

    mode = str(dataset_inventory).strip().lower()
    if mode not in {"auto", "required", "inference"}:
        raise BaceInferenceError(
            f"dataset_inventory must be 'auto', 'required', or 'inference'; "
            f"got {dataset_inventory!r}."
        )

    try:
        if mode == "inference":
            validate_manifest_for_inference(manifest)
        elif mode == "required":
            inventory = load_bace_split_inventory()
            validate_manifest(manifest, inventory=inventory)
        else:
            # auto
            if bace_dataset_inventory_available():
                inventory = load_bace_split_inventory()
                validate_manifest(manifest, inventory=inventory)
            else:
                validate_manifest_for_inference(manifest)
    except SupportManifestError as exc:
        raise BaceInferenceError(str(exc)) from None
    return manifest


def _require_finite(name: str, tensor) -> None:
    import torch

    if not torch.is_tensor(tensor):
        raise BaceInferenceError(f"{name} must be a tensor.")
    if not torch.isfinite(tensor).all():
        raise BaceInferenceError(f"{name} contains non-finite values.")


def mean_prototypes_by_original_label(
    support_embeddings,
    support_labels,
):
    """
    Build class prototypes as means of embeddings with original labels 0 and 1.

    Does not remap episode class order.
    """
    import torch

    _require_finite("support_embeddings", support_embeddings)
    if support_embeddings.dim() != 2:
        raise BaceInferenceError("support_embeddings must have shape [N, D].")
    if len(support_labels) != support_embeddings.size(0):
        raise BaceInferenceError("support label count must match embeddings.")

    labels = [int(x) for x in support_labels]
    if any(y not in (0, 1) for y in labels):
        raise BaceInferenceError("Support labels must be original binary 0/1.")

    prototypes = []
    for cls in (0, 1):
        mask = [i for i, y in enumerate(labels) if y == cls]
        if not mask:
            raise BaceInferenceError(f"No support embeddings for class {cls}.")
        proto = support_embeddings[mask].mean(dim=0)
        _require_finite(f"prototype_{cls}", proto)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)


def classify_query_embedding(query_embedding, prototypes) -> BaceInferenceResult:
    """
    Euclidean nearest-prototype classification.

    Logits are not returned as probabilities. calibrated_probability is always False.
    """
    import torch

    _require_finite("query_embedding", query_embedding)
    _require_finite("prototypes", prototypes)
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    if query_embedding.size(0) != 1:
        raise BaceInferenceError("Exactly one query embedding is required.")
    if prototypes.shape != (2, query_embedding.size(1)):
        raise BaceInferenceError(
            f"prototypes must have shape [2, {query_embedding.size(1)}], "
            f"got {tuple(prototypes.shape)}."
        )

    distances = torch.cdist(query_embedding, prototypes, p=2).squeeze(0)
    _require_finite("distances", distances)
    d0 = float(distances[0].item())
    d1 = float(distances[1].item())
    pred = int(distances.argmin().item())
    if pred not in (0, 1):
        raise BaceInferenceError(f"Unexpected predicted class {pred}.")

    return BaceInferenceResult(
        predicted_class=pred,
        predicted_class_name=CLASS_NAME[pred],
        distance_to_class_0=d0,
        distance_to_class_1=d1,
        distance_gap=abs(d0 - d1),
        checkpoint_sha256="",
        support_manifest_id="",
        calibrated_probability=False,
        support_dataset_indices=tuple(),
    )


def embed_graphs(encoder: GraphEncoder, graphs: list, device: str = "cpu"):
    """Batch-embed PyG graphs; returns graph embeddings only (not projection)."""
    import torch
    from torch_geometric.data import Batch

    if not graphs:
        raise BaceInferenceError("No graphs to embed.")

    encoder.eval()
    batch = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        out = encoder(batch)
        if isinstance(out, tuple):
            # GINEncoder returns (projection, graph_embedding, ...)
            if len(out) < 2:
                raise BaceInferenceError("Encoder return value missing graph embedding.")
            graph_embedding = out[1]
        else:
            graph_embedding = out
    _require_finite("graph_embedding", graph_embedding)
    if graph_embedding.size(0) != len(graphs):
        raise BaceInferenceError(
            f"Expected {len(graphs)} embeddings, got {graph_embedding.size(0)}."
        )
    return graph_embedding.detach()


def support_graphs_from_manifest(manifest: dict[str, Any]) -> tuple[list, list[int], list[int]]:
    """Convert manifest support SMILES to graphs; preserve original labels."""
    graphs = []
    labels = []
    indices = []
    seen: set[int] = set()
    for row in manifest["support"]:
        idx = int(row["dataset_index"])
        label = int(row["label"])
        if idx in seen:
            raise BaceInferenceError(f"Duplicate support index {idx}.")
        seen.add(idx)
        if label not in (0, 1):
            raise BaceInferenceError(f"Invalid support label {label} at index {idx}.")
        try:
            data = smiles_to_pyg_data(row["smiles_canonical"])
        except GraphConversionError as exc:
            raise BaceInferenceError(
                f"Support index {idx} graph conversion failed: {exc}"
            ) from None
        # Tag for tests: ensure query is not silently merged into support elsewhere
        data.bace_support_index = idx
        graphs.append(data)
        labels.append(label)
        indices.append(idx)
    return graphs, labels, indices


@dataclass
class BacePrototypeModel:
    """
    Frozen support prototypes for scoring queries.

    Construct via from_checkpoint_and_manifest or from_encoder_and_manifest
    (tests may inject a fake encoder).
    """

    encoder: Any
    prototypes: Any
    support_labels: tuple[int, ...]
    support_dataset_indices: tuple[int, ...]
    support_manifest_id: str
    checkpoint_sha256: str
    device: str = "cpu"
    _support_size: int = 0

    @classmethod
    def from_encoder_and_manifest(
        cls,
        encoder: GraphEncoder,
        manifest: dict[str, Any],
        checkpoint_sha256: str = "test",
        device: str = "cpu",
    ) -> "BacePrototypeModel":
        graphs, labels, indices = support_graphs_from_manifest(manifest)
        encoder.eval()
        # Move encoder if it is an nn.Module
        if hasattr(encoder, "to"):
            encoder = encoder.to(device)
        support_emb = embed_graphs(encoder, graphs, device=device)
        prototypes = mean_prototypes_by_original_label(support_emb, labels)
        return cls(
            encoder=encoder,
            prototypes=prototypes.detach(),
            support_labels=tuple(labels),
            support_dataset_indices=tuple(indices),
            support_manifest_id=str(manifest.get("manifest_id", "")),
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            _support_size=len(indices),
        )

    @classmethod
    def from_checkpoint_and_manifest(
        cls,
        checkpoint_path: Path | str | None = None,
        manifest_path: Path | str = CANONICAL_MANIFEST,
        device: str = "cpu",
        *,
        dataset_inventory: str = "auto",
    ) -> "BacePrototypeModel":
        """
        Load encoder + pinned support manifest for interactive or offline use.

        Default ``dataset_inventory='auto'`` allows interactive CLI/visualization
        without a local OGB BACE cache. Offline evaluation should pass
        ``dataset_inventory='required'``.
        """
        ckpt_path = resolve_checkpoint_path(checkpoint_path)
        digest = sha256_file(ckpt_path)
        manifest = load_validated_support_manifest(
            manifest_path,
            dataset_inventory=dataset_inventory,
        )
        try:
            encoder, _ckpt = load_encoder_from_research_checkpoint(
                ckpt_path, map_location=device
            )
        except CheckpointCompatibilityError as exc:
            raise BaceInferenceError(str(exc)) from None
        return cls.from_encoder_and_manifest(
            encoder=encoder,
            manifest=manifest,
            checkpoint_sha256=digest,
            device=device,
        )

    def score_query_smiles(self, smiles: str | None) -> BaceInferenceResult:
        if smiles is None or not str(smiles).strip():
            raise BaceInferenceError("Empty query SMILES.")
        try:
            data = smiles_to_pyg_data(str(smiles).strip())
        except GraphConversionError as exc:
            raise BaceInferenceError(f"Query graph conversion failed: {exc}") from None

        # Query must not be appended to support
        if self._support_size <= 0:
            raise BaceInferenceError("Support set is empty.")

        emb = embed_graphs(self.encoder, [data], device=self.device)
        base = classify_query_embedding(emb, self.prototypes)
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles).strip())
        canonical = Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

        return BaceInferenceResult(
            predicted_class=base.predicted_class,
            predicted_class_name=base.predicted_class_name,
            distance_to_class_0=base.distance_to_class_0,
            distance_to_class_1=base.distance_to_class_1,
            distance_gap=base.distance_gap,
            checkpoint_sha256=self.checkpoint_sha256,
            support_manifest_id=self.support_manifest_id,
            calibrated_probability=False,
            support_dataset_indices=self.support_dataset_indices,
            query_smiles_canonical=canonical,
        )

    def score_query_embedding(self, query_embedding) -> BaceInferenceResult:
        base = classify_query_embedding(query_embedding, self.prototypes)
        return BaceInferenceResult(
            predicted_class=base.predicted_class,
            predicted_class_name=base.predicted_class_name,
            distance_to_class_0=base.distance_to_class_0,
            distance_to_class_1=base.distance_to_class_1,
            distance_gap=base.distance_gap,
            checkpoint_sha256=self.checkpoint_sha256,
            support_manifest_id=self.support_manifest_id,
            calibrated_probability=False,
            support_dataset_indices=self.support_dataset_indices,
        )


def documented_class_name(label: int) -> str:
    if label not in (0, 1):
        raise BaceInferenceError(f"Invalid label {label}.")
    # Keep CLASS_NAME short; LABEL_SEMANTICS holds the long form.
    _ = LABEL_SEMANTICS[str(label)]
    return CLASS_NAME[label]
