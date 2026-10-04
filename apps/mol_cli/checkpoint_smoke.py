"""
Safe local loading of the research GraphCL GINEncoder checkpoint.

Deserialization uses torch.load(..., weights_only=True). Architecture and
weights are then applied through GINEncoder.from_checkpoint on the loaded
dict (no second pickle pass with weights_only=False).

This is a software compatibility helper for CLI smoke tests. It does not
implement BACE prediction, prototypes, or accuracy claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Documented canonical research checkpoint (Phase 2B / experiment_config).
DEFAULT_CHECKPOINT_PATH = Path("outputs/core_pipeline/graphcl_checkpoint.pt")

DOCUMENTED_ENCODER_CONFIG = {
    "in_dim": 9,
    "hidden_dim": 128,
    "num_layers": 5,
    "dropout": 0.1,
    "pool_type": "sum",
}

DOCUMENTED_PARAM_COUNT = 184_197

REQUIRED_CHECKPOINT_KEYS = ("encoder_config", "encoder_state_dict")


class CheckpointCompatibilityError(RuntimeError):
    """Checkpoint missing, unsafe to load, or incompatible with the encoder."""


def checkpoint_file_info(path: Path | str = DEFAULT_CHECKPOINT_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointCompatibilityError(f"Checkpoint not found: {path}")
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
    }


def load_checkpoint_dict_safe(
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """
    Load checkpoint bytes with weights_only=True (tensors + primitives only).

    Does not use GINEncoder.from_checkpoint's path-string branch (that branch
    deserializes without weights-only mode). Pass a dict into from_checkpoint instead.
    """
    import torch

    path = Path(path)
    info = checkpoint_file_info(path)

    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Safe checkpoint load failed for {path} ({exc}). "
            "Refusing to fall back to non-weights-only deserialization in the CLI smoke path."
        ) from None

    if not isinstance(obj, dict):
        raise CheckpointCompatibilityError(
            f"Checkpoint root must be a dict, got {type(obj).__name__}."
        )

    for key in REQUIRED_CHECKPOINT_KEYS:
        if key not in obj:
            raise CheckpointCompatibilityError(
                f"Checkpoint missing required key {key!r}."
            )

    if not isinstance(obj["encoder_config"], dict):
        raise CheckpointCompatibilityError("encoder_config must be a dict.")
    if not isinstance(obj["encoder_state_dict"], dict):
        raise CheckpointCompatibilityError("encoder_state_dict must be a dict.")

    obj["_smoke_meta"] = info
    return obj


def load_encoder_from_research_checkpoint(
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    map_location: str = "cpu",
):
    """
    Build GINEncoder via the research from_checkpoint API on a safely loaded dict.
    """
    from src.models import GINEncoder

    ckpt = load_checkpoint_dict_safe(path, map_location=map_location)
    # Pass dict so from_checkpoint does not call torch.load again.
    encoder = GINEncoder.from_checkpoint(ckpt, map_location=map_location)
    encoder.eval()
    return encoder, ckpt


def assert_documented_config(encoder_config: dict[str, Any]) -> None:
    for key, expected in DOCUMENTED_ENCODER_CONFIG.items():
        if key not in encoder_config:
            raise CheckpointCompatibilityError(
                f"encoder_config missing {key!r}."
            )
        actual = encoder_config[key]
        if key == "dropout":
            if abs(float(actual) - float(expected)) > 1e-12:
                raise CheckpointCompatibilityError(
                    f"encoder_config[{key!r}]={actual!r}, expected {expected!r}."
                )
        elif actual != expected:
            raise CheckpointCompatibilityError(
                f"encoder_config[{key!r}]={actual!r}, expected {expected!r}."
            )
