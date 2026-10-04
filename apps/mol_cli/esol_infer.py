"""
Interactive ESOL solubility prediction (frozen GraphCL GIN + Phase 4E.3 head).

Pipeline (same as evaluate_frozen_gin_regression.py):
  SMILES -> OGB molecular graph -> frozen GIN graph embedding (128-d)
        -> regression head (standardized scale) -> denormalize to measured logS

Does not retrain models, touch frozen splits, or evaluate the locked test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch

# Ensure repo root is on sys.path when run as a script or module.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.mol_cli.checkpoint_smoke import (
    CheckpointCompatibilityError,
    load_encoder_from_research_checkpoint,
)
from apps.mol_cli.ogb_graph import GraphConversionError, smiles_to_pyg_data

DEFAULT_ENCODER_CKPT = (
    REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
)
DEFAULT_HEAD_CKPT = (
    REPO_ROOT
    / "outputs"
    / "esol"
    / "phase4e3_frozen_gin_regression"
    / "regression_head.pt"
)

# Target scale used when the head was trained / when validation metrics were reported.
TARGET_DESCRIPTION = "measured log solubility in mols per litre"


class RegressionHead(nn.Module):
    """Same architecture as Phase 4E.3: Linear(128→64)→ReLU→Linear(64→1)."""

    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EsolInferenceError(RuntimeError):
    """Missing files, bad checkpoint payload, or inference failure."""


class EsolSolubilityModel:
    """Frozen GraphCL encoder + Phase 4E.3 regression head for one-molecule inference."""

    def __init__(
        self,
        encoder_ckpt: Path | str = DEFAULT_ENCODER_CKPT,
        head_ckpt: Path | str = DEFAULT_HEAD_CKPT,
    ):
        self.encoder_ckpt = Path(encoder_ckpt)
        self.head_ckpt = Path(head_ckpt)

        if not self.encoder_ckpt.is_file():
            raise EsolInferenceError(
                f"GraphCL checkpoint not found: {self.encoder_ckpt}"
            )
        if not self.head_ckpt.is_file():
            raise EsolInferenceError(
                f"ESOL regression head not found: {self.head_ckpt}"
            )

        # 1) Load the research GIN encoder with the safe weights_only loader.
        try:
            self.encoder, _ckpt = load_encoder_from_research_checkpoint(
                self.encoder_ckpt, map_location="cpu"
            )
        except CheckpointCompatibilityError as exc:
            raise EsolInferenceError(str(exc)) from None
        except Exception as exc:
            raise EsolInferenceError(
                f"Failed to load GraphCL encoder from {self.encoder_ckpt}: {exc}"
            ) from None

        # Keep the encoder frozen (no gradient updates) and in eval mode.
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # 2) Load the trained regression head + train-only target normalization.
        try:
            payload = torch.load(
                self.head_ckpt, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise EsolInferenceError(
                f"Failed to load regression head from {self.head_ckpt}: {exc}"
            ) from None

        if not isinstance(payload, dict):
            raise EsolInferenceError(
                "regression_head.pt root must be a dict of tensors/primitives."
            )
        for key in ("head_config", "head_state_dict", "target_normalization"):
            if key not in payload:
                raise EsolInferenceError(
                    f"regression_head.pt missing required key {key!r}."
                )

        cfg = payload["head_config"]
        self.head = RegressionHead(
            in_dim=int(cfg["in_dim"]),
            hidden=int(cfg["hidden"]),
        )
        self.head.load_state_dict(payload["head_state_dict"])
        self.head.eval()

        # Head was trained on standardized targets: (y - train_mean) / train_std.
        # Predictions must be denormalized back to the original measured scale.
        norm = payload["target_normalization"]
        self.y_mean = float(norm["train_mean"])
        self.y_std = float(norm["train_std_population"])
        if self.y_std <= 0:
            raise EsolInferenceError(
                f"Invalid train_std_population in head checkpoint: {self.y_std}"
            )

    def predict_log_solubility(self, smiles: str) -> float:
        """
        Predict ESOL log solubility for one SMILES on the original measured scale.

        Steps:
          SMILES -> PyG Data (OGB atom/bond features)
                -> Batch of size 1
                -> encoder(...)[1] graph embedding (128-d; not the projection head)
                -> regression head (standardized units)
                -> pred * y_std + y_mean  (original measured logS scale)
        """
        if smiles is None or not str(smiles).strip():
            raise GraphConversionError("Empty SMILES.")

        smiles = str(smiles).strip()

        # Chemistry / graph conversion failures raise GraphConversionError.
        graph = smiles_to_pyg_data(smiles)
        batch = Batch.from_data_list([graph])

        with torch.no_grad():
            # GINEncoder returns (node_out, graph_embedding); index [1] is used
            # by the Phase 4E.3 evaluation script.
            graph_embedding = self.encoder(batch)[1]
            pred_standardized = self.head(graph_embedding)
            pred_original = pred_standardized * self.y_std + self.y_mean

        value = float(pred_original.reshape(-1)[0].item())
        return value


def _print_prediction(smiles: str, pred: float) -> None:
    print()
    print(f"  Input SMILES: {smiles}")
    print(f"  Predicted ESOL log solubility: {pred:.4f}")
    print(f"  Target scale: {TARGET_DESCRIPTION}")
    print(
        "  Note: this is a model prediction from the frozen GraphCL GIN + "
        "Phase 4E.3 regression head, not an experimental measurement."
    )
    print()


def _print_banner() -> None:
    print("ESOL interactive solubility predictor")
    print("Model: frozen GraphCL GIN + Phase 4E.3 regression head")
    print(f"Scale: {TARGET_DESCRIPTION}")
    print("Enter a SMILES string to predict, or quit / q / exit to leave.")
    print()


def interactive_loop(model: EsolSolubilityModel) -> int:
    _print_banner()
    while True:
        try:
            raw = input("SMILES> ")
        except (EOFError, KeyboardInterrupt):
            print()
            print("Exiting.")
            return 0

        text = raw.strip()
        if not text:
            print("Empty input. Enter a SMILES string, or quit / q / exit.")
            continue
        if text.lower() in {"quit", "q", "exit"}:
            print("Exiting.")
            return 0

        try:
            pred = model.predict_log_solubility(text)
        except GraphConversionError as exc:
            print(f"Could not convert SMILES to a molecular graph: {exc}")
            continue
        except EsolInferenceError as exc:
            print(f"Inference error: {exc}")
            continue
        except Exception as exc:
            print(f"Unexpected error during prediction: {exc}")
            continue

        _print_prediction(text, pred)


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved for future optional paths; keep CLI simple for now
    try:
        model = EsolSolubilityModel()
    except EsolInferenceError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    return interactive_loop(model)


if __name__ == "__main__":
    raise SystemExit(main())
