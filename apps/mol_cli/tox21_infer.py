"""
Interactive Tox21 multi-task prediction (frozen GraphCL GIN + Phase 4F.4 head).

Pipeline (same as evaluate_tox21_baselines.py):
  SMILES -> OGB molecular graph -> frozen GIN graph embedding (128-d)
        -> multi-task head -> 12 logits -> sigmoid scores for display

Does not retrain models, touch frozen splits, or evaluate the locked test set.
Scores are model-derived (sigmoid of logits), not calibrated probabilities.
This tool does not make clinical toxicity determinations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch

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
    / "tox21"
    / "phase4f4_baselines"
    / "classification_head.pt"
)

# Fallback order matches evaluate_tox21_baselines.py / classification_head.pt["tasks"].
DEFAULT_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


class MultiTaskHead(nn.Module):
    """Phase 4F.4 head: Linear(128→64)→ReLU→Dropout→Linear(64→12)."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 64,
        n_tasks: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Tox21InferenceError(RuntimeError):
    """Missing files, bad checkpoint payload, or inference failure."""


class Tox21MultiTaskModel:
    """Frozen GraphCL encoder + Phase 4F.4 multi-task classification head."""

    def __init__(
        self,
        encoder_ckpt: Path | str = DEFAULT_ENCODER_CKPT,
        head_ckpt: Path | str = DEFAULT_HEAD_CKPT,
    ):
        self.encoder_ckpt = Path(encoder_ckpt)
        self.head_ckpt = Path(head_ckpt)

        if not self.encoder_ckpt.is_file():
            raise Tox21InferenceError(
                f"GraphCL checkpoint not found: {self.encoder_ckpt}"
            )
        if not self.head_ckpt.is_file():
            raise Tox21InferenceError(
                f"Tox21 classification head not found: {self.head_ckpt}"
            )

        # 1) Load the research GIN encoder with the safe weights_only loader.
        try:
            self.encoder, _ckpt = load_encoder_from_research_checkpoint(
                self.encoder_ckpt, map_location="cpu"
            )
        except CheckpointCompatibilityError as exc:
            raise Tox21InferenceError(str(exc)) from None
        except Exception as exc:
            raise Tox21InferenceError(
                f"Failed to load GraphCL encoder from {self.encoder_ckpt}: {exc}"
            ) from None

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # 2) Load the trained multi-task head (BCEWithLogits → raw logits).
        try:
            payload = torch.load(
                self.head_ckpt, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise Tox21InferenceError(
                f"Failed to load classification head from {self.head_ckpt}: {exc}"
            ) from None

        if not isinstance(payload, dict):
            raise Tox21InferenceError(
                "classification_head.pt root must be a dict of tensors/primitives."
            )
        for key in ("head_config", "head_state_dict"):
            if key not in payload:
                raise Tox21InferenceError(
                    f"classification_head.pt missing required key {key!r}."
                )

        cfg = payload["head_config"]
        n_tasks = int(cfg["n_tasks"])
        self.head = MultiTaskHead(
            in_dim=int(cfg["in_dim"]),
            hidden=int(cfg["hidden"]),
            n_tasks=n_tasks,
            dropout=float(cfg["dropout"]),
        )
        self.head.load_state_dict(payload["head_state_dict"])
        self.head.eval()

        # Prefer task names/order saved in the head checkpoint.
        tasks = payload.get("tasks")
        if isinstance(tasks, list) and len(tasks) == n_tasks:
            self.tasks = [str(t) for t in tasks]
        elif n_tasks == len(DEFAULT_TASKS):
            self.tasks = list(DEFAULT_TASKS)
        else:
            raise Tox21InferenceError(
                f"Cannot resolve task names for n_tasks={n_tasks}."
            )

        if len(self.tasks) != 12:
            raise Tox21InferenceError(
                f"Expected 12 Tox21 tasks, got {len(self.tasks)}."
            )

    def predict_task_scores(self, smiles: str) -> list[dict[str, float | str]]:
        """
        Predict per-task logits and sigmoid scores for one SMILES.

        Steps:
          SMILES -> PyG Data (OGB atom/bond features)
                -> Batch of size 1
                -> encoder(...)[1] graph embedding (128-d; not the projection head)
                -> multi-task head -> 12 logits
                -> score = sigmoid(logit) for display only

        Primary research metrics (ROC-AUC / AP) are threshold-free. Reports also
        list Acc@0 (hard label at logit > 0), but that is not treated here as a
        calibrated clinical decision rule — only scores are shown.
        """
        if smiles is None or not str(smiles).strip():
            raise GraphConversionError("Empty SMILES.")

        smiles = str(smiles).strip()
        graph = smiles_to_pyg_data(smiles)
        batch = Batch.from_data_list([graph])

        with torch.no_grad():
            # Same embedding index as evaluate_tox21_baselines.py: encoder(...)[1]
            graph_embedding = self.encoder(batch)[1]
            logits = self.head(graph_embedding).reshape(-1)
            scores = torch.sigmoid(logits)

        if int(logits.numel()) != len(self.tasks):
            raise Tox21InferenceError(
                f"Head returned {int(logits.numel())} logits; "
                f"expected {len(self.tasks)}."
            )

        results: list[dict[str, float | str]] = []
        for i, task in enumerate(self.tasks):
            logit = float(logits[i].item())
            score = float(scores[i].item())
            if not math.isfinite(logit) or not math.isfinite(score):
                raise Tox21InferenceError(
                    f"Non-finite output for task {task}: logit={logit}, score={score}"
                )
            results.append(
                {
                    "task": task,
                    "logit": logit,
                    "score": score,
                }
            )
        return results


def _print_banner() -> None:
    print("Tox21 interactive multi-task predictor")
    print("Model: frozen GraphCL GIN + Phase 4F.4 classification head")
    print("Outputs: 12 assay task scores = sigmoid(logit)")
    print(
        "Scores are model-derived, not calibrated probabilities, "
        "and are not a clinical toxicity determination."
    )
    print("Enter a SMILES string to predict, or quit / q / exit to leave.")
    print()


def _print_prediction(smiles: str, rows: list[dict[str, float | str]]) -> None:
    print()
    print(f"  Input SMILES: {smiles}")
    print("  Task scores (sigmoid of model logits; not calibrated probabilities):")
    print(f"  {'Task':<14}  {'Logit':>10}  {'Score':>10}")
    print(f"  {'-' * 14}  {'-' * 10}  {'-' * 10}")
    for row in rows:
        print(
            f"  {row['task']:<14}  {row['logit']:10.4f}  {row['score']:10.4f}"
        )
    print(
        "  Note: model prediction for research Tox21 assay heads only — "
        "not an experimental measurement or clinical judgment."
    )
    print()


def interactive_loop(model: Tox21MultiTaskModel) -> int:
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
            rows = model.predict_task_scores(text)
        except GraphConversionError as exc:
            print(f"Could not convert SMILES to a molecular graph: {exc}")
            continue
        except Tox21InferenceError as exc:
            print(f"Inference error: {exc}")
            continue
        except Exception as exc:
            print(f"Unexpected error during prediction: {exc}")
            continue

        if len(rows) != 12:
            print(f"Inference error: expected 12 task outputs, got {len(rows)}.")
            continue

        _print_prediction(text, rows)


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved for future optional paths
    try:
        model = Tox21MultiTaskModel()
    except Tox21InferenceError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    return interactive_loop(model)


if __name__ == "__main__":
    raise SystemExit(main())
