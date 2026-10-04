"""
Interactive multi-panel visualization for BACE, ESOL, and Tox21 predictions.

Reuses existing inference classes:
  - apps.mol_cli.bace_infer.BacePrototypeModel
  - apps.mol_cli.esol_infer.EsolSolubilityModel
  - apps.mol_cli.tox21_infer.Tox21MultiTaskModel

Does not retrain models, alter checkpoints, or evaluate locked test sets.
A visualization does not establish model correctness or causal explanations.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_visualization_dependencies() -> list[str]:
    """Return missing package names required for this visualizer."""
    missing: list[str] = []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import Draw  # noqa: F401
        from rdkit.Chem import rdDepictor  # noqa: F401
    except ImportError:
        missing.append("rdkit")
    return missing


@dataclass
class TrackResult:
    """One model track: either a successful payload or an explicit error."""

    name: str
    ok: bool
    data: Any = None
    error: str | None = None


@dataclass
class LoadedModels:
    bace: Any | None
    esol: Any | None
    tox21: Any | None
    bace_error: str | None
    esol_error: str | None
    tox21_error: str | None


def load_models() -> LoadedModels:
    """Load each track independently so one failure does not block the others."""
    bace = esol = tox21 = None
    bace_error = esol_error = tox21_error = None

    try:
        from apps.mol_cli.bace_infer import BaceInferenceError, BacePrototypeModel

        print("Loading BACE prototype model...")
        bace = BacePrototypeModel.from_checkpoint_and_manifest()
        print(
            f"  BACE ready (manifest {bace.support_manifest_id}, "
            f"ckpt {bace.checkpoint_sha256[:8]}...)"
        )
    except Exception as exc:
        bace_error = str(exc)
        print(f"  BACE unavailable: {bace_error}", file=sys.stderr)

    try:
        from apps.mol_cli.esol_infer import EsolInferenceError, EsolSolubilityModel

        print("Loading ESOL solubility model...")
        esol = EsolSolubilityModel()
        print("  ESOL ready (frozen GIN + Phase 4E.3 head)")
    except Exception as exc:
        esol_error = str(exc)
        print(f"  ESOL unavailable: {esol_error}", file=sys.stderr)

    try:
        from apps.mol_cli.tox21_infer import Tox21InferenceError, Tox21MultiTaskModel

        print("Loading Tox21 multi-task model...")
        tox21 = Tox21MultiTaskModel()
        print(f"  Tox21 ready ({len(tox21.tasks)} tasks)")
    except Exception as exc:
        tox21_error = str(exc)
        print(f"  Tox21 unavailable: {tox21_error}", file=sys.stderr)

    if bace is None and esol is None and tox21 is None:
        raise RuntimeError(
            "No prediction models could be loaded. "
            f"BACE: {bace_error}; ESOL: {esol_error}; Tox21: {tox21_error}"
        )
    return LoadedModels(
        bace=bace,
        esol=esol,
        tox21=tox21,
        bace_error=bace_error,
        esol_error=esol_error,
        tox21_error=tox21_error,
    )


def parse_molecule(smiles: str):
    """Parse SMILES with RDKit; raise ValueError on invalid/empty input."""
    from rdkit import Chem

    if smiles is None or not str(smiles).strip():
        raise ValueError("Empty SMILES.")
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if mol.GetNumAtoms() <= 0:
        raise ValueError(f"SMILES has no atoms: {smiles!r}")
    return mol, smiles


def collect_predictions(smiles: str, models: LoadedModels) -> dict[str, TrackResult]:
    """Run each available model; never invent placeholder prediction values."""
    from apps.mol_cli.bace_infer import BaceInferenceError
    from apps.mol_cli.esol_infer import EsolInferenceError, TARGET_DESCRIPTION
    from apps.mol_cli.ogb_graph import GraphConversionError
    from apps.mol_cli.tox21_infer import Tox21InferenceError

    out: dict[str, TrackResult] = {}

    if models.bace is None:
        out["bace"] = TrackResult(
            "BACE", False, error=models.bace_error or "BACE model not loaded"
        )
    else:
        try:
            result = models.bace.score_query_smiles(smiles)
            out["bace"] = TrackResult("BACE", True, data=result)
        except (BaceInferenceError, GraphConversionError, Exception) as exc:
            out["bace"] = TrackResult("BACE", False, error=str(exc))

    if models.esol is None:
        out["esol"] = TrackResult(
            "ESOL", False, error=models.esol_error or "ESOL model not loaded"
        )
    else:
        try:
            value = models.esol.predict_log_solubility(smiles)
            if not math.isfinite(value):
                raise EsolInferenceError(f"Non-finite ESOL prediction: {value}")
            out["esol"] = TrackResult(
                "ESOL",
                True,
                data={"value": value, "scale": TARGET_DESCRIPTION},
            )
        except (EsolInferenceError, GraphConversionError, Exception) as exc:
            out["esol"] = TrackResult("ESOL", False, error=str(exc))

    if models.tox21 is None:
        out["tox21"] = TrackResult(
            "Tox21", False, error=models.tox21_error or "Tox21 model not loaded"
        )
    else:
        try:
            rows = models.tox21.predict_task_scores(smiles)
            if len(rows) != 12:
                raise Tox21InferenceError(
                    f"Expected 12 Tox21 tasks, got {len(rows)}"
                )
            for row in rows:
                if not math.isfinite(float(row["score"])):
                    raise Tox21InferenceError(
                        f"Non-finite Tox21 score for {row['task']}"
                    )
            out["tox21"] = TrackResult("Tox21", True, data=rows)
        except (Tox21InferenceError, GraphConversionError, Exception) as exc:
            out["tox21"] = TrackResult("Tox21", False, error=str(exc))

    return out


def _draw_structure(ax, mol, smiles: str) -> None:
    from rdkit.Chem import Draw

    ax.set_title("2D structure", fontsize=11, pad=4)
    ax.axis("off")
    try:
        img = Draw.MolToImage(mol, size=(480, 380))
        ax.imshow(img, aspect="equal")
        # Crop whitespace around the rendered molecule.
        ax.margins(0.02)
        ax.text(
            0.5,
            0.02,
            smiles,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Structure render failed:\n{exc}",
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )


def _draw_molecular_graph(ax, mol) -> None:
    """Atom/bond graph using RDKit 2D coordinates (same molecule as structure)."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    ax.set_title("Molecular graph (atoms / bonds)", fontsize=11, pad=4)
    try:
        mol = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        xs, ys, labels = [], [], []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xs.append(pos.x)
            ys.append(pos.y)
            labels.append(f"{atom.GetSymbol()}{atom.GetIdx()}")

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#555555", lw=1.8, zorder=1)

        ax.scatter(xs, ys, s=320, c="#dceaf7", edgecolors="#1f4e79", zorder=2)
        for x, y, lab in zip(xs, ys, labels):
            ax.text(x, y, lab, ha="center", va="center", fontsize=8, zorder=3)

        # Tight equal-aspect view so small molecules fill the panel.
        pad = 0.55
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    except Exception as exc:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Graph render failed:\n{exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def _panel_error(ax, title: str, message: str) -> None:
    ax.set_title(title, fontsize=11, pad=4)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Unavailable\n{message}",
        ha="center",
        va="center",
        wrap=True,
        transform=ax.transAxes,
        color="#8b0000",
        fontsize=9,
    )


def _draw_bace(ax, track: TrackResult) -> None:
    title = "BACE few-shot prototypes"
    if not track.ok:
        _panel_error(ax, title, track.error or "unknown error")
        return

    result = track.data
    labels = ["inactive (0)", "active (1)"]
    distances = [result.distance_to_class_0, result.distance_to_class_1]
    colors = ["#6c8ebf", "#d79b00"]
    bars = ax.bar(labels, distances, color=colors, edgecolor="#333333", width=0.62)
    pred_idx = int(result.predicted_class)
    bars[pred_idx].set_edgecolor("#c00000")
    bars[pred_idx].set_linewidth(2.5)
    ax.set_ylabel("Euclidean distance", fontsize=9)
    # Keep the title short so it fits the top-right panel width.
    ax.set_title(
        f"{title}\n"
        f"pred: {result.predicted_class_name} (class {result.predicted_class})",
        fontsize=9.5,
        pad=6,
    )
    ymax = max(distances) if max(distances) > 0 else 1.0
    ax.set_ylim(0, ymax * 1.20)
    for bar, dist in zip(bars, distances):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            dist + 0.015 * ymax,
            f"{dist:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel(
        "Distances are not probabilities / calibrated confidence",
        fontsize=7.5,
        style="italic",
        labelpad=4,
    )


def _draw_esol(ax, track: TrackResult) -> None:
    title = "ESOL log solubility"
    if not track.ok:
        _panel_error(ax, title, track.error or "unknown error")
        return

    value = float(track.data["value"])
    scale = str(track.data["scale"])

    # Vertical single-bar chart; value shown in title + above the bar (in-axes).
    bars = ax.bar([0], [value], color="#5b9bd5", edgecolor="#333333", width=0.55)
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["prediction"], fontsize=9)
    # Full scale text on x-axis (avoids a long left-side ylabel that can clip).
    ax.set_xlabel(
        f"{scale}\nModel prediction on original measured scale; not experimental",
        fontsize=7.5,
        labelpad=10,
    )
    ax.set_title(
        f"{title}\npredicted value = {value:.4f}",
        fontsize=9.5,
        pad=6,
    )

    pad = max(0.12, abs(value) * 0.45, 0.20)
    y_lo = min(0.0, value) - 0.08 * pad
    y_hi = max(0.0, value) + pad
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.8, 0.8)

    bar = bars[0]
    # Annotation just above the bar, still inside the padded y-limit.
    y_ann = value + 0.18 * pad if value >= 0 else value - 0.18 * pad
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_ann,
        f"{value:.4f}",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=11,
        fontweight="bold",
        color="#102a43",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#5b9bd5",
            "alpha": 0.95,
        },
        clip_on=True,
        zorder=5,
    )
    ax.tick_params(axis="y", labelsize=8)


def _draw_tox21(ax, track: TrackResult) -> None:
    title = "Tox21 task scores (sigmoid)"
    if not track.ok:
        _panel_error(ax, title, track.error or "unknown error")
        return

    rows = track.data
    tasks = [str(r["task"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    # Reverse so first saved-order task appears at the top.
    tasks_rev = list(reversed(tasks))
    scores_rev = list(reversed(scores))
    y = list(range(len(tasks_rev)))
    bars = ax.barh(y, scores_rev, color="#70ad47", edgecolor="#333333", height=0.70)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks_rev, fontsize=8)

    # Keep original sigmoid scores; zoom x-axis to the data range so small
    # values remain distinguishable. This does not change the scores.
    max_score = max(scores_rev) if scores_rev else 0.0
    if max_score <= 0:
        x_max = 0.01
    elif max_score < 0.2:
        # Headroom for numeric labels just past the longest bar.
        x_max = max_score * 1.28
    else:
        x_max = min(max(max_score * 1.15, 0.05), 1.0)
    ax.set_xlim(0.0, x_max)

    for bar, score in zip(bars, scores_rev):
        width = bar.get_width()
        if width < 0.82 * x_max:
            ax.text(
                width + 0.012 * x_max,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                ha="left",
                fontsize=7.5,
                clip_on=True,
            )
        else:
            ax.text(
                max(width - 0.01 * x_max, 0.0),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                ha="right",
                fontsize=7.5,
                color="white",
                clip_on=True,
            )

    ax.set_xlabel(
        "sigmoid(logit) model score (axis zoomed for readability; "
        "not a calibrated probability)",
        fontsize=8,
        labelpad=4,
    )
    ax.set_title(
        f"{title}\n"
        f"{len(tasks)} tasks in saved model order; "
        "not a clinical toxicity determination",
        fontsize=9.5,
        pad=6,
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.55)


def build_figure(smiles: str, mol, predictions: dict[str, TrackResult]):
    """Build one multi-panel Matplotlib figure; caller may show or close it."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # constrained_layout only (do not also call tight_layout).
    # Slightly smaller canvas so the full figure fits a normal desktop window.
    fig = plt.figure(figsize=(12.8, 8.8), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.12,   # keep BACE right spine/labels inside the canvas
        h_pad=0.14,   # extra bottom room for the ESOL x-axis label
        hspace=0.08,
        wspace=0.06,
    )
    fig.suptitle(
        f"Molecular property predictions for {smiles}\n"
        "Visualization only - not model validation, clinical toxicity, "
        "or causal explanation",
        fontsize=11.5,
    )

    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.0, 1.40],
        width_ratios=[1.0, 1.0, 1.10],
    )
    ax_struct = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_bace = fig.add_subplot(gs[0, 2])
    ax_esol = fig.add_subplot(gs[1, 0])
    ax_tox = fig.add_subplot(gs[1, 1:])

    _draw_structure(ax_struct, mol, smiles)
    _draw_molecular_graph(ax_graph, mol)
    _draw_bace(ax_bace, predictions["bace"])
    _draw_esol(ax_esol, predictions["esol"])
    _draw_tox21(ax_tox, predictions["tox21"])

    return fig


def visualize_smiles(
    smiles: str,
    models: LoadedModels,
    *,
    show: bool = True,
):
    """
    Parse SMILES, run available models, and build (optionally show) the figure.

    Returns (figure, predictions). Does not fabricate missing-track values.
    """
    import matplotlib.pyplot as plt

    mol, smiles = parse_molecule(smiles)
    predictions = collect_predictions(smiles, models)

    print()
    print(f"Input SMILES: {smiles}")
    for key in ("bace", "esol", "tox21"):
        track = predictions[key]
        if track.ok:
            if key == "bace":
                r = track.data
                print(
                    f"  BACE: class={r.predicted_class_name} "
                    f"(d0={r.distance_to_class_0:.4f}, d1={r.distance_to_class_1:.4f}); "
                    "distances are not probabilities"
                )
            elif key == "esol":
                print(
                    f"  ESOL: {track.data['value']:.4f} "
                    f"({track.data['scale']})"
                )
            else:
                print(f"  Tox21: {len(track.data)} task scores ready")
        else:
            print(f"  {track.name} FAILED: {track.error}")
    print()

    fig = build_figure(smiles, mol, predictions)
    if show:
        plt.show()
    return fig, predictions


def interactive_loop(models: LoadedModels) -> int:
    import matplotlib.pyplot as plt

    print()
    print("Interactive prediction visualizer (BACE + ESOL + Tox21)")
    print("Enter a SMILES string to plot, or quit / q / exit to leave.")
    print("Close the figure window to continue entering molecules.")
    print()

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
            fig, _preds = visualize_smiles(text, models, show=True)
            plt.close(fig)
        except ValueError as exc:
            print(f"Invalid SMILES: {exc}")
        except Exception as exc:
            print(f"Visualization failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    del argv
    missing = check_visualization_dependencies()
    if missing:
        print(
            "Missing required package(s) for visualization: "
            + ", ".join(missing)
            + ".\nInstall the missing package(s) in your research Python "
            "environment, then re-run:\n"
            "  python -m apps.mol_cli.visualize_predictions",
            file=sys.stderr,
        )
        return 1

    # Also need torch / PyG / ogb for model inference paths.
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        from ogb.utils.mol import smiles2graph  # noqa: F401
    except ImportError as exc:
        print(
            "Missing inference dependency for model predictions: "
            f"{exc}. Need torch, torch_geometric, and ogb "
            "(see environment.txt).",
            file=sys.stderr,
        )
        return 1

    try:
        models = load_models()
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    return interactive_loop(models)


if __name__ == "__main__":
    raise SystemExit(main())
