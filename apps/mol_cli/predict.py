"""Interactive molecular summary CLI (descriptors + optional OGB graph build)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Any, Sequence


def _print_rdkit_missing() -> None:
    print(
        "RDKit is not available in this Python interpreter.\n"
        "This CLI does not install packages. Use the environment that "
        "already has RDKit (see the repository environment.txt), then run:\n"
        "  python -m apps.mol_cli.predict",
        file=sys.stderr,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive molecular CLI: RDKit descriptors, optional OGB graph "
            "build, and optionally gated experimental BACE few-shot prototypes."
        )
    )
    parser.add_argument(
        "--bace",
        action="store_true",
        help=(
            "Enable experimental BACE few-shot prototype scoring "
            "(explicit opt-in; loads the GraphCL checkpoint once at startup)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _try_build_graph_summary(canonical_smiles: str) -> dict | None:
    """
    Lazily convert SMILES to an OGB graph summary.

    Returns None if OGB/torch/PyG are unavailable. Raises GraphConversionError
    for chemistry/conversion failures (caller may catch).
    """
    from .ogb_graph import (
        GraphConversionError,
        ogb_graph_deps_available,
        smiles_to_pyg_data,
        summarize_pyg_data,
    )

    ok, detail = ogb_graph_deps_available()
    if not ok:
        print(f"Note: {detail}")
        return None

    data = smiles_to_pyg_data(canonical_smiles)
    return asdict(summarize_pyg_data(canonical_smiles, data))


def _initialize_bace_model() -> Any:
    """
    Lazily import and build BacePrototypeModel once.

    Requires OGB graph dependencies. Raises BaceInferenceError (or wraps) on
    failure. Does not fall back to random weights.
    """
    from .ogb_graph import ogb_graph_deps_available

    ok, detail = ogb_graph_deps_available()
    if not ok:
        from .bace_infer import BaceInferenceError

        raise BaceInferenceError(
            "BACE mode requires OGB graph dependencies "
            f"(torch, torch_geometric, ogb). {detail}"
        )

    from .bace_infer import BacePrototypeModel

    return BacePrototypeModel.from_checkpoint_and_manifest()


def _print_startup_banner(*, bace_enabled: bool) -> None:
    print("Molecular property CLI (RDKit descriptors + optional OGB graph build)")
    if bace_enabled:
        print(
            "Experimental BACE mode ENABLED (few-shot prototypes). "
            "Other learned endpoints remain unavailable."
        )
        print(
            "BACE uses a fixed train-only support manifest and the GraphCL "
            "checkpoint. Distances are not calibrated probabilities."
        )
    else:
        print(
            "Learned BACE, solubility, toxicity, and ADMET predictions "
            "are not available."
        )
        print(
            "Graph conversion is not a prediction and does not load "
            "the GIN checkpoint."
        )
    print("Type a SMILES string, or quit / q / exit to leave.")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        _print_rdkit_missing()
        return 1

    from .descriptors import compute_descriptors
    from .ogb_graph import GraphConversionError
    from .report import format_report
    from .smiles_input import SmilesInputError, is_quit_command, parse_smiles

    bace_model = None
    if args.bace:
        # Lazy import only on the explicit --bace path.
        from .bace_infer import BaceInferenceError

        try:
            print("Initializing experimental BACE prototype model...")
            bace_model = _initialize_bace_model()
            print(
                f"Loaded support manifest {bace_model.support_manifest_id}; "
                f"checkpoint SHA-256 prefix "
                f"{bace_model.checkpoint_sha256[:8]}..."
            )
            print()
        except BaceInferenceError as exc:
            print(f"BACE initialization failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            # Startup must not enter the REPL with an unusable model.
            print(f"BACE initialization failed: {exc}", file=sys.stderr)
            return 1

    _print_startup_banner(bace_enabled=bace_model is not None)

    while True:
        try:
            raw = input("SMILES> ")
        except EOFError:
            print("\nExiting.")
            return 0
        except KeyboardInterrupt:
            print("\nExiting.")
            return 0

        if is_quit_command(raw):
            print("Goodbye.")
            return 0

        try:
            parsed = parse_smiles(raw)
            descriptors = compute_descriptors(parsed.mol)
            graph_summary = None
            try:
                graph_summary = _try_build_graph_summary(parsed.canonical_smiles)
            except GraphConversionError as exc:
                print(f"Graph conversion error: {exc}")

            bace_result = None
            if bace_model is not None:
                from .bace_infer import BaceInferenceError

                try:
                    bace_result = bace_model.score_query_smiles(
                        parsed.canonical_smiles
                    )
                except BaceInferenceError as exc:
                    print(f"BACE inference error: {exc}")
                    bace_result = None
                except Exception as exc:
                    print(f"BACE inference error: could not score this molecule ({exc}).")
                    bace_result = None

            print(
                format_report(
                    parsed,
                    descriptors,
                    graph_summary=graph_summary,
                    bace_result=bace_result,
                )
            )
        except SmilesInputError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Error: could not process this input ({exc}).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
