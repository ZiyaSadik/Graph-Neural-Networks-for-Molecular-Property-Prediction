"""
Orchestrate existing mol_cli inference into one structured molecular report.

Never substitutes a random/untrained model when a checkpoint is missing.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

GRAPHCL_CKPT = REPO_ROOT / "outputs" / "core_pipeline" / "graphcl_checkpoint.pt"
ESOL_HEAD = (
    REPO_ROOT / "outputs" / "esol" / "phase4e3_frozen_gin_regression" / "regression_head.pt"
)
TOX21_HEAD = (
    REPO_ROOT / "outputs" / "tox21" / "phase4f4_baselines" / "classification_head.pt"
)
BACE_MANIFEST = (
    REPO_ROOT / "apps" / "mol_cli" / "support_manifests" / "bace_train_k5_v1.json"
)
BACE_EVAL_REPORT = (
    REPO_ROOT / "apps" / "mol_cli" / "evaluation_reports" / "bace_fixed_support_eval_v1.json"
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ComponentStatus:
    available: bool
    detail: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None


@dataclass
class MolecularReport:
    input_smiles: str | None
    valid: bool
    error: str | None
    canonical_smiles: str | None = None
    num_atoms: int | None = None
    num_bonds: int | None = None
    structure_png: bytes | None = None
    descriptors: dict[str, float | int] | None = None
    descriptors_are_calculated_not_learned: bool = True
    bace: dict[str, Any] | None = None
    esol: dict[str, Any] | None = None
    tox21: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    explanation_png: bytes | None = None
    reliability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # bytes not JSON-friendly
        d["structure_png"] = None if self.structure_png is None else "<png bytes>"
        d["explanation_png"] = None if self.explanation_png is None else "<png bytes>"
        return d


class ChemistAnalysisService:
    """Lazy-loads existing task models; reports missing checkpoints explicitly."""

    def __init__(self):
        self._bace = None
        self._esol = None
        self._tox21 = None
        self._bace_error: str | None = None
        self._esol_error: str | None = None
        self._tox21_error: str | None = None
        self.status = self._probe_status()

    def _probe_status(self) -> dict[str, ComponentStatus]:
        return {
            "graphcl_encoder": ComponentStatus(
                available=GRAPHCL_CKPT.is_file(),
                detail="Shared GraphCL GIN encoder",
                checkpoint_path=str(GRAPHCL_CKPT.relative_to(REPO_ROOT)).replace("\\", "/"),
                checkpoint_sha256=_sha256(GRAPHCL_CKPT),
            ),
            "bace_manifest": ComponentStatus(
                available=BACE_MANIFEST.is_file(),
                detail="Fixed train K=5 support manifest (Protocol B)",
                checkpoint_path=str(BACE_MANIFEST.relative_to(REPO_ROOT)).replace("\\", "/"),
                checkpoint_sha256=_sha256(BACE_MANIFEST),
            ),
            "esol_head": ComponentStatus(
                available=ESOL_HEAD.is_file(),
                detail="ESOL regression head (Phase 4E.3)",
                checkpoint_path=str(ESOL_HEAD.relative_to(REPO_ROOT)).replace("\\", "/"),
                checkpoint_sha256=_sha256(ESOL_HEAD),
            ),
            "tox21_head": ComponentStatus(
                available=TOX21_HEAD.is_file(),
                detail="Tox21 12-task classification head (Phase 4F.4)",
                checkpoint_path=str(TOX21_HEAD.relative_to(REPO_ROOT)).replace("\\", "/"),
                checkpoint_sha256=_sha256(TOX21_HEAD),
            ),
        }

    def _ensure_bace(self):
        if self._bace is not None or self._bace_error:
            return
        if not GRAPHCL_CKPT.is_file() or not BACE_MANIFEST.is_file():
            self._bace_error = (
                "BACE unavailable: missing GraphCL checkpoint or support manifest. "
                "Refusing to use an untrained substitute."
            )
            return
        try:
            from apps.mol_cli.bace_infer import BacePrototypeModel

            self._bace = BacePrototypeModel.from_checkpoint_and_manifest(
                checkpoint_path=GRAPHCL_CKPT,
                manifest_path=BACE_MANIFEST,
                device="cpu",
                dataset_inventory="auto",
            )
        except Exception as exc:
            self._bace_error = f"BACE model failed to load: {exc}"

    def _ensure_esol(self):
        if self._esol is not None or self._esol_error:
            return
        if not GRAPHCL_CKPT.is_file() or not ESOL_HEAD.is_file():
            self._esol_error = (
                "ESOL unavailable: missing encoder or regression_head.pt. "
                "Refusing to use an untrained substitute."
            )
            return
        try:
            from apps.mol_cli.esol_infer import EsolSolubilityModel

            self._esol = EsolSolubilityModel(
                encoder_ckpt=GRAPHCL_CKPT, head_ckpt=ESOL_HEAD
            )
        except Exception as exc:
            self._esol_error = f"ESOL model failed to load: {exc}"

    def _ensure_tox21(self):
        if self._tox21 is not None or self._tox21_error:
            return
        if not GRAPHCL_CKPT.is_file() or not TOX21_HEAD.is_file():
            self._tox21_error = (
                "Tox21 unavailable: missing encoder or classification_head.pt. "
                "Refusing to use an untrained substitute."
            )
            return
        try:
            from apps.mol_cli.tox21_infer import Tox21MultiTaskModel

            self._tox21 = Tox21MultiTaskModel(
                encoder_ckpt=GRAPHCL_CKPT, head_ckpt=TOX21_HEAD
            )
        except Exception as exc:
            self._tox21_error = f"Tox21 model failed to load: {exc}"

    def analyze(
        self,
        smiles: str,
        *,
        run_explanation: bool = True,
        explain_epochs: int = 40,
        explain_seed: int = 0,
    ) -> MolecularReport:
        from apps.mol_cli.descriptors import compute_descriptors
        from apps.mol_cli.smiles_input import SmilesInputError, parse_smiles
        from apps.chemist_app.drawing import (
            mol_to_png_with_atom_importance,
            mol_to_png_with_atom_indices,
        )

        try:
            parsed = parse_smiles(smiles)
        except SmilesInputError as exc:
            return MolecularReport(
                input_smiles=smiles,
                valid=False,
                error=str(exc),
                reliability=self._reliability_base(input_ok=False),
            )

        try:
            structure_png = mol_to_png_with_atom_indices(parsed.mol)
        except Exception as exc:
            structure_png = None
            struct_err = str(exc)
        else:
            struct_err = None

        descriptors = compute_descriptors(parsed.mol)

        # --- BACE ---
        bace_block: dict[str, Any]
        self._ensure_bace()
        if self._bace_error:
            bace_block = {"available": False, "error": self._bace_error}
        else:
            try:
                result = self._bace.score_query_smiles(parsed.canonical_smiles)
                bace_block = {
                    "available": True,
                    "protocol": (
                        "Protocol B: fixed K=5-per-class support from OGB BACE "
                        "scaffold TRAIN only (manifest bace_train_k5_v1, seed 42). "
                        "Not interchangeable with paper episodic ProtoNet (Protocol A)."
                    ),
                    "support_manifest_id": result.support_manifest_id,
                    "support_seed": 42,
                    "k_shot_per_class": 5,
                    "n_support_total": len(result.support_dataset_indices),
                    "support_dataset_indices": list(result.support_dataset_indices),
                    "predicted_class": result.predicted_class,
                    "predicted_class_name": result.predicted_class_name,
                    "distance_to_class_0_inactive": result.distance_to_class_0,
                    "distance_to_class_1_active": result.distance_to_class_1,
                    "distance_gap": result.distance_gap,
                    "calibrated_probability": False,
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "limitations": [
                        "Euclidean distances are NOT probabilities.",
                        "A completed inference does not mean the prediction is correct.",
                        "Pinned Protocol B test metrics (locked report): bal_acc≈0.66, "
                        "sensitivity≈0.43 — many actives may be predicted inactive.",
                        "Not clinical advice; applicability domain not formally certified.",
                    ],
                    "locked_eval_evidence": self._bace_locked_summary(),
                }
            except Exception as exc:
                bace_block = {"available": False, "error": f"BACE inference failed: {exc}"}

        # --- ESOL ---
        esol_block: dict[str, Any]
        self._ensure_esol()
        if self._esol_error:
            esol_block = {"available": False, "error": self._esol_error}
        else:
            try:
                from apps.mol_cli.esol_infer import TARGET_DESCRIPTION

                pred = float(self._esol.predict_log_solubility(parsed.canonical_smiles))
                esol_block = {
                    "available": True,
                    "predicted_logS": pred,
                    "target_scale": TARGET_DESCRIPTION,
                    "note": (
                        "Model estimate from frozen GraphCL GIN + Phase 4E.3 head — "
                        "not an experimental measurement."
                    ),
                    "locked_test_reference": (
                        "Locked-test MAE≈0.7245, RMSE≈1.0635, R²≈0.746 "
                        "(outputs/esol/phase4e_test_locked/); not re-tuned here."
                    ),
                }
            except Exception as exc:
                esol_block = {"available": False, "error": f"ESOL inference failed: {exc}"}

        # --- Tox21 ---
        tox_block: dict[str, Any]
        self._ensure_tox21()
        if self._tox21_error:
            tox_block = {"available": False, "error": self._tox21_error}
        else:
            try:
                rows = self._tox21.predict_task_scores(parsed.canonical_smiles)
                tox_block = {
                    "available": True,
                    "tasks": rows,
                    "score_definition": "sigmoid(logit) for display only",
                    "calibrated_probability": False,
                    "warnings": [
                        "Sigmoid scores are NOT established calibrated probabilities.",
                        "Do not classify the molecule as categorically safe or toxic "
                        "from these outputs alone.",
                        "Not a clinical toxicity determination.",
                    ],
                    "locked_test_reference": (
                        "Locked-test macro ROC-AUC≈0.7283 "
                        "(outputs/tox21/phase4f_test_locked/)."
                    ),
                }
            except Exception as exc:
                tox_block = {"available": False, "error": f"Tox21 inference failed: {exc}"}

        # --- Explainability (BACE only when model available) ---
        explanation = None
        explanation_png = None
        if run_explanation and bace_block.get("available"):
            from apps.chemist_app.explain import explain_bace_query

            expl = explain_bace_query(
                self._bace.encoder,
                self._bace.prototypes,
                parsed.canonical_smiles,
                epochs=explain_epochs,
                seed=explain_seed,
            )
            explanation = expl.to_dict()
            if expl.succeeded and expl.node_mask is not None:
                try:
                    import numpy as np

                    explanation_png = mol_to_png_with_atom_importance(
                        parsed.mol, np.asarray(expl.node_mask, dtype=np.float64)
                    )
                except Exception as exc:
                    explanation = {
                        **explanation,
                        "succeeded": False,
                        "validated_status": "failed",
                        "error": (
                            f"Mask computed but visualization failed (not inventing "
                            f"an alternate explanation): {exc}"
                        ),
                        "node_mask": None,
                        "edge_mask": None,
                    }
                    explanation_png = None
        elif run_explanation:
            explanation = {
                "succeeded": False,
                "validated_status": "not_run",
                "error": "BACE model unavailable; explanation not attempted.",
                "caveats": [],
            }

        reliability = self._reliability_base(input_ok=True)
        reliability["structure_draw_error"] = struct_err
        reliability["components"] = {
            k: asdict(v) for k, v in self.status.items()
        }
        reliability["bace_available"] = bool(bace_block.get("available"))
        reliability["esol_available"] = bool(esol_block.get("available"))
        reliability["tox21_available"] = bool(tox_block.get("available"))
        reliability["explanation_status"] = (
            None if explanation is None else explanation.get("validated_status")
        )
        reliability["applicability_domain"] = (
            "Formal applicability-domain checks are unavailable. "
            "Locked-test metrics summarize performance on held-out split molecules, "
            "not a certificate that this query is in-distribution."
        )

        return MolecularReport(
            input_smiles=parsed.original_smiles,
            valid=True,
            error=None,
            canonical_smiles=parsed.canonical_smiles,
            num_atoms=parsed.num_atoms,
            num_bonds=parsed.num_bonds,
            structure_png=structure_png,
            descriptors=descriptors,
            descriptors_are_calculated_not_learned=True,
            bace=bace_block,
            esol=esol_block,
            tox21=tox_block,
            explanation=explanation,
            explanation_png=explanation_png,
            reliability=reliability,
        )

    def _bace_locked_summary(self) -> dict[str, Any]:
        if not BACE_EVAL_REPORT.is_file():
            return {"available": False}
        import json

        rep = json.loads(BACE_EVAL_REPORT.read_text(encoding="utf-8"))
        return {
            "available": True,
            "report_id": rep.get("report_id"),
            "validation_metrics": {
                k: rep["validation"]["metrics"].get(k)
                for k in (
                    "balanced_accuracy",
                    "sensitivity_class_1",
                    "specificity_class_0",
                    "auroc",
                )
            },
            "test_metrics": {
                k: rep["test"]["metrics"].get(k)
                for k in (
                    "balanced_accuracy",
                    "sensitivity_class_1",
                    "specificity_class_0",
                    "auroc",
                )
            },
            "note": "Validated Protocol B evidence from locked report; not this query's accuracy.",
        }

    def _reliability_base(self, *, input_ok: bool) -> dict[str, Any]:
        return {
            "input_ok": input_ok,
            "encoder_id": "GraphCL GINEncoder 5×128 sum-pool",
            "encoder_sha256": self.status["graphcl_encoder"].checkpoint_sha256,
            "no_untrained_substitutes": True,
            "validated_vs_exploratory": {
                "descriptors": "calculated (RDKit), not learned",
                "bace_protocol_b_locked_metrics": "validated evaluation evidence (split-level)",
                "bace_single_query": "inference only — not a correctness claim",
                "esol_locked_test": "validated evaluation evidence (split-level)",
                "tox21_locked_test": "validated evaluation evidence (split-level)",
                "gnnexplainer": "exploratory unless fidelity/stability validated for the case",
            },
        }
