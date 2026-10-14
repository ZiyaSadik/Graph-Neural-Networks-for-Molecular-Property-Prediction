#!/usr/bin/env python3
"""
Chemist-facing Streamlit app for molecular property analysis.

Uses existing GraphCL/GIN checkpoints and mol_cli inference adapters.
Does not invent results or substitute untrained models.

Run from repo root:
  streamlit run apps/chemist_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from apps.chemist_app.service import ChemistAnalysisService


st.set_page_config(
    page_title="Explainable Few-Shot GNN — Molecular Analysis",
    page_icon=None,
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models (first run may take a minute)…")
def get_service() -> ChemistAnalysisService:
    return ChemistAnalysisService()


def main() -> None:
    st.title("Explainable Few-Shot GNN for Molecular Property Prediction")
    st.caption(
        "Research inference UI · GraphCL–GIN encoder · Protocol B BACE prototypes · "
        "ESOL / Tox21 heads · Exploratory GNNExplainer"
    )

    with st.expander("Important limitations (read first)", expanded=True):
        st.markdown(
            """
- **Not clinical advice.** Outputs are research model estimates.
- BACE uses **Euclidean prototype distances**, not calibrated probabilities.
- Tox21 **sigmoid scores** are not established calibrated probabilities; do not label a molecule
  as categorically safe or toxic from these scores alone.
- ESOL is a **model estimate** of measured log solubility, not a lab measurement.
- GNNExplainer visualizations are **exploratory** unless marked otherwise; they do not establish
  chemical causality.
- BACE Protocol B (app) is **not interchangeable** with the paper's episodic ProtoNet protocol.
            """
        )

    svc = get_service()
    cols = st.columns(4)
    for i, (name, status) in enumerate(svc.status.items()):
        with cols[i % 4]:
            mark = "OK" if status.available else "MISSING"
            st.metric(name, mark)

    smiles = st.text_input(
        "SMILES",
        value="CCO",
        help="Enter a valid molecular SMILES string",
    )
    run_xai = st.checkbox(
        "Run exploratory GNNExplainer (slower)",
        value=True,
        help="Uses PrototypeDistanceLogits; may fail — failures are reported, never invented.",
    )
    c1, c2 = st.columns(2)
    with c1:
        epochs = st.number_input("Explainer epochs", min_value=10, max_value=120, value=40, step=10)
    with c2:
        seed = st.number_input("Explainer seed", min_value=0, max_value=9999, value=0, step=1)

    if st.button("Analyze molecule", type="primary"):
        with st.spinner("Running inference…"):
            report = svc.analyze(
                smiles,
                run_explanation=run_xai,
                explain_epochs=int(epochs),
                explain_seed=int(seed),
            )

        if not report.valid:
            st.error(report.error or "Invalid SMILES")
            return

        # A. Overview
        st.header("A. Molecule overview")
        left, right = st.columns([1, 1])
        with left:
            st.write(f"**Input:** `{report.input_smiles}`")
            st.write(f"**Canonical:** `{report.canonical_smiles}`")
            st.write(f"**Atoms / bonds:** {report.num_atoms} / {report.num_bonds}")
            if report.structure_png:
                st.image(report.structure_png, caption="2D structure with atom indices")
            else:
                st.warning("Structure drawing unavailable.")
        with right:
            st.subheader("Calculated descriptors (RDKit)")
            st.caption("These are **calculated**, not learned model outputs.")
            if report.descriptors:
                st.table(
                    {
                        "descriptor": list(report.descriptors.keys()),
                        "value": list(report.descriptors.values()),
                    }
                )

        # B. BACE
        st.header("B. Few-shot BACE classification (Protocol B)")
        bace = report.bace or {}
        if not bace.get("available"):
            st.error(bace.get("error", "BACE unavailable"))
        else:
            st.info(bace["protocol"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted class", f"{bace['predicted_class']} ({bace['predicted_class_name']})")
            m2.metric("d → inactive (0)", f"{bace['distance_to_class_0_inactive']:.3f}")
            m3.metric("d → active (1)", f"{bace['distance_to_class_1_active']:.3f}")
            m4.metric("|d0 − d1| gap", f"{bace['distance_gap']:.3f}")
            st.caption(
                f"Support: K={bace['k_shot_per_class']}/class, seed={bace['support_seed']}, "
                f"n_support={bace['n_support_total']}, manifest=`{bace['support_manifest_id']}`. "
                "**Distances are not probabilities.**"
            )
            st.markdown("**Limitations**")
            for line in bace.get("limitations", []):
                st.write(f"- {line}")
            ev = bace.get("locked_eval_evidence") or {}
            if ev.get("available"):
                with st.expander("Locked Protocol B evaluation evidence (split-level)"):
                    st.write("Validation metrics:", ev.get("validation_metrics"))
                    st.write("Test metrics:", ev.get("test_metrics"))
                    st.caption(ev.get("note", ""))

        # C. ESOL
        st.header("C. ESOL aqueous solubility")
        esol = report.esol or {}
        if not esol.get("available"):
            st.error(esol.get("error", "ESOL unavailable"))
        else:
            st.metric("Predicted logS", f"{esol['predicted_logS']:.4f}")
            st.write(f"**Target scale:** {esol['target_scale']}")
            st.write(esol["note"])
            st.caption(esol.get("locked_test_reference", ""))

        # D. Tox21
        st.header("D. Tox21 (12 tasks)")
        tox = report.tox21 or {}
        if not tox.get("available"):
            st.error(tox.get("error", "Tox21 unavailable"))
        else:
            for w in tox.get("warnings", []):
                st.warning(w)
            rows = tox.get("tasks") or []
            st.dataframe(
                {
                    "task": [r["task"] for r in rows],
                    "logit": [r["logit"] for r in rows],
                    "sigmoid_score": [r["score"] for r in rows],
                },
                use_container_width=True,
            )
            st.caption(tox.get("locked_test_reference", ""))

        # E. Explainability
        st.header("E. Explainability")
        expl = report.explanation
        if expl is None:
            st.write("Explanation not requested.")
        elif not expl.get("succeeded"):
            st.error(
                f"Explanation failed or not run ({expl.get('validated_status')}): "
                f"{expl.get('error')}"
            )
            st.caption("No explanation was invented.")
            for c in expl.get("caveats") or []:
                st.write(f"- {c}")
        else:
            st.warning(
                f"Status: **{expl.get('validated_status')}** — "
                "not a validated causal chemical mechanism."
            )
            st.write(
                f"Method: `{expl.get('method')}` · epochs={expl.get('epochs')} · "
                f"seed={expl.get('seed')} · index_mapping_ok={expl.get('index_mapping_ok')}"
            )
            if report.explanation_png:
                st.image(
                    report.explanation_png,
                    caption="Atom importance (redder = higher mask). Indices match OGB nodes.",
                )
            for c in expl.get("caveats") or []:
                st.write(f"- {c}")

        # F. Reliability
        st.header("F. Reliability panel")
        rel = report.reliability
        st.json(rel)


if __name__ == "__main__":
    main()
