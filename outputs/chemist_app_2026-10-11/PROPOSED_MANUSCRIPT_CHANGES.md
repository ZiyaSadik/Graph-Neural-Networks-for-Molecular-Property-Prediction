# Proposed manuscript changes (not applied)

Verified results only. Do **not** treat this file as an automatic paper rewrite.

1. **Methods — application note (optional):** Mention that Protocol B inference is exposed via a local chemist UI that reuses the same GraphCL checkpoint and support manifest as the locked CLI evaluation.

2. **Results — Protocol B multi-K validation table:** Cite `outputs/phase_strengthen_2026-10-11/phase3_fewshot_bace/` (or the slim copy `outputs/chemist_app_2026-10-11/fewshot_comparison_summary.json`). Keep Protocol A episodic accuracy in a separate table.

3. **Explainability:** State that GNNExplainer on `PrototypeDistanceLogits` is exploratory; atom-index mapping was checked; do not claim causality or validated fidelity for routine use.

4. **Calibration:** Explicitly state that BACE distances and Tox21 sigmoid scores are **not** calibrated probabilities (see Phase 5 assessment).

5. **GraphCL vs No-GraphCL:** Keep Protocol A ablation as recorded. Do **not** claim a Protocol B No-GraphCL comparison until a compatible checkpoint exists.
