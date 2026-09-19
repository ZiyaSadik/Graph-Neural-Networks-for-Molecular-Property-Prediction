# Archive (diagnostic only)

**Do not cite these as paper or GraphCL results.** Canonical outputs: `outputs/core_pipeline/` (few-shot **0.5910 ± 0.1698**). See repository `README.md` and `outputs/HISTORICAL_DIAGNOSTIC_ONLY.txt`.

These files are **not** part of the scientific pipeline. They are retained so the invalid numbers can be traced.

- `experiment_mse_noise_pretrain.py`: trained a separate 3-layer GIN by matching projections to `torch.randn_like(z)` on 300 BACE molecules. The **0.5180 ± 0.1590** figure in `outputs/final_results.txt` comes from this script.
- `legacy_explainability_script.py`: auto-running 3-layer explainer that loaded `outputs/final_model.pt` and scored one 10-query episode (**9/10**). Not a paper result.

Do not load `outputs/final_model.pt` into `GINEncoder`. Architectures are incompatible.
