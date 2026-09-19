"""
QUARANTINED.

This entry point previously ran MSE-to-random-noise "pretraining" on BACE.
That is not GraphCL and must not be used for paper results.

Canonical pipeline:
    python main.py --help
    python main.py --smoke
"""

raise SystemExit(
    "experiment.py is quarantined.\n"
    "It used MSE(projection, randn) on BACE, not GraphCL.\n"
    "Historical 0.5180 accuracy is diagnostic only.\n"
    "Use: python main.py --smoke\n"
    "Archived script: archive/experiment_mse_noise_pretrain.py"
)
