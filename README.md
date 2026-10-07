# BehaviorTrace

Open evaluation harness for gradient-based training-data attribution in online RL (GRPO) fine-tuning,
accompanying the paper "Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data
Attribution in Online RL."

BehaviorTrace provides full-gradient (CountSketch) sketching, a planted-behavior setup with known
ground truth, the estimators we evaluate (a cross-step cosine estimator which is GAS, a TRAK-style
whitened estimator, TracInCP, a target-free norm-only magnitude control, and a local-buffer
baseline), and controls for gradient magnitude, fluency, headroom, and seed and draw variance,
distilled into an evaluation checklist.

## Install

Requires Python 3 and [uv](https://github.com/astral-sh/uv).

    uv sync
    uv pip install -e .

## Reproduce the paper's tables and figures (no Google Drive needed)

The per-seed score, ground-truth, token-level, and minimal-pair files are included under
`data/results/runs/`. Point the artifacts root at that directory and run the export:

    export BEHAVIORTRACE_ARTIFACTS_DIR=$PWD/data/results
    python notebooks/colab/E2_1_export_results.py     # regenerates data/results/results_table.csv
    python scripts/plots/make_results_figures.py       # regenerates docs/figures/*.png

The export regenerates all 63 rows of `results_table.csv`; every point estimate matches the committed
table within 5e-4 (the confidence-interval columns differ by bootstrap noise). `data/results/README.md`
states exactly what each table and figure is sourced from.

## What reproduces from this repository, and what needs compute

- Tables 1 and 2, and Table 3 seed 2, recompute from the included files.
- Table 3 seeds 0 and 1, and all Table 3 intervals, are in `data/results/rollout_seed01.json`
  (extracted from the saved run notebooks, which predate per-rollout row saving; the executed
  notebooks are available on request).
- Table 4 means come from the token-level files; the figures from the results table and the
  run-notebook intervals.
- Full end-to-end training (the GRPO rebuilds that produce the score files) runs from the notebooks in
  `notebooks/colab/` but needs a GPU; the training checkpoints are not shipped.

Additional experiment logs and intermediate runs are available to researchers on request.

## Layout

- `src/behaviortrace/` the harness: sketching, estimators, metrics, the GRPO backend, the contaminator.
- `notebooks/colab/` the experiment notebooks; `runs/` holds executed, seed-named records.
- `data/results/` the results table and the per-seed score, ground-truth, token-level, and row files.
- `data/probes/hidden_trigger/` the planted-behavior probe set.
- `scripts/`, `tests/`, `configs/`.

## Citation

See `CITATION.cff`.

## License

Apache-2.0. See `LICENSE`.
