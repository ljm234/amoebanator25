# scripts/

Utility and pipeline scripts for the model, organized into functional
subpackages. Run everything from the repository root with `PYTHONPATH=.` set.
Each subdirectory is a Python package, so modules are imported as
`from scripts.<group>.<module> import ...`.

## Subpackages

- **`conformal/`** - Split-conformal prediction: fitting (global, grouped,
  label-conditional, and from precomputed probabilities or a held-out split),
  evaluation, coverage sweeps, and abstention/coverage trade-off analysis with
  plots.
- **`ood/`** - Out-of-distribution detection: fitting energy, entropy, and
  Mahalanobis scorers and their thresholds, OOD gates, refitting on training
  statistics, evaluation from prediction logs, and a synthetic OOD benchmark.
- **`calibration/`** - Calibration and metrics: feature statistics, bootstrap
  confidence intervals, reliability/calibration analysis, and calibration +
  decision-curve plots.
- **`inference/`** - Command-line inference entry points, with and without OOD
  gating.
- **`vignettes/`** - Clinical vignette and test-fixture generation, plus
  validation-data preparation.
- **`experiments/`** - Ablation studies.

## Root orchestrators

These stay at the top level of `scripts/`:

- **`run_full_pipeline.sh`** - Runs the full artifact pipeline end to end
  (`PYTHONPATH=. bash scripts/run_full_pipeline.sh`).
- **`train.sh`** - Trains the model (`python -m ml.training`).
- **`regenerate_all_artifacts.py`** - Regenerates all derived artifacts via
  subprocess orchestration; supports `--dry-run` to check artifact presence
  without executing
  (`PYTHONPATH=. python scripts/regenerate_all_artifacts.py --dry-run`).
