# Reproducibility, Amoebanator

This document describes how to reproduce the V1.0 results from a clean
checkout, and states plainly what is and is not reproducible. The model and
data cards (`docs/model_card.md`, `docs/data_card.md`) carry the artefact and
dataset detail; this document is the operational recipe and its honest
limits.

---

## 1. Scope

A clean checkout reproduces every V1.0 synthetic-data result: the trained
model, its calibration, and the conformal, out-of-distribution, and
decision-curve artefacts, along with the full test suite. The planned
MIMIC-IV proxy study is not reproducible from this repository, because that
data is not redistributable and must be obtained from PhysioNet under
credentialed access; its pre-specified protocol is in
`docs/rare_class_design.md`.

## 2. Environment

* **Interpreter.** Python 3.12. The V1.0 artefacts were produced under Python
  3.12.11 on macOS 14 (Apple Silicon); continuous integration runs Python
  3.12 on Linux.
* **Dependencies.** All versions are pinned in `requirements.txt`: torch
  2.9.1, scikit-learn 1.8.0, numpy 2.2.6, pandas 2.3.3, scipy 1.16.3,
  matplotlib 3.10.6, and streamlit 1.52.0, together with the development tools
  pytest 9.0.2, pytest-cov 7.0.0, hypothesis 6.149.1, ruff 0.14.10, and mypy
  1.19.1, plus pydantic and jsonschema for the vignette schema. Install with
  `pip install -r requirements.txt`.
* **Accelerator.** None required. PyTorch runs CPU-only by default; a CPU
  wheel index for Linux CI is noted in `requirements.txt`.

## 3. Determinism and seeds

`ml/seeds.py` is the single source of seed configuration. `set_global_seeds()`
(default seed 42, overridable through the `AMOEBANATOR_SEED` environment
variable) pins Python's `random` module, NumPy, and PyTorch (CPU, plus MPS and
CUDA when present), disables cuDNN benchmark mode, and enables deterministic
algorithms, so that two runs on the same hardware produce a bit-identical
`model.pt`. The train and validation split uses `random_state=42`. As with any
floating-point pipeline, bit-level reproducibility holds on the same hardware
and library build; different hardware may diverge in the last bits.

## 4. Reproducing the model and metrics

* **Model and calibration.** `python -m ml.training_calib_dca` reads the
  synthetic dataset (`outputs/diagnosis_log_pro.csv`), trains the MLP under
  seed 42, fits L-BFGS temperature scaling, and writes the model artefacts
  (`outputs/model/`: `model.pt`, `features.json`, `temperature_scale.json`)
  and the validation predictions used for the calibration and decision-curve
  plots.
* **Metrics and figures.** The artefacts under `outputs/metrics/` (calibration
  curve, coverage sweep, decision-curve, abstain Pareto, the conformal and
  OOD JSON, bootstrap confidence intervals, and the ablation table) are
  produced by the training and evaluation pipeline;
  `outputs/metrics/regeneration_summary.json` records the regeneration. Every
  figure cited in the model card traces to a file in `outputs/metrics/` and is
  reproducible from the bundled synthetic CSV.

## 5. Verifying a reproduction

The reproduction is checked by the test suite. Under Python 3.12 with the
pinned requirements, `pytest` passes (1907 passed, 2 skipped, 1 xfailed),
`ruff check .` reports no issues, and `mypy` reports no issues across the
source tree. ruff runs in continuous integration on every push (Python 3.12);
the full gate of ruff, mypy, and pytest, with every tool version pinned in
`requirements.txt`, is the development standard.

## 6. Data

The 30-row synthetic dataset ships in the repository at
`outputs/diagnosis_log_pro.csv` and is sufficient to reproduce every V1.0
result. The planned MIMIC-IV proxy cohort is not redistributable and requires
PhysioNet credentialing and the MIMIC-IV data use agreement; see
`docs/data_card.md` Section 6 and the protocol in
`docs/rare_class_design.md`.

## 7. Hardware

CPU-only. The model is small enough to train and evaluate in seconds on a
laptop CPU, and no accelerator is required. The V1.0 artefacts were produced
on macOS 14 (Apple Silicon); continuous integration runs on ubuntu-latest.

## Honesty signal

The synthetic-data results are fully and deterministically reproducible, but
reproducing them reproduces an infrastructure proof on a 6-row validation
split, not clinical performance. The headline metrics carry their n caveat in
the model card, and the real-data study that would produce clinically
meaningful numbers is the planned MIMIC-IV proxy study, not yet run. Reproducibility here means the
pipeline is honest and re-runnable, not that the numbers are clinically
validated.

## References

See `docs/references.bib` and the model and data cards for the methods and
dataset detail. Dependency versions are pinned in `requirements.txt`; the
test, lint, and type configuration is in `pyproject.toml` and
`.github/workflows/ci.yml`.
