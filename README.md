---
title: Amoebanator
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Calibrated, abstention-aware PAM triage (synthetic data)
license: mit
---

# Amoebanator

Research codebase for a calibrated, abstention-aware triage signal for Primary Amebic
Meningoencephalitis (PAM), the rare and near-uniformly fatal CNS infection caused by
*Naegleria fowleri*. The classifier is small and honest by design, wrapped in
reviewer-grade safety machinery: temperature-scaled calibration, split conformal
prediction with explicit abstention, energy-based out-of-distribution detection, and
decision curve analysis at clinically realistic prevalences. A literature-anchored
registry of meningoencephalitis vignettes provides the differential-diagnosis context.

> **For research and educational use.** Not a cleared medical device, not a substitute
> for clinical judgment, and not validated for unsupervised use.

## Scope and status

Amoebanator is a clinical-ML infrastructure project for CNS-infection triage, using
primary amebic meningoencephalitis (Naegleria fowleri) as a high-risk must-not-miss
example. This release demonstrates the engineering stack: conformal prediction,
calibrated selective abstention, OOD detection, and reproducible training and
evaluation. The classifier is trained and evaluated on a small synthetic cohort and is
an infrastructure demonstration, not a clinically validated diagnostic tool. Validation
on real clinical data is ongoing work and is not part of this release. Not for clinical
use.

## License and disclaimer

The code and documentation in this repository are released under the MIT
License (see `LICENSE`). The software is provided for research and educational
purposes. It is not a cleared medical device, not a substitute for clinical
judgment, and has not been validated for clinical or unsupervised use; nothing
here should be used to make clinical decisions.

*Repository under active construction.*
