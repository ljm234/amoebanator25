# Data card, Amoebanator bundled dataset

Per Gebru T et al., *Datasheets for Datasets*, Communications of the ACM
2021;64(12):86-92 (DOI 10.1145/3458723; arXiv:1803.09010).

This card documents `outputs/diagnosis_log_pro.csv`, the only dataset that
ships with the V1.0 release. It is **30 simulated patient vignettes**, not
real patient data. The card also documents the *planned* MIMIC-IV cohort
(V1.1 roadmap) so that the data lineage of any future preprint figure is
traceable.

---

## 1. Motivation

* **For what purpose was the dataset created?** To demonstrate the
  Amoebanator calibration / conformal / OOD / DCA pipeline end-to-end on a
  tractable synthetic problem. The dataset exists to prove that the
  *infrastructure* runs, not to produce population-level performance
  estimates. Replacing it with a real clinical cohort is the explicit V1.1
  goal.
* **Who created the dataset?** Luis Jordan Montenegro-Calla (single-author
  research). No institutional dataset commission.
* **Funding.** Unfunded.
* **Other comments.** The synthetic vignettes are clinically *plausible*:
  feature distributions mimic published PAM presentations (Yoder JS et al.,
  *Epidemiol Infect* 2010;138:968-975; Cope JR & Ali IK, *Curr Infect Dis
  Rep* 2016;18:31). They are *not* drawn from any patient population.

## 2. Composition

* **What do the instances represent?** Each row is a hypothetical patient
  presentation with: demographic features (`age`, `sex`), CSF lab values
  (`csf_glucose`, `csf_protein`, `csf_wbc`), three binary clinical findings
  (`pcr`, `microscopy`, `exposure`), a semicolon-separated symptom string
  (`symptoms`), an integer `risk_score`, and an outcome label
  (`risk_label` in {"Low", "High"}). Provenance metadata (`case_id`,
  `source`, `physician`, `timestamp_tz`, `comments`) is also included for
  audit-chain attribution.
* **How many instances?** **30 rows.** Stratified 80/20 train/val split
  yields **n_train = 24**, **n_val = 6**.
* **Is the dataset a sample of a larger set?** No, every row was generated
  synthetically. The dataset is not a sample of a clinical population.
* **What does each instance consist of?** Raw tabular features (16
  columns). Labels (`risk_label`, `risk_score`). Provenance fields
  (`case_id`, `source`, `physician`, `timestamp_tz`, `comments`).
* **Is there a label/target?** Yes, binary `risk_label` in {"Low", "High"}
  (encoded as `y = 1` for High in the trainer). 11 of 30 rows are High in
  the bundled CSV.
* **Is any information missing?** Every row is complete (no missing cells in
  the bundled CSV).
* **Are relationships between instances explicit?** No. Each row is
  independent.
* **Recommended splits.** Stratified 80/20 train/val at `random_state = 42`
  per `ml.training_calib_dca`. The `ml.splits.stratified_split` helper
  supports a 60/20/20 train/val/test split for downstream evaluation.
* **Errors / noise / redundancies.** No deliberate noise injection. Cases
  are drawn to span the PAM-typical feature region; some rows represent
  benign-meningitis controls.
* **Self-contained?** Self-contained. No external dependencies.
* **Confidential data?** None. Synthetic.
* **Offensive / sensitive content?** None.
* **Data relate to people?** Hypothetical people only. No real individuals.
* **Identifies subpopulations?** Sex distribution is intentional: in the
  bundled rows, the male/female balance is biased toward male, consistent
  with Yoder 2010's 79.3 % male PAM cohort. Other demographics (race,
  ethnicity, geography) are not encoded.
* **Possible to identify individuals?** No. There are no real individuals
  in the dataset.
* **Sensitive attributes?** None.

## 3. Collection process

* **How was the data acquired?** Synthetic generation. The 30 rows were
  hand-curated to span the clinical feature region of published PAM
  case-series.
* **Mechanisms / procedures.** A synthetic cohort, manually authored (not software-generated).
  The `ml.case_series.synthesize_yoder_cohort` function provides a
  programmatic synthesis path that draws from Yoder 2010 marginals; rows it
  produces carry `source = "synthetic_from_yoder2010"` and are not part of
  the bundled 30-row CSV today.
* **Sampling strategy.** Not applicable (no underlying population).
* **Who was involved?** Single author. No crowdworkers, contractors, or
  annotators.
* **Timeframe.** 2025-2026 (the `timestamp_tz` field carries plausible but
  synthetic dates within this window).
* **Ethical review.** Not required; no human subjects. Real-data extension
  via MIMIC-IV uses de-identified records under the signed PhysioNet DUA;
  secondary analysis is IRB-exempt.
* **Data relate to people?** No real people.
* **Notification / consent / impact analysis.** Not applicable.

## 4. Preprocessing / cleaning / labeling

* **Preprocessing applied?** Three transformations are applied at load time
  (`ml/data_loader.load_tabular_safe_harbor`):
  1. **Safe Harbor de-identification** (HIPAA 45 CFR 164.514(b)(2)): ages
     > 89 capped to 89, `physician` field blanked, dates generalised to
     year, free-text > 20 chars passed through the `SafeHarborProcessor`
     scrubber.
  2. **One-hot symptom expansion**: `symptoms` string -> `sym_<token>`
     binary indicators.
  3. **Vectorisation**: `feats = ["age", "csf_glucose", "csf_protein",
     "csf_wbc", "pcr", "microscopy", "exposure", "sym_*"]` ->
     `df[feats].fillna(0).astype(float).values`.
  Bundled rows do not contain any age > 89, so the cap is a no-op today; it
  is load-bearing for any future MIMIC-IV-shaped CSV.
* **Raw data preserved?** Yes, `outputs/diagnosis_log_pro.csv` is the raw
  form. Preprocessed `(X, y, feats)` is computed in-memory and not
  persisted as a separate artefact.
* **Preprocessing software.** All in `ml/training.py`,
  `ml/training_calib_dca.py`, and `ml/data_loader.py`. Open-sourced as part
  of this repository.

## 5. Uses

* **Used for any tasks already?** Yes, the bundled MLP in
  `outputs/model/model.pt` was trained on this dataset. Calibration,
  conformal qhat, energy thresholds, DCA threshold, ablation table, and
  coverage sweep figures all derive from it. Every figure under
  `outputs/metrics/` is downstream of the bundled CSV.
* **Repository linking to papers / systems using the dataset.** This
  repository is the only known consumer. The forthcoming preprint will cite
  this data card.
* **What other tasks could the dataset be used for?** Synthetic-data
  benchmarking of small-sample calibration and conformal prediction
  techniques. Pedagogical examples of decision curve analysis at low
  prevalence.
* **Composition / collection issues that impact future use?** The most
  important issue: **n = 30 is too small to fit anything reliably.** Any
  quoted metric must be paired with the n caveat. The dataset is not
  intended as a benchmark for model performance; it is a load-bearing
  fixture for the surrounding safety machinery.
* **Tasks for which the dataset should not be used.**
  - Quoting AUC / recall / sensitivity / specificity as if they were
    population estimates.
  - Training a model intended for any clinical use.
  - Benchmarking against published meningitis-triage classifiers.

## 6. Distribution

* **Distributed to third parties?** Yes, bundled with the open-source
  Amoebanator code release.
* **How will it be distributed?** Same channel as the code (Git
  repository).
* **When?** Now, included in the V1.0 release.
* **License.** Released under the same MIT License as the code (see
  `LICENSE`). The dataset is provided for research and education and is not
  intended or validated for clinical or commercial use (see `README.md`,
  License and disclaimer section).
* **Third-party IP restrictions.** None; all rows are synthetic.
* **Export / regulatory restrictions.** None applicable to synthetic data.

## 7. Maintenance

* **Who maintains the dataset?** Luis Jordan Montenegro-Calla.
* **How to contact the maintainer.** Contact the maintainer through the
  repository.
* **Errata?** None at the V1.0 release. Errata will be tracked in the
  repository's release notes.
* **Will the dataset be updated?** Yes, the V1.1 milestone replaces the
  bundled 30-row synthetic CSV with a MIMIC-IV-derived bacterial-vs-viral
  meningitis cohort, now that PhysioNet credentialed access is in place. The bundled
  synthetic CSV will remain in the repository as a fixture for the test
  suite, but headline metrics will switch to the real-data cohort.
* **Retention limits?** Not applicable (synthetic).
* **Older versions supported?** Yes, a `git tag` will mark the V1.0 release
  commit; the V1.0 CSV remains accessible through the repository history.
* **Mechanism for contributions.** Pull requests via the project
  repository. Adding new synthetic rows requires (a) explicit
  `source = "synthetic_*"` provenance, (b) re-running the audit chain to
  record the addition, (c) re-fitting all downstream metrics so the model
  card stays synchronised.

---

## Planned dataset (V1.1, de-identified MIMIC-IV)

With PhysioNet credentialed access in place, the V1.1 dataset
will be a MIMIC-IV cohort with the schema below. Documenting it here so the
lineage of any future preprint figure is traceable from this card.

| Field | Source | Notes |
|-------|--------|-------|
| `subject_id` | `hosp.patients.subject_id` | Surrogate ID; never a real MRN |
| `csf_glucose` | `hosp.labevents` itemid 51790 | mg/dL; median per subject |
| `csf_protein` | `hosp.labevents` itemid 51802 | mg/dL; median per subject |
| `csf_wbc` | `hosp.labevents` itemid 52286 | "Total Nucleated Cells, CSF", closest available; MIMIC-IV does not carry "WBC, CSF" |
| `csf_polys_pct` | `hosp.labevents` itemid 52281 | neutrophil % proxy |
| `microscopy` | `hosp.microbiologyevents` | 1 if any positive Gram stain on `spec_type_desc == 'CSF;SPINAL FLUID'` |
| `risk_label` | `hosp.diagnoses_icd` | High = G00.x bacterial; Low = A87.x viral; OOD held-out = B60.2 PAM |
| `pcr`, `exposure` | not in MIMIC-IV | constant 0 in V1.1 (PAM-specific) |
| `age` | `hosp.patients` | computed at admission |

Loader: `ml/mimic_iv_loader.assemble_cohort`. Smoke-tested end-to-end
against synthetic MIMIC-shaped CSVs in `tests/test_mimic_iv_loader.py`.

---

## Honesty signal

This data card devotes roughly a third of its length to limitations,
intended-not-uses, and the "do not quote AUC without the n caveat" warning.
That ratio is intentional: at n = 30 the only defensible scientific
contribution is honest disclosure of what the dataset is and is not.
