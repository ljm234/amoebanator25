# Model card, Amoebanator MLP triage classifier

Per Mitchell et al., *Model Cards for Model Reporting*, FAccT 2019
(DOI 10.1145/3287560.3287596). All quantitative figures below come from
JSON files under `outputs/metrics/` produced by the bundled training
pipeline; every claim is defensible against a current run.

---

## 1. Model details

* **Developer.** Luis Jordan Montenegro Calla, independent researcher
  affiliated with Weber State University and Ensign College. Single-author
  research codebase.
* **Model date.** Trained April 2026; this card reflects model artefacts
  produced on 2026-04-25.
* **Model version.** V1.0. The state_dict at `outputs/model/model.pt` is
  regenerable via `python -m ml.training_calib_dca`.
* **Model type.** Two-class multilayer perceptron (PyTorch); architecture
  `Linear(d, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 2)`.
  Trained with cross-entropy + class weighting; calibrated with L-BFGS
  temperature scaling (Guo et al. 2017).
* **Training algorithms / parameters.** Adam (lr=1e-3), 60 epochs,
  full-batch on the bundled 30-row dataset. Class weight clamped to
  `[1, 10]` so tiny datasets do not produce explosive losses. Random
  seed `42`.
* **Features.** Ten tabular features after one-hot expansion of symptoms:
  `age`, `csf_glucose`, `csf_protein`, `csf_wbc`, `pcr`, `microscopy`,
  `exposure`, `sym_fever`, `sym_headache`, `sym_nuchal_rigidity`. Schema
  pinned in `outputs/model/features.json`.
* **Citation / paper / resource.** This repository
  (github.com/ljm234/amoebanator25); methods documentation in
  `docs/rare_class_design.md`.
* **License.** Released under the MIT License (see `LICENSE`). The model and
  code are provided for research and education and are not intended or
  validated for clinical or commercial use. See the "License and disclaimer"
  section of `README.md`.
* **Where to send questions.** Single-author project; contact the developer
  through the repository.

## 2. Intended use

* **Primary intended use.** Methodology research on calibrated,
  abstention-aware triage models for low-prevalence neurological
  infections. The classifier exists to exercise the surrounding safety
  stack (temperature scaling, split-conformal abstain, Mahalanobis + energy
  OOD gates, decision curve analysis) on a clinically relevant target. The
  model output is a calibrated probability of "high-risk" tier given a
  sparse set of presenting features.
* **Primary intended users.** Methods researchers and PhD-program reviewers
  evaluating the calibration / OOD / DCA pipeline. Educators teaching
  rare-disease ML. The Streamlit live-patient widget exists to make the
  end-to-end pipeline inspectable, not to support any clinical decision.
* **Out-of-scope use cases.**
  - Any real clinical triage decision for an individual patient.
  - Any inferred PAM diagnosis or rule-out.
  - Any deployment in a production EHR or clinical decision-support system.
  - Any substitute for institutional infectious-disease or critical-care
    consultation.

## 3. Factors

* **Relevant factors.** Patient age (the bundled dataset has a median age
  matching Yoder et al. 2010 PAM cohort: 12 years, range 8 months to 66
  years). Sex (the published US PAM cohort is 79.3 % male). Geographic
  exposure source (lakes/ponds/reservoirs dominate at 73.6 %). The
  classifier sees age and a symptom checklist, but no race/ethnicity,
  socio-economic, or geographic metadata are encoded in the bundled
  features.
* **Evaluation factors.** The current evaluation is *unstratified* by
  demographics because the bundled n = 6 validation split cannot support
  meaningful subgroup analysis. Planned future work (a real test set via a
  MIMIC-IV bacterial-vs-viral meningitis cohort) will introduce age-band
  and sex-stratified evaluation.

## 4. Metrics

* **Performance measures and rationale.**
  - **AUC of the calibrated probability of the High class.** Standard
    discrimination metric for ranked-list triage applications. Source of
    truth: `outputs/metrics/metrics.json`, currently `auc_calibrated: 1.0`
    on the n = 6 validation split.
  - **Recall (sensitivity) at the operating threshold.** Captures
    miss-rate, the clinically expensive failure mode. Currently
    `recall_high@0.5: 1.0` on the same n = 6 split.
  - **Conformal coverage and abstain rate at alpha = 0.10.** Currently
    `qhat = 0.0162` from `outputs/metrics/conformal.json` (n = 6, with
    `SmallCalibrationWarning` emitted).
  - **Decision-curve net benefit at the chosen threshold.** Currently
    `net_benefit: 0.298` at `threshold: 0.05` from
    `outputs/metrics/threshold_pick.json`. Net benefit equals the
    treat-all baseline at this threshold, which is the expected behaviour
    when prevalence equals the threshold.
  - **OOD gate thresholds.** Logit-energy `tau = -0.99`
    (`outputs/metrics/energy_threshold.json`), neg-energy on probability
    `tau ~= -1e-8` (`outputs/metrics/ood_energy.json`). Both fitted on
    n = 6 via `scripts/ood/fit_gates.py`.
* **Decision thresholds.** Operating threshold `0.05` chosen by argmax
  net-benefit sweep within `[0.05, 0.30]`. Documented at
  `outputs/metrics/threshold_pick.json`.
* **Uncertainty estimation.** Bootstrap percentile 95 % CI with
  `n_resamples = 2000` available via `ml.metrics.bootstrap.bootstrap_ci`.
  Conformal prediction sets emit ABSTAIN when the band contains both
  classes.

## 5. Evaluation data

* **Datasets used.** A single 6-row stratified validation split from the
  bundled `outputs/diagnosis_log_pro.csv` (30 simulated rows, 80/20
  train/val split at `random_state = 42`). The same split feeds calibration
  and gate fitting today, which is the n = 6 caveat documented in
  Limitations and called out by `SmallCalibrationWarning` at every fit.
* **Motivation.** The project's V1.0 goal was to ship a defensible
  *infrastructure* (calibration, conformal, OOD, DCA) end-to-end, not a
  clinically valid model. Planned work swaps the evaluation set for a
  MIMIC-IV bacterial-vs-viral meningitis cohort once PhysioNet credentialed
  access clears.
* **Preprocessing.** `pd.read_csv` -> Safe Harbor scrub via
  `ml/data_loader.deidentify_dataframe` (caps ages at 89, blanks
  `physician`, generalises dates to year, scrubs free text > 20 chars) ->
  one-hot expansion of `symptoms` -> `df.fillna(0)` -> split.

## 6. Training data

* **Source.** `outputs/diagnosis_log_pro.csv`, 30 simulated rows generated
  to mimic published PAM presentations. Every row carries
  `source = "simulated"` and `physician = "demo"`. No real patient data is
  included.
* **Distribution over factors.** The bundled marginals were tuned to match
  Yoder JS et al. 2010 (median age 12, ~ 79 % male, lake / freshwater
  exposure dominant). Per-row provenance is preserved in the audit log
  (`AMOEBANATOR_AUDIT_PATH`).
* **Other relevant details.** Class weight clamped to `[1, 10]` because
  the small training fold has unstable positive counts. Random seed `42`
  is pinned in `ml/training.py` so the n = 24 train fold is bit-identical
  across runs.

## 7. Quantitative analyses

* **Unitary results (overall).** Calibrated AUC = 1.0; recall (High) at
  0.5 = 1.0; calibration temperature `T ~= 0.27`. Both numbers reflect
  perfect separation on a 6-row validation set and **are not evidence of
  clinical accuracy.**
* **Intersectional results.** Not reported. The validation split has
  insufficient counts to support stratified analysis; formal subgroup
  reporting is planned future work.
* **Ablation table.** `outputs/metrics/ablation_table.{json,csv}` contains
  the four-cell ablation (base / +calibration / +conformal / +OOD) for the
  Amoebanator MLP plus three calibrated baselines (logistic + Platt,
  RF + isotonic, GBM + isotonic). Bootstrap 95 % CIs on every row.
* **Coverage sweep.** `outputs/metrics/coverage_sweep.{json,png}` reports
  empirical coverage at alpha in {0.05, 0.10, 0.20} on a half-half split of
  the validation set (n = 3 calibration / n = 3 test today; the conformal
  framework refuses to write population-level qhats fit on n < 100 unless
  `--force-small`).

## 8. Ethical considerations

* **Sensitive data.** None in the shipped dataset (synthetic, no
  identifiers). Real-data extension via MIMIC-IV is gated by IRB exemption
  and PhysioNet DUA.
* **Model effects on human life / rights / safety.** *Potential* effects in
  the deployment scenario the model targets are catastrophic; PAM is
  near-uniformly fatal. The model card's *Out-of-scope use cases*
  (Section 2) exists to prevent this from being read as a deployable triage
  signal. Misuse risk is concentrated on the demo dashboard: a clinician
  seeing `prediction: "High"` with `p_high = 1.0` could over-anchor without
  reading the surrounding caveats.
* **Mitigation strategies.** Three production gates fire on every inference
  call: Mahalanobis OOD (returns ABSTAIN/OOD when feature vectors leave the
  training cloud), logit-energy gate (returns
  ABSTAIN/LogitEnergyAboveOODShift when logit energy is above the validation
  95th percentile), conformal abstain (returns ABSTAIN/ConformalAmbiguity
  when the prediction set contains both classes). Every model output carries
  the abstain reason field. The Streamlit widget displays the research-only
  disclaimer above the form and the raw safety-signal breakdown below the
  prediction.
* **Risks and harms (residual).** A reviewer or trainee could quote the
  AUC = 1.0 figure without the small-sample caveat. The model card and
  README document this risk explicitly. The `SmallCalibrationWarning` fires
  at every conformal fit until n >= 100.

## 9. Caveats and recommendations

* **n = 6 validation set.** The single most load-bearing limitation. Every
  headline metric (AUC, recall, conformal qhat, energy thresholds) is
  computed on six rows. Until a real held-out set lands (PhysioNet ->
  MIMIC-IV bacterial-vs-viral meningitis cohort) these numbers are
  *infrastructure proofs*, not population estimates.
* **Synthetic training data.** All 30 rows carry `source = "simulated"`. The
  model has never seen a real patient. The synthesis function
  (`ml/case_series.synthesize_yoder_cohort`) draws from Yoder 2010
  marginals; rows it produces carry `source = "synthetic_from_yoder2010"`
  so they can be filtered out of any real-data quoted metric.
* **Conformal coverage guarantee.** Holds at 1 - alpha only as n_cal grows.
  At n = 6 the slack is +/- 1/(n+1) ~= +/- 0.143 around the target; the
  `SmallCalibrationWarning` makes this explicit and the held-out framework
  refuses to ship a population-level claim until n >= 100.
* **No bacterial / viral / fungal class labels yet.** Planned real OOD
  evaluation (B45.x fungal vs B60.2 PAM via MIMIC-IV) remains blocked on
  PhysioNet credentialing. The bundled synthetic OOD benchmark honestly
  reports AUC ~= 0.5 on label-shift (correctly, because flipping labels does
  not change the feature distribution).
* **Cope 2016 CSF numerics not encoded.** The Cope JR & Ali IK 2016 review
  reports CSF abnormalities qualitatively only. For numeric per-case PAM CSF
  values, Capewell LG et al. 2015 (PMID 26582886) is the better source;
  integrating it into `ml/case_series.py` is documented as optional future
  work.
* **No neonatal PAM in the corpus.** The 60-vignette PAM corpus does not
  include extreme-age (neonatal) PAM presentations, because neonatal PAM is
  a distinct epidemiologic category that does not fit the
  cluster structure (lake_pond, river, splash_pad, nasal_irrigation,
  hot_springs, pakistan_ablution). Reserved for a future extension. **Model
  performance on neonatal PAM is undefined.** See `docs/rare_class_design.md`
  for the cluster-structure rationale.
* **Recommendations.**
  - Do not quote AUC, recall, conformal coverage, or DCA net benefit in any
    document without the n = 6 caveat *in the same paragraph*.
  - Do not deploy the Streamlit widget on any clinical-facing surface.
  - When the model is re-fit via `python -m ml.training_calib_dca`, update
    the Section 4 metrics in this card to match the regenerated
    `outputs/metrics/` JSON.

## References

See `docs/references.bib` for full BibTeX entries. Key citations: Mitchell M
et al. FAccT 2019 (model cards spec); Guo C et al. ICML 2017 (temperature
scaling); Vovk V Mach Learn 2013 (label-conditional conformal); Lei J et al.
JASA 2018 (split conformal coverage bound); Liu W et al. NeurIPS 2020 (energy
OOD); Lee K et al. NeurIPS 2018 (Mahalanobis OOD); Vickers AJ & Elkin EB Med
Decis Making 2006 (decision curve analysis); Yoder JS et al. Epidemiol Infect
2010 (US PAM epidemiology); Cope JR & Ali IK Curr Infect Dis Rep 2016 (PAM
clinical review); CDC 2025 *About Primary Amebic Meningoencephalitis*.
