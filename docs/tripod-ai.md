# TRIPOD+AI reporting checklist, Amoebanator MLP triage classifier

Per Collins GS et al., *TRIPOD+AI statement: updated guidance for reporting
clinical prediction models that use regression or machine learning methods*,
BMJ 2024;385:e078378 (DOI 10.1136/bmj-2023-078378).

This document maps the Amoebanator release to the TRIPOD+AI 2024 reporting
checklist for clinical prediction models that use machine learning. It
distinguishes what the current V1.0 release reports, a calibration /
conformal / out-of-distribution / decision-curve infrastructure exercised on
a 30-row synthetic dataset with a 6-row validation split, from the planned
clinical prediction study, a MIMIC-IV bacterial-vs-viral meningitis proxy
with Naegleria fowleri (PAM) held out for out-of-distribution evaluation and
gated on PhysioNet credentialed access. Items that only the planned study can
satisfy are marked **Planned**. The pre-specified analysis protocol for that
study is `docs/rare_class_design.md`; model and dataset specifics are in
`docs/model_card.md` and `docs/data_card.md` and are not duplicated here.
Every quantitative figure cited below carries its sample-size caveat in the
same sentence, per the model card's standing rule.

---

## 1. Title and abstract

* **Title.** The repository and the forthcoming preprint identify the work as
  the development of a multivariable prediction model, a binary triage
  classifier, for a low-prevalence neurological-infection target (PAM)
  operationalised through a bacterial-vs-viral meningitis proxy. Target
  population and predicted outcome are stated in `model_card.md` Section 2.
* **Abstract. Planned.** The V1.0 repository carries no standalone manuscript
  abstract. The forthcoming preprint will include a structured abstract
  compliant with the TRIPOD+AI-for-Abstracts checklist (objectives, data
  sources, outcome, predictors, sample size, model type, performance with
  uncertainty, and limitations).

## 2. Introduction

* **Background and rationale.** The healthcare context is diagnostic triage.
  `rare_class_design.md` Section 1 states the clinical question: a patient
  presents with an acute meningitis-like syndrome and the clinician must
  decide how aggressively to escalate, where the cost of missing PAM is
  catastrophic and the cost of over-triaging benign meningitis is modest. The
  rationale for a machine-learning treatment is that standard modelling on a
  single-digit-positive PAM corpus cannot separate model behaviour from
  sampling noise (`rare_class_design.md` Section 2; CDC 2025; Yoder 2010). No
  published calibrated, abstention-aware PAM triage model exists; the
  contribution is the surrounding safety stack, not a new diagnostic test.
* **Objectives and intended use.** The objective is to develop and internally
  exercise a calibrated, abstention-aware triage classifier together with its
  trustworthy-ML safety stack (temperature scaling, split-conformal abstain,
  Mahalanobis and energy OOD gates, decision curve analysis), and to
  pre-specify the real-data proxy study. V1.0 is development plus an
  infrastructure proof on synthetic data; V1.1 (Planned) is development plus
  internal validation on the real proxy cohort. Intended users, intended use,
  and out-of-scope uses are enumerated in `model_card.md` Section 2; the model
  is explicitly not for any clinical decision.

## 3. Methods

* **Source of data.** V1.0 uses a 30-row synthetic dataset
  (`outputs/diagnosis_log_pro.csv`) hand-curated to span published PAM feature
  distributions (Yoder 2010; Cope 2016); it is not a sample of any patient
  population, and its timestamps are synthetic dates within 2025-2026
  (`data_card.md` Sections 1-3). The data were not used in any prior study and
  no blinding applies because there are no real outcomes. **Planned (V1.1):**
  MIMIC-IV (`hosp.labevents`, `hosp.diagnoses_icd`, `hosp.microbiologyevents`),
  a retrospective, de-identified, single-centre research database, used here
  for the first time for this task (`rare_class_design.md` Section 3).
* **Participants.** V1.0 instances are hypothetical patients; eligibility is
  synthetic presentations spanning the PAM-typical feature region plus
  benign-meningitis controls (`data_card.md` Section 2). No treatments are
  modelled. **Planned:** admissions with at least one CSF analyte and at least
  one ICD-10 meningitis code, with G00.x bacterial as the positive class,
  A87.x viral as the negative class, and B60.2 PAM held out for OOD; splits
  are group-disjoint by `subject_id` so no patient appears in more than one
  partition (`rare_class_design.md` Section 3).
* **Outcome.** V1.0 predicts a binary `risk_label` in {Low, High} (High
  encoded as `y = 1`), assigned at row authoring, with 11 of 30 rows High
  (`data_card.md` Section 2); no outcome-assessment blinding applies to
  synthetic labels. **Planned:** the bacterial (G00.x) versus viral (A87.x)
  ICD-10 label, chosen because it maps onto the clinically actionable boundary
  of whether to start empiric antibacterial therapy now; the outcome is
  derived from coded diagnoses, independent of predictor extraction
  (`rare_class_design.md` Section 3).
* **Predictors.** Ten tabular features after one-hot expansion of symptoms:
  `age`, `csf_glucose`, `csf_protein`, `csf_wbc`, `pcr`, `microscopy`,
  `exposure`, `sym_fever`, `sym_headache`, `sym_nuchal_rigidity`, with the
  schema pinned in `outputs/model/features.json` (`model_card.md` Section 1;
  `data_card.md` Section 4). Predictors are measured at presentation. `pcr`
  and `exposure` are PAM-specific signals absent from MIMIC-IV and are dropped
  or held constant zero in the proxy (`rare_class_design.md` Section 3). No
  predictor-assessment blinding applies.
* **Sample size.** V1.0 is n = 30 (n_train = 24, n_val = 6). There is no
  formal power calculation; the size is a deliberately small fixture to
  exercise the infrastructure and is the single load-bearing limitation
  (`model_card.md` Section 9; `data_card.md` Section 5). The conformal
  framework refuses to write a population-level qhat fit on n < 100 unless
  explicitly forced. **Planned:** the proxy cohort size is reported with the
  real extraction, and the n >= 100 floor together with the Lei 2018
  coverage-slack bound govern when population-level claims are permitted.
* **Missing data.** The bundled V1.0 CSV has no missing cells; the loader
  applies `df.fillna(0)` after one-hot expansion (`data_card.md` Sections 2
  and 4). **Planned:** MIMIC-IV missingness is handled in
  `ml/mimic_iv_loader.assemble_cohort` with per-subject medians for repeated
  labs and constant zero for the PAM-specific columns.
* **Analytical methods.** A two-class multilayer perceptron (PyTorch),
  `Linear(d, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 2)`, trained
  with cross-entropy and class weighting using Adam (lr = 1e-3) for 60 epochs,
  full-batch, with random seed 42 (`model_card.md` Section 1). Three
  calibrated reference baselines are reported for comparison: logistic
  regression with Platt scaling, random forest with isotonic calibration, and
  gradient-boosted trees with isotonic calibration (`model_card.md`
  Section 7). Internal validation is a stratified hold-out, with a
  group-disjoint splitter reserved for the planned proxy. Hyperparameters are
  fixed rather than tuned given the fixture size, and this is documented as
  such.
* **Class imbalance.** Addressed with class weighting, clamped to the interval
  [1, 10] so that the small positive count does not produce explosive losses
  (`model_card.md` Sections 1 and 6).
* **Model output.** The model emits a calibrated probability of the High tier
  via L-BFGS temperature scaling (Guo et al. 2017), a discrete prediction,
  and, when the conformal prediction set contains both classes, an explicit
  ABSTAIN carrying a reason field (`model_card.md` Sections 1 and 4). Output
  is produced at inference time.
* **Training.** The model is trained on the 24-row fold; the calibration
  temperature is fit by L-BFGS, and the conformal qhat and OOD gate thresholds
  are fit on the validation rows (`model_card.md` Sections 1 and 6). A
  small-calibration warning fires at every fit until n >= 100.
* **Evaluation.** Performance measures are the AUC of the calibrated High
  probability, recall at the decision-curve-chosen threshold, conformal
  coverage and ABSTAIN rate at alpha in {0.05, 0.10, 0.20}, decision-curve net
  benefit, and OOD detection AUC for the Mahalanobis, logit-energy, and
  neg-energy gates evaluated on in-distribution versus PAM rows, each with
  bootstrap 95% confidence intervals (n_resamples = 2000) (`model_card.md`
  Section 4; `rare_class_design.md` Section 4). On the current 6-row
  validation split the discrimination figures are at ceiling and are
  infrastructure proofs, not clinical performance.
* **Fairness.** Relevant factors are age, sex, and exposure source
  (`model_card.md` Section 3). V1.0 evaluation is unstratified because the
  6-row split cannot support subgroup analysis, so no fairness metric is
  computed (`model_card.md` Sections 3 and 7). The shipped model encodes no
  race, ethnicity, socio-economic, or geographic attributes. **Planned:**
  age-band and sex-stratified evaluation on the real proxy cohort.

## 4. Open science

* **Funding.** Unfunded single-author research (`data_card.md` Section 1).
  There is no funder and therefore no funder role in design, analysis, or
  reporting.
* **Competing interests.** None declared for this research codebase.
* **Protocol and registration.** Not registered. This is a research-stage
  methods project, not a prospective clinical study. The analysis protocol for
  the planned proxy study is pre-specified in `docs/rare_class_design.md`.
* **Data availability.** The 30-row synthetic dataset ships in the repository
  at `outputs/diagnosis_log_pro.csv`. The planned MIMIC-IV cohort is not
  redistributable and is available only to PhysioNet-credentialed users under
  the MIMIC-IV data use agreement (`data_card.md` Section 6).
* **Code availability.** All code, covering the model, calibration, conformal
  prediction, OOD gates, decision curve analysis, data loaders, and the test
  suite, is in this repository under the license stated in `README.md`.

## 5. Patient and public involvement

There was no patient or public involvement. The V1.0 dataset is synthetic and
the planned cohort is a retrospective, de-identified research database; no
patients or members of the public were involved in the design, conduct, or
reporting of this work.

## 6. Results

* **Participants.** V1.0 comprises 30 synthetic rows, 11 High and 19 Low, with
  an 80/20 stratified split (n_train = 24, n_val = 6); the marginals are tuned
  to Yoder 2010 (median age 12 years, approximately 79% male, freshwater
  exposure dominant) (`data_card.md` Section 2; `model_card.md` Section 3).
  **Planned:** a participant-flow diagram, from admissions screened through the
  CSF and ICD filter to the final cohort by class, will accompany the real
  extraction.
* **Model development.** The final model is the state_dict at
  `outputs/model/model.pt`, regenerable via `python -m ml.training_calib_dca`,
  with the feature schema in `features.json` and a calibration temperature of
  approximately 0.27 on the current fit (`model_card.md` Sections 1 and 7);
  that temperature is fit on 6 rows and is therefore provisional.
* **Model performance.** On the 6-row validation split the calibrated AUC is
  1.0 and recall for the High class at threshold 0.5 is 1.0, figures that
  reflect perfect separation on six rows and are not evidence of clinical
  accuracy (`model_card.md` Sections 4 and 7). The conformal qhat is 0.0162 at
  alpha = 0.10 with a small-calibration warning on n = 6, and the
  decision-curve net benefit is 0.298 at threshold 0.05, equal to the
  treat-all baseline at that prevalence (`model_card.md` Section 4). Every
  figure carries its n caveat and is reproducible from `outputs/metrics/`.
* **Model evaluation and updating.** The held-out conformal framework refuses
  population-level claims until n >= 100. The V1.1 milestone swaps the
  synthetic fixture for the real proxy cohort and re-fits every downstream
  metric, after which this checklist and the model card are updated together
  (`data_card.md` Section 7; `model_card.md` Section 9).

## 7. Discussion

* **Interpretation.** The V1.0 contribution is an honest, reproducible
  trustworthy-ML pipeline for a low-prevalence, high-asymmetric-cost triage
  problem, not a validated PAM classifier. The headline discrimination figures
  are infrastructure proofs on six validation rows (`model_card.md` Sections 7
  and 9; `rare_class_design.md` Section 5).
* **Limitations.** The load-bearing limitations are the 6-row validation set,
  the 30-row synthetic training data, the conformal coverage guarantee holding
  only as the calibration set grows, the absence of real bacterial, viral, and
  fungal labels (blocked on PhysioNet), and undefined performance on neonatal
  PAM (`model_card.md` Section 9; `data_card.md` Sections 5 and 7).
* **Usability and future research.** Out-of-scope uses, namely no clinical
  triage, no PAM diagnosis or rule-out, and no EHR or clinical-decision-support
  deployment, are enumerated in `model_card.md` Section 2. Future work is the
  PhysioNet to MIMIC-IV proxy study with stratified and fairness evaluation,
  per `docs/rare_class_design.md`.

## Other information

This checklist follows TRIPOD+AI (Collins GS et al., BMJ 2024); the original
TRIPOD statement (Collins GS et al., 2015) is superseded for machine-learning
models. The model card, the data card, and the proxy-task design document
together constitute the supplementary reporting. Full BibTeX entries are in
`docs/references.bib`.

## Honesty signal

Like the model and data cards, this checklist is explicit that the current
release satisfies the development and infrastructure half of TRIPOD+AI on
synthetic data, and that the participant, real-outcome, real-performance, and
fairness items are deferred to the planned proxy study. Marking those items
Planned rather than quietly omitting them is the point: the reporting is
complete about what is, and is not, yet done.

## References

See `docs/references.bib` for full BibTeX entries. Key citations: Collins GS
et al. BMJ 2024 (TRIPOD+AI); Collins GS et al. 2015 (original TRIPOD,
superseded for machine learning); Mitchell M et al. FAccT 2019 (model cards);
Gebru T et al. CACM 2021 (datasheets); Guo C et al. ICML 2017 (temperature
scaling); Vovk V Mach Learn 2013 and Lei J et al. JASA 2018 (split conformal);
Liu W et al. NeurIPS 2020 (energy OOD); Lee K et al. NeurIPS 2018 (Mahalanobis
OOD); Vickers AJ and Elkin EB Med Decis Making 2006 (decision curve analysis);
Yoder JS et al. Epidemiol Infect 2010 and Cope JR and Ali IK Curr Infect Dis
Rep 2016 (PAM epidemiology and clinical review).
