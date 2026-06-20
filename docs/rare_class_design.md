# Rare-class triage as a low-prevalence proxy task

This document defines the supervised task, explains why PAM is operationalised
through a proxy, and pins down the evaluation protocol so the downstream
experiments can be reproduced from it alone.

---

## 1. The clinical question

A patient presents with acute meningitis-like syndrome (fever, headache,
nuchal rigidity, altered mental status). The clinician must decide, in the
first minutes of the workup, how aggressively to escalate. The worst
outcome, Primary Amebic Meningoencephalitis (PAM), is functionally
untreatable once intracranial pressure rises, so the cost of *missing* a PAM
case is catastrophic. The cost of *over-triaging* a benign meningitis is a
few hours of additional workup and one extra dose of broad-spectrum empiric
therapy.

This is a textbook low-prevalence, high-asymmetric-cost decision problem.

## 2. Why a proxy task

PAM is too rare to train on directly. The CDC reports 167 cumulative U.S.
cases between 1962 and 2024 (CDC, *About Primary Amebic Meningoencephalitis*,
2025). Yoder et al. 2010 (Epidemiol Infect 138(7):968-975) document 111
cases over a 47-year window with a case-fatality rate of 99.1%. A standard
80/10/10 split on this corpus yields a test set that is single-digit
positive, incapable of distinguishing model behaviour from sampling noise.

The proxy task substitutes a *related* high-prevalence supervised problem
that exercises the same triage decision and the same calibration / OOD
machinery, then evaluates the trained model on the rare class as an OOD
held-out set.

## 3. The proxy task

**Source cohort:** MIMIC-IV (`hosp.labevents`, `hosp.diagnoses_icd`,
`hosp.microbiologyevents`) filtered to admissions with at least one CSF
analyte and at least one ICD-10 code in the meningitis ranges:

  * **G00.x**, Bacterial meningitis (positive class, "High")
  * **A87.x**, Viral meningitis (negative class, "Low")
  * **B60.2**, Naegleriasis / PAM (held-out OOD class)

The labeling rule treats bacterial meningitis as the positive class because
the clinical decision the model supports, *should the patient receive
empiric antibacterial therapy now?*, maps onto bacterial-vs-viral
discrimination at exactly the clinically actionable boundary. PAM and other
amebic encephalitides (B60.x) sit outside the training distribution and are
reserved for OOD evaluation.

**Features (extracted from MIMIC-IV per subject):**

| Feature | Source | Notes |
|---------|--------|-------|
| `csf_glucose` | labevents itemid 51790 | median over admissions |
| `csf_protein` | labevents itemid 51802 | median |
| `csf_wbc`     | labevents itemid 52286 | "Total Nucleated Cells, CSF", closest available; MIMIC-IV does not carry a "WBC, CSF" item |
| `csf_polys_pct` | labevents itemid 52281 | neutrophil predominance proxy |
| `microscopy` | microbiologyevents | 1 if any positive CSF Gram stain |
| `age` | patients table | computed at admission |
| `pcr`, `exposure` | not in MIMIC-IV | dropped or NaN-imputed for the proxy |

`pcr` and `exposure` are PAM-specific signals not represented in MIMIC. The
proxy classifier is fit without them (or with them as constant zero) to
match the available column set; the live-patient widget continues to expose
both because they are the dominant signals for actual PAM cases.

**Splits:** 60/20/20 train/val/test, stratified by `icd_label` and
group-disjoint by `subject_id`. With the group-disjoint splitter in
`ml/splits.py`, no patient appears in more than one partition.

## 4. Evaluation protocol

For each combination of model and ablation cell we report:

  * **AUC (calibrated probability)** with bootstrap 95% CI (n_resamples=2000,
    stratified, alpha=0.05) via `ml.metrics.bootstrap.bootstrap_ci`.
  * **Recall at the operating threshold** (the DCA-chosen threshold) with
    the same CI protocol.
  * **Conformal coverage at alpha = 0.05, 0.10, 0.20** on the held-out
    calibration split.
  * **ABSTAIN rate** at the chosen alpha.
  * **OOD detection AUC**: Mahalanobis, logit-energy, and neg-energy gates
    each evaluated on (in-distribution = bacterial+viral test rows) vs.
    (OOD = PAM rows from B60.2). This is the only experiment that uses the
    PAM rows.

The target empirical coverage matches 1 - alpha to within +/- 2 / (n+2) per
the Lei et al. 2018 bound. The PAM OOD AUC target is >= 0.85, well above
chance, distinctly below the perfect 1.0 that would suggest data leakage.

## 5. Why this is honest

* **No PAM-specific training.** The PAM rows are only ever seen at OOD
  evaluation time; the supervised loss never touches them. The headline
  classifier discriminates bacterial vs viral, which is a real, learnable,
  high-prevalence task.
* **No fabricated cases.** Every row comes from MIMIC-IV (when access is
  granted) or from `ml/case_series.synthesize_yoder_cohort` (which carries
  `source="synthetic_from_yoder2010"` and is excluded from quoted metrics).
* **No claim that the proxy = PAM.** The preprint explicitly states that the
  bacterial-vs-viral classifier is a *proxy for the calibration and OOD
  machinery*; the PAM-specific deployment claim requires the prospective
  validation flagged in the Limitations section.

## 6. Open dependencies

* PhysioNet credentialed access (CITI training + DUA). Blocks all real-data
  evaluation.
* Optional: Capewell LG et al., *J Pediatric Infect Dis Soc* 2015;4(4):e68-e75
  (PMID 26582886) for tabulated per-case PAM CSF values. Cope 2016 reports
  the patterns qualitatively only.

---

**References:**

* Yoder JS, Eddy BA, Visvesvara GS, Capewell L, Beach MJ. *Epidemiol Infect*
  2010;138(7):968-975. PMID 19845995.
* Cope JR, Ali IK. *Curr Infect Dis Rep* 2016;18(10):31. PMID 27614893.
* CDC. *About Primary Amebic Meningoencephalitis (PAM).* 2025.
  https://www.cdc.gov/naegleria/about/index.html
* Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. *J Am Stat Assoc*
  2018;113(523):1094-1111.
* Vovk V. *Mach Learn* 2013;92(2-3):349-376. (Mondrian / label-conditional
  conformal.)
