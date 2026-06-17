"""References page - Phase 4.5 Mini-2 T2.3.

Renders the 22 BibTeX entries from ``docs/references.bib`` as a
reviewer-friendly grouped list, ordered by category to mirror the
research-narrative structure (clinical → calibration → OOD →
governance → tools).

Q8.C locked: full references page is the discovery surface;
inline-tooltip references on the Predict page (heuristic per
p_high bucket) are a separate Mini-2 enhancement and consume the
same bib keys.

Each entry: ``[bib_key] Authors. Title. *Venue* Year. PMID/DOI.``
Anchor links via ``st.markdown`` heading IDs so the inline-tooltip
plumbing can deep-link to specific bib_keys.
"""
from __future__ import annotations

import streamlit as st

from app.disclaimer import render_disclaimer


st.set_page_config(page_title="References - Amoebanator 25")
render_disclaimer()

st.title("References")

st.caption(
    "22 entries from `docs/references.bib`. Methodology references "
    "(Guo 2017, Vovk 2005/2013, Lei 2018, Liu 2020, Lee 2018, "
    "Mitchell 2019, etc.) are intentionally retained at original "
    "publication year - post-2022 companion citations will be added "
    "during the medRxiv preprint prep (Phase 8.5)."
)


# -- PAM clinical (6 refs) ---------------------------------------------
st.subheader("PAM clinical")
st.markdown(
    """
- **`cope2016pam`** Cope JR, Ali IKM. *Primary Amebic Meningoencephalitis: What Have We Learned in the Last 5 Years?* **Current Infectious Disease Reports** 2016. PMID 27614893. DOI 10.1007/s11908-016-0539-4.
- **`yoder2010pam`** Yoder JS, Eddy BA, Visvesvara GS, Capewell L, Beach MJ. *The epidemiology of primary amoebic meningoencephalitis in the USA, 1962-2008.* **Epidemiology and Infection** 2010. PMID 19845995. DOI 10.1017/S0950268809991014.
- **`capewell2015pam`** Capewell LG, Harris AM, Yoder JS, Cope JR, et al. *Diagnosis, Clinical Course, and Treatment of Primary Amoebic Meningoencephalitis in the United States, 1937-2013: A Review of the Historical Literature.* **Journal of the Pediatric Infectious Diseases Society** 2015. PMID 26582886. DOI 10.1093/jpids/piu103.
- **`cdc2025pam`** Centers for Disease Control and Prevention. *About Primary Amebic Meningoencephalitis (PAM).* 2025. Source for the 97% case-fatality figure (167 cumulative U.S. cases / 4 survivors through 2024).
- **`tunkel2004idsa`** Tunkel AR, Hartman BJ, Kaplan SL, et al. *Practice guidelines for the management of bacterial meningitis.* **Clinical Infectious Diseases** 2004. PMID 15494903. DOI 10.1086/425368. IDSA canonical reference; used in the Q11 inline-tooltip heuristic.
- **`seehusen2003csf`** Seehusen DA, Reeves MM, Fomin DA. *Cerebrospinal fluid analysis.* **American Family Physician** 2003. PMID 14524396. AAFP review used in the Q11 normal-CSF tooltip bucket.
"""
)


# -- Calibration & conformal (6 refs) ----------------------------------
st.subheader("Calibration & conformal prediction")
st.markdown(
    """
- **`guo2017calibration`** Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration of Modern Neural Networks.* **ICML** 2017. The Phase 4.5 demo applies temperature scaling per this paper; T=0.27 amplification is disclosed on the Predict page badge + About-page §3.
- **`vovk2005alrw`** Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer 2005. Foundational split-conformal reference.
- **`vovk2013mondrian`** Vovk V. *Conditional validity of inductive conformal predictors.* **Machine Learning** 2013. DOI 10.1007/s10994-013-5355-6. Source for the label-conditional Mondrian split conformal in `ml/conformal_advanced.py`.
- **`lei2018distributionfree`** Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. *Distribution-Free Predictive Inference for Regression.* **Journal of the American Statistical Association** 2018. DOI 10.1080/01621459.2017.1307116. Source for the split-conformal coverage bound `1 − α + 2/(n+2)` cited in `ml/conformal_advanced.py`.
- **`platt1999probabilistic`** Platt J. *Probabilistic outputs for support vector machines.* **Advances in Large-Margin Classifiers** 1999. Used by `ml/baselines/logistic.py` (Platt-scaled LR).
- **`niculescu2005calibration`** Niculescu-Mizil A, Caruana R. *Predicting good probabilities with supervised learning.* **ICML** 2005. Used by `ml/baselines/random_forest.py` (isotonic calibration fallback).
"""
)


# -- OOD detection (2 refs) --------------------------------------------
st.subheader("Out-of-distribution detection")
st.markdown(
    """
- **`lee2018mahalanobis`** Lee K, Lee K, Lee H, Shin J. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* **NeurIPS** 2018. Source for the Mahalanobis OOD gate at `ml/ood_simple.py` + `ml/robust.py`.
- **`liu2020energy`** Liu W, Wang X, Owens JD, Li Y. *Energy-based Out-of-distribution Detection.* **NeurIPS** 2020. Source for the canonical "energy > τ → ABSTAIN" semantics fixed in commit `b8f62e3` (Q5 / Q11.A.fix gate inversion); the Phase 4.5 web demo's OOD abstain reason `LogitEnergyAboveOODShift` is named directly after this paper's framing.
"""
)


# -- Governance & model documentation frameworks (7 refs) --------------
st.subheader("Governance & model documentation")
st.markdown(
    """
- **`mitchell2019modelcards`** Mitchell M, Wu S, Zaldivar A, et al. *Model Cards for Model Reporting.* **FAccT** 2019. DOI 10.1145/3287560.3287596. Format used for `docs/model_card.md`.
- **`vasey2022decideai`** Vasey B, Nagendran M, Campbell B, et al. *DECIDE-AI: Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence.* **Nature Medicine** 2022. PMID 35585198. DOI 10.1038/s41591-022-01772-9. Format used for `docs/decide-ai.md`.
- **`collins2024tripodai`** Collins GS, Moons KGM, Dhiman P, et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods.* **BMJ** 2024. PMID 38626948. DOI 10.1136/bmj-2023-078378. Format used for `docs/tripod-ai.md`.
- **`collins2015tripod`** Collins GS, et al. *TRIPOD: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis.* **BMJ** 2015. Original TRIPOD; comparison anchor for the TRIPOD+AI doc.
- **`gebru2021datasheets`** Gebru T, Morgenstern J, Vecchione B, et al. *Datasheets for Datasets.* **Communications of the ACM** 2021. Format used for `docs/data_card.md`.
- **`hipaa2012deident`** U.S. Department of Health and Human Services. *Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the HIPAA Privacy Rule.* 2012. Source for the Safe Harbor wrapper logic in `ml/data_loader.py`.
- **`vickers2006dca`** Vickers AJ, Elkin EB. *Decision curve analysis: a novel method for evaluating prediction models.* **Medical Decision Making** 2006. PMID 17099194. DOI 10.1177/0272989X06295361. Methodology for Phase 6 DCA (deferred until MIMIC-IV cohort lands).
"""
)


# -- Tools (1 ref) -----------------------------------------------------
st.subheader("Tools")
st.markdown(
    """
- **`ke2017lightgbm`** Ke G, Meng Q, Finley T, et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* **NeurIPS** 2017. Used by `ml/baselines/gbm.py` for the gradient-boosting baseline (with sklearn `GradientBoostingClassifier` fallback when LightGBM unavailable).
"""
)


# Footer pointer
st.caption(
    "Full BibTeX source: `docs/references.bib` in the repo "
    "(github.com/ljm234/amoebanator-25). Reviewers wanting machine-"
    "readable citations should pull the .bib file directly."
)
