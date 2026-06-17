"""
Phase 2.3 - Published PAM case-series summary statistics.

This module encodes *only* what is published in peer-reviewed sources, with
explicit citations. It does not invent any case-level data. The summary
statistics support two downstream uses:

  1. As a sanity-check distribution for the Streamlit live-patient widget
     (so default form values match the median age and exposure pattern of
     real US PAM cases).
  2. As a generator of synthetic case-level rows whose marginals match
     published distributions, when the bundled simulated data needs to be
     augmented for unit tests of the OOD pipeline. Generated rows are
     marked source="synthetic_from_yoder2010" so they cannot be confused
     with real records downstream.

Sources:
  * Yoder JS, Eddy BA, Visvesvara GS, Capewell L, Beach MJ.
    "The epidemiology of primary amoebic meningoencephalitis in the USA,
    1962-2008." Epidemiol Infect 2010;138(7):968-975.
    DOI 10.1017/S0950268809991014 ; PMID 19845995.
  * Cope JR, Ali IK. "Primary Amebic Meningoencephalitis: What Have We
    Learned in the Last 5 Years?" Curr Infect Dis Rep 2016;18(10):31.
    DOI 10.1007/s11908-016-0539-4 ; PMID 27614893.
  * CDC. "About Primary Amebic Meningoencephalitis (PAM)."
    https://www.cdc.gov/naegleria/about/index.html (last verified 2026-04-24).

Cope 2016 does NOT publish numeric CSF lab summary statistics - the CSF
distributions below are stated qualitatively in that paper ("predominantly
neutrophilic pleocytosis, elevated protein, low glucose"). For numeric CSF
ranges in PAM cases, Capewell LG et al., J Pediatric Infect Dis Soc 2015;
4(4):e68-e75 (PMID 26582886) is the better source; we encode the
qualitative ranges here and flag the citation gap explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YoderEpidemiology:
    """
    US PAM cases 1962-2008. n=111 reported; 110/111 fatal (CFR 99.1%).
    Source: Yoder et al. 2010, Epidemiol Infect 138(7):968-975.
    """
    n_cases: int = 111
    n_fatal: int = 110
    age_median_years: float = 12.0
    age_min_years: float = 8.0 / 12.0
    age_max_years: float = 66.0
    male_n: int = 88
    female_n: int = 23
    exposure_known_n: int = 91
    # Exposure source distribution (n with known exposure = 91)
    exposure_distribution: dict[str, int] = field(default_factory=lambda: {
        "lake_pond_reservoir": 67,
        "canal_ditch_puddle": 7,
        "river_stream": 7,
        "geothermal_hot_spring": 5,
        "untreated_drinking_water_recreational": 3,
        "swimming_pool": 2,
    })
    geographic_concentration: dict[str, int] = field(default_factory=lambda: {
        "Texas": 30,
        "Florida": 29,
    })
    citation: str = (
        "Yoder JS, Eddy BA, Visvesvara GS, Capewell L, Beach MJ. "
        "Epidemiol Infect 2010;138(7):968-975. PMID 19845995."
    )


@dataclass(frozen=True)
class CDCAggregateStats:
    """CDC About page (last verified 2026-04-24)."""
    cumulative_cases_through_2024: int = 167
    cumulative_survivors_through_2024: int = 4
    case_fatality_rate: float = 163.0 / 167.0
    typical_annual_us_cases: str = "fewer than 10"
    citation: str = "CDC. https://www.cdc.gov/naegleria/about/index.html (verified 2026-04-24)."


@dataclass(frozen=True)
class CopeQualitativeCSF:
    """
    Cope 2016 reports CSF abnormalities qualitatively only. The numeric
    plausibility ranges below are *not* from Cope 2016 - they are clinical
    reference ranges for bacterial-meningitis-pattern CSF, used to bound
    synthetic sampling. Real numeric ranges per PAM case should be sourced
    from Capewell 2015 (PMID 26582886) when that paper is added.
    """
    glucose_pattern: str = "low (typically <40 mg/dL)"
    protein_pattern: str = "elevated (typically >100 mg/dL)"
    wbc_pattern: str = "neutrophilic pleocytosis (typically >1000 cells/µL)"
    incubation_days_median: float = 5.0
    onset_to_death_days_median: float = 5.0
    citation: str = (
        "Cope JR, Ali IK. Curr Infect Dis Rep 2016;18(10):31. "
        "PMID 27614893. Note: numeric CSF ranges qualitative only; "
        "see Capewell 2015 PMID 26582886 for tabulated values."
    )


YODER_2010 = YoderEpidemiology()
CDC_AGGREGATE = CDCAggregateStats()
COPE_QUALITATIVE = CopeQualitativeCSF()


def yoder_male_fraction() -> float:
    return YODER_2010.male_n / float(YODER_2010.male_n + YODER_2010.female_n)


def yoder_exposure_probabilities() -> dict[str, float]:
    total = float(sum(YODER_2010.exposure_distribution.values()))
    return {k: v / total for k, v in YODER_2010.exposure_distribution.items()}


def synthesize_yoder_cohort(
    n: int = 100,
    seed: int = 0,
    csf_pattern: str = "pam_typical",
) -> pd.DataFrame:
    """
    Generate `n` synthetic PAM-like rows whose marginals match published
    Yoder 2010 distributions (age, sex, exposure source). Every row carries
    `source="synthetic_from_yoder2010"` and `risk_label="High"` so it cannot
    be mistaken for a real case in downstream analyses.

    CSF labs are sampled from Cope 2016's qualitative ranges (low glucose,
    high protein, high WBC with neutrophil predominance). When real per-case
    CSF values from Capewell 2015 are added to this module, swap the sampler
    to use the empirical distribution instead.
    """
    if n <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    # Age: log-normal anchored at median 12, capped at the published range.
    log_med = float(np.log(YODER_2010.age_median_years))
    log_sigma = 0.7
    ages = rng.lognormal(mean=log_med, sigma=log_sigma, size=n)
    ages = np.clip(ages, YODER_2010.age_min_years, YODER_2010.age_max_years).round(0).astype(int)
    sex = np.where(rng.random(n) < yoder_male_fraction(), "M", "F")

    probs = yoder_exposure_probabilities()
    sources = list(probs.keys())
    weights = np.array(list(probs.values()), dtype=float)
    weights = weights / weights.sum()
    exposure_kind = rng.choice(sources, size=n, p=weights)

    if csf_pattern == "pam_typical":
        # Glucose ~ N(20, 8) clipped to [5, 60]
        glucose = np.clip(rng.normal(20.0, 8.0, size=n), 5.0, 60.0)
        # Protein ~ Lognormal anchored at 200 mg/dL
        protein = np.exp(rng.normal(np.log(200.0), 0.6, size=n)).clip(40.0, 1500.0)
        # WBC ~ Lognormal anchored at 1500 cells/µL
        wbc = np.exp(rng.normal(np.log(1500.0), 0.7, size=n)).clip(100.0, 12000.0)
    else:
        raise ValueError(f"unknown csf_pattern={csf_pattern!r}")

    pcr = (rng.random(n) < 0.7).astype(int)
    microscopy = (rng.random(n) < 0.5).astype(int)
    exposure_flag = np.ones(n, dtype=int)  # exposure is the inclusion criterion

    df = pd.DataFrame({
        "case_id": [f"yoder2010_synth_{i:04d}" for i in range(n)],
        "source": "synthetic_from_yoder2010",
        "physician": "synthetic",
        "age": ages,
        "sex": sex,
        "csf_glucose": np.round(glucose, 1),
        "csf_protein": np.round(protein, 1),
        "csf_wbc": np.round(wbc, 0).astype(int),
        "symptoms": "fever;headache;nuchal_rigidity",
        "pcr": pcr,
        "microscopy": microscopy,
        "exposure": exposure_flag,
        "exposure_kind": exposure_kind,
        "risk_score": 14,
        "risk_label": "High",
        "comments": "Synthesized from Yoder 2010 marginals; not a real case.",
    })
    return df


def published_constants() -> dict[str, object]:
    """One-shot snapshot of every published number this module encodes."""
    return {
        "yoder2010": {
            "n_cases": YODER_2010.n_cases,
            "n_fatal": YODER_2010.n_fatal,
            "case_fatality_rate": YODER_2010.n_fatal / YODER_2010.n_cases,
            "age_median_years": YODER_2010.age_median_years,
            "age_min_years": YODER_2010.age_min_years,
            "age_max_years": YODER_2010.age_max_years,
            "male_fraction": yoder_male_fraction(),
            "exposure_distribution": dict(YODER_2010.exposure_distribution),
            "geographic_concentration": dict(YODER_2010.geographic_concentration),
            "citation": YODER_2010.citation,
        },
        "cdc_aggregate": {
            "cumulative_cases_through_2024": CDC_AGGREGATE.cumulative_cases_through_2024,
            "cumulative_survivors_through_2024": CDC_AGGREGATE.cumulative_survivors_through_2024,
            "case_fatality_rate": CDC_AGGREGATE.case_fatality_rate,
            "typical_annual_us_cases": CDC_AGGREGATE.typical_annual_us_cases,
            "citation": CDC_AGGREGATE.citation,
        },
        "cope2016_qualitative": {
            "glucose_pattern": COPE_QUALITATIVE.glucose_pattern,
            "protein_pattern": COPE_QUALITATIVE.protein_pattern,
            "wbc_pattern": COPE_QUALITATIVE.wbc_pattern,
            "incubation_days_median": COPE_QUALITATIVE.incubation_days_median,
            "onset_to_death_days_median": COPE_QUALITATIVE.onset_to_death_days_median,
            "citation": COPE_QUALITATIVE.citation,
        },
    }
