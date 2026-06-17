"""Phase 2.3 - tests for ml.case_series (Yoder 2010, CDC, Cope 2016 constants)."""
from __future__ import annotations

import numpy as np
import pytest

from ml.case_series import (
    CDC_AGGREGATE,
    COPE_QUALITATIVE,
    YODER_2010,
    published_constants,
    synthesize_yoder_cohort,
    yoder_exposure_probabilities,
    yoder_male_fraction,
)


def test_yoder_constants_match_paper() -> None:
    """Hard-coded Yoder 2010 numbers must match the published paper exactly."""
    assert YODER_2010.n_cases == 111
    assert YODER_2010.n_fatal == 110
    assert YODER_2010.age_median_years == 12.0
    assert YODER_2010.male_n == 88
    assert YODER_2010.female_n == 23
    assert YODER_2010.exposure_known_n == 91
    # Exposure distribution must sum to known n
    assert sum(YODER_2010.exposure_distribution.values()) == 91


def test_cdc_constants_have_correct_cfr() -> None:
    """CDC: 167 cases, 4 survivors → CFR ≈ 97.6%."""
    assert CDC_AGGREGATE.cumulative_cases_through_2024 == 167
    assert CDC_AGGREGATE.cumulative_survivors_through_2024 == 4
    assert CDC_AGGREGATE.case_fatality_rate == pytest.approx(163 / 167, rel=1e-9)


def test_cope_qualitative_present() -> None:
    """Cope 2016 has only qualitative CSF - make sure that's documented."""
    assert "qualitative" in COPE_QUALITATIVE.citation.lower() or "see Capewell" in COPE_QUALITATIVE.citation
    assert COPE_QUALITATIVE.incubation_days_median == 5.0


def test_male_fraction_matches_published() -> None:
    """88 / (88 + 23) = 0.7928..."""
    assert yoder_male_fraction() == pytest.approx(88 / 111, rel=1e-9)


def test_exposure_probabilities_sum_to_one() -> None:
    probs = yoder_exposure_probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_exposure_probabilities_lake_dominant() -> None:
    """Lakes/ponds/reservoirs is 67/91 ≈ 73.6% of known exposures."""
    probs = yoder_exposure_probabilities()
    assert probs["lake_pond_reservoir"] == pytest.approx(67 / 91, rel=1e-9)
    # Lake-class > sum of all others combined
    assert probs["lake_pond_reservoir"] > sum(v for k, v in probs.items() if k != "lake_pond_reservoir")


def test_synthesize_yoder_cohort_marginals_match() -> None:
    """At n=2000 the male fraction and median age should track Yoder."""
    df = synthesize_yoder_cohort(n=2000, seed=0)
    assert len(df) == 2000
    male_frac = float((df["sex"] == "M").mean())
    assert abs(male_frac - yoder_male_fraction()) < 0.05
    assert int(df["age"].median()) in range(int(YODER_2010.age_median_years) - 5, int(YODER_2010.age_median_years) + 5)


def test_synthesize_yoder_cohort_columns_complete() -> None:
    df = synthesize_yoder_cohort(n=10, seed=1)
    required = {"case_id", "source", "physician", "age", "sex",
                "csf_glucose", "csf_protein", "csf_wbc", "symptoms",
                "pcr", "microscopy", "exposure", "risk_score", "risk_label", "comments"}
    assert required.issubset(df.columns)
    assert (df["source"] == "synthetic_from_yoder2010").all()
    assert (df["risk_label"] == "High").all()


def test_synthesize_yoder_cohort_age_within_published_range() -> None:
    df = synthesize_yoder_cohort(n=500, seed=2)
    assert df["age"].min() >= int(YODER_2010.age_min_years)
    assert df["age"].max() <= int(YODER_2010.age_max_years)


def test_synthesize_yoder_cohort_csf_in_pam_pattern() -> None:
    """PAM-typical CSF: low glucose, high protein, high WBC."""
    df = synthesize_yoder_cohort(n=1000, seed=3)
    assert df["csf_glucose"].median() < 40.0  # low glucose
    assert df["csf_protein"].median() > 100.0  # high protein
    assert df["csf_wbc"].median() > 500.0  # high WBC


def test_synthesize_yoder_cohort_zero_n_returns_empty() -> None:
    df = synthesize_yoder_cohort(n=0)
    assert df.empty


def test_synthesize_yoder_cohort_unknown_csf_pattern_raises() -> None:
    with pytest.raises(ValueError, match="csf_pattern"):
        synthesize_yoder_cohort(n=10, csf_pattern="bogus")


def test_synthesize_yoder_cohort_seed_reproducible() -> None:
    a = synthesize_yoder_cohort(n=20, seed=42)
    b = synthesize_yoder_cohort(n=20, seed=42)
    np.testing.assert_array_equal(a["age"].to_numpy(), b["age"].to_numpy())


def test_published_constants_includes_all_three_sources() -> None:
    pc = published_constants()
    assert {"yoder2010", "cdc_aggregate", "cope2016_qualitative"} <= set(pc.keys())
    # Citations must be present
    for source in pc.values():
        assert isinstance(source, dict) and "citation" in source
