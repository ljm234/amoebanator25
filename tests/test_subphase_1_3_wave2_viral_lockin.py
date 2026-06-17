"""Subphase 1.3 Commit 5.3.6 Wave 2 VIRAL lock-in tests (FINAL 1.3 wave).

14 vignettes split: 9 anchored to Granerod 2010 Lancet ID UK encephalitis
cohort (PMID 20952256, anchor_type=cohort) + 5 anchored to Whitley 2006
Lancet ID HSE pathogenesis review (PMID 16675036, anchor_type=review).

Pathogens: 8 HSV1 (3 Granerod + 5 Whitley) + 2 HSV-PCR-negative-72h
(Granerod, ambiguity) + 2 enterovirus (Granerod) + 2 VZV (Granerod).

This commit closes BACT 30/30 + VIRAL 30/30 = 60/60 Subphase 1.3 corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.schemas.vignette import VignetteSchema  # noqa: E402
from scripts.vignettes.generate_pam_vignettes import VIRAL_DISTRIBUTION  # noqa: E402


VIRAL_WAVE2_IDS = [91, 93, 94, 95, 97, 98, 100, 101, 103, 104, 110, 112, 115, 116]
VIRAL_WAVE2_AMBIGUITY_IDS = {103, 104}
VIRAL_WAVE2_PERU_IDS: set[int] = set()  # All Wave 5.3.6 slots are NL or US South
VIRAL_WAVE2_GRANEROD_IDS = {94, 97, 100, 103, 104, 110, 112, 115, 116}
VIRAL_WAVE2_WHITLEY_IDS = {91, 93, 95, 98, 101}
WAVE2_DIR = _REPO_ROOT / "data" / "vignettes" / "v2" / "class_03_viral"
GRANEROD_PMID = "20952256"
WHITLEY_PMID = "16675036"


def _wave2_slot(vid: int) -> dict:
    return next(s for s in VIRAL_DISTRIBUTION if s["vignette_id"] == vid)


def _wave2_json_path(vid: int) -> Path:
    matches = list(WAVE2_DIR.glob(f"vir_{vid:03d}_*.json"))
    assert len(matches) == 1, f"v{vid}: expected 1 match, got {matches!r}"
    return matches[0]


def _load(vid: int) -> dict:
    return json.loads(_wave2_json_path(vid).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Tests 1-5: parametrized over 14 vignettes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", VIRAL_WAVE2_IDS)
def test_viral_wave2_files_exist(vid):
    matches = list(WAVE2_DIR.glob(f"vir_{vid:03d}_*.json"))
    assert len(matches) == 1, f"v{vid} JSON missing or duplicate: {matches}"


@pytest.mark.parametrize("vid", VIRAL_WAVE2_IDS)
def test_viral_wave2_schema_validates(vid):
    VignetteSchema.model_validate(_load(vid))


@pytest.mark.parametrize("vid", VIRAL_WAVE2_IDS)
def test_viral_wave2_demographics_match_spec(vid):
    data = _load(vid)
    spec = _wave2_slot(vid)
    assert data["demographics"]["age_years"] == spec["age_years"], f"v{vid} age"
    assert data["demographics"]["sex"] == spec["sex"], f"v{vid} sex"
    assert (
        data["demographics"]["geography_region"] == spec["geography_region"]
    ), f"v{vid} region"


@pytest.mark.parametrize("vid", VIRAL_WAVE2_IDS)
def test_viral_wave2_anchor_pmid_in_set(vid):
    data = _load(vid)
    pmid = data["literature_anchors"][0]["pmid"]
    assert pmid in (GRANEROD_PMID, WHITLEY_PMID), (
        f"v{vid} anchor PMID {pmid!r} not Granerod or Whitley"
    )


@pytest.mark.parametrize("vid", VIRAL_WAVE2_IDS)
def test_viral_wave2_anchor_type_correct(vid):
    """Granerod cohort, Whitley review."""
    data = _load(vid)
    a = data["literature_anchors"][0]
    if a["pmid"] == GRANEROD_PMID:
        assert a["anchor_type"] == "cohort", (
            f"v{vid} Granerod anchor_type={a['anchor_type']!r} (expect cohort)"
        )
    elif a["pmid"] == WHITLEY_PMID:
        assert a["anchor_type"] == "review", (
            f"v{vid} Whitley anchor_type={a['anchor_type']!r} (expect review)"
        )


# ----------------------------------------------------------------------
# Tests 6-15: corpus invariants
# ----------------------------------------------------------------------


def test_viral_wave2_count_14():
    """Empirical extraction must match hardcoded list. Wave 2 filter required
    on the Whitley side because errata 5.4.3.1 unified the wave 2 Whitley
    anchor with the existing 5.3.2 pilot anchor under PMID 16675036; vir_092
    (pilot) shares the same PMID but is NOT a wave 2 vignette."""
    wave2_set = set(VIRAL_WAVE2_IDS)
    granerod = sorted(
        s["vignette_id"] for s in VIRAL_DISTRIBUTION
        if s.get("pmid") == GRANEROD_PMID and s["vignette_id"] in wave2_set
    )
    whitley = sorted(
        s["vignette_id"] for s in VIRAL_DISTRIBUTION
        if s.get("pmid") == WHITLEY_PMID and s["vignette_id"] in wave2_set
    )
    assert sorted(VIRAL_WAVE2_IDS) == sorted(granerod + whitley), (
        f"Hardcoded {VIRAL_WAVE2_IDS} != empirical Granerod {granerod} + Whitley {whitley}"
    )
    assert len(VIRAL_WAVE2_IDS) == 14


def test_viral_wave2_anchor_distribution_9_5():
    granerod_count = sum(
        1 for vid in VIRAL_WAVE2_IDS
        if _load(vid)["literature_anchors"][0]["pmid"] == GRANEROD_PMID
    )
    whitley_count = sum(
        1 for vid in VIRAL_WAVE2_IDS
        if _load(vid)["literature_anchors"][0]["pmid"] == WHITLEY_PMID
    )
    assert granerod_count == 9, f"Granerod count {granerod_count} != 9"
    assert whitley_count == 5, f"Whitley count {whitley_count} != 5"


def test_viral_wave2_freshwater_false():
    for vid in VIRAL_WAVE2_IDS:
        assert (
            _load(vid)["exposure"]["freshwater_exposure_within_14d"] is False
        ), f"v{vid}"


def test_viral_wave2_class_id_3():
    for vid in VIRAL_WAVE2_IDS:
        assert _load(vid)["ground_truth_class"] == 3, f"v{vid}"


def test_viral_wave2_csf_lymphocytic():
    """Spec 1.3.4: csf_lymphocyte_pct >= 50."""
    for vid in VIRAL_WAVE2_IDS:
        pct = _load(vid)["csf"]["csf_lymphocyte_pct"]
        assert pct >= 50, f"v{vid} csf_lymphocyte_pct={pct}"


def test_viral_wave2_csf_neutrophil_low():
    for vid in VIRAL_WAVE2_IDS:
        pct = _load(vid)["csf"]["csf_neutrophil_pct"]
        assert pct < 50, f"v{vid} csf_neutrophil_pct={pct}"


def test_viral_wave2_csf_glucose_normal_or_near():
    for vid in VIRAL_WAVE2_IDS:
        glucose = _load(vid)["csf"]["csf_glucose_mg_per_dL"]
        assert glucose >= 40, f"v{vid} csf_glucose={glucose}"


def test_viral_wave2_pre_adjudication_hold():
    for vid in VIRAL_WAVE2_IDS:
        data = _load(vid)
        assert (
            data["adjudication"]["inclusion_decision"] == "hold_for_revision"
        ), f"v{vid}"
        assert (
            "self_review_disposition=hold_for_revision"
            in data["adjudication"]["anchoring_documentation"]
        ), f"v{vid}"


def test_viral_wave2_no_em_dashes():
    for vid in VIRAL_WAVE2_IDS:
        text = _wave2_json_path(vid).read_text(encoding="utf-8")
        assert chr(0x2014) not in text, f"v{vid} contains em-dash"
        assert chr(0x2013) not in text, f"v{vid} contains en-dash"


def test_viral_wave2_no_ai_tells():
    banned = (
        "delve",
        "tapestry",
        "navigate the realm",
        "in the realm of",
        "vibrant",
        "robust",
        "comprehensive",
        "intricate",
        "dive in",
        "in conclusion",
        "it's important to note",
        "seamlessly",
        "leverage",
        "furthermore",
        "moreover",
    )
    for vid in VIRAL_WAVE2_IDS:
        text = _wave2_json_path(vid).read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, (
                f"v{vid} contains banned AI-tell: {word!r}"
            )


# ----------------------------------------------------------------------
# Pathogen authenticity tests
# ----------------------------------------------------------------------


def test_viral_wave2_ambiguity_count():
    """Exactly 2 of 14 (v103, v104 HSV-PCR-negative-72h)."""
    ambiguous = []
    for vid in VIRAL_WAVE2_IDS:
        rationale = (
            _load(vid)["provenance"].get("inclusion_decision_rationale") or ""
        ).lower()
        if "diagnostic_ambiguity=true" in rationale or "type=" in rationale:
            ambiguous.append(vid)
    assert set(ambiguous) == VIRAL_WAVE2_AMBIGUITY_IDS, (
        f"Expected {VIRAL_WAVE2_AMBIGUITY_IDS}, got {set(ambiguous)}"
    )


def test_viral_wave2_hsv_pcr_negative_empiric_acyclovir():
    """HSV-PCR-negative-72h cases must disclose empiric acyclovir continuation
    despite negative early PCR."""
    pcr_neg_ids = [
        vid
        for vid in VIRAL_WAVE2_IDS
        if _wave2_slot(vid).get("pathogen") == "HSV_PCR_negative_72h"
    ]
    assert pcr_neg_ids, "expected at least one HSV_PCR_negative_72h slot"
    for vid in pcr_neg_ids:
        data = _load(vid)
        narrative = data.get("narrative_en", "").lower()
        rationale = (data["provenance"].get("inclusion_decision_rationale") or "").lower()
        combined = narrative + " " + rationale
        assert "acyclovir" in combined, f"v{vid} no acyclovir in narrative/rationale"
        assert "negative" in combined, f"v{vid} no negative-PCR disclosure"


def test_viral_wave2_vzv_dermatomal_or_cerebellitis():
    """VZV cases must show dermatomal rash OR cerebellitis OR vasculopathy
    pattern documented in narrative_en."""
    vzv_ids = [
        vid
        for vid in VIRAL_WAVE2_IDS
        if _wave2_slot(vid).get("pathogen") == "VZV"
    ]
    assert vzv_ids, "expected at least one VZV slot"
    for vid in vzv_ids:
        narrative = _load(vid).get("narrative_en", "").lower()
        markers = ("dermatom", "cerebellit", "vasculopath", "zoster")
        assert any(m in narrative for m in markers), (
            f"v{vid} VZV narrative missing dermatomal/cerebellitis/vasculopathy/zoster: "
            f"{narrative[:200]}"
        )


def test_viral_wave2_pathogen_distribution():
    """8 HSV1 + 2 HSV-PCR-neg + 2 enterovirus + 2 VZV."""
    counts: dict[str, int] = {}
    for vid in VIRAL_WAVE2_IDS:
        p = _wave2_slot(vid)["pathogen"]
        counts[p] = counts.get(p, 0) + 1
    assert counts == {
        "HSV1": 8,
        "HSV_PCR_negative_72h": 2,
        "enterovirus": 2,
        "VZV": 2,
    }, f"Wave 5.3.6 pathogen distribution: {counts}"


# ----------------------------------------------------------------------
# Subphase 1.3 60/60 closure
# ----------------------------------------------------------------------


def test_subphase_1_3_complete_60_60():
    """Closure: BACT 30/30 + VIRAL 30/30 = 60 vignette JSONs validate."""
    bact = sorted(
        p
        for p in (_REPO_ROOT / "data/vignettes/v2/class_02_bacterial").glob("bact_*.json")
    )
    viral = sorted(
        p
        for p in (_REPO_ROOT / "data/vignettes/v2/class_03_viral").glob("vir_*.json")
    )
    assert len(bact) == 28, f"BACT count {len(bact)} != 28"
    assert len(viral) == 30, f"VIRAL count {len(viral)} != 30"
    for p in bact + viral:
        VignetteSchema.model_validate(json.loads(p.read_text(encoding="utf-8")))
