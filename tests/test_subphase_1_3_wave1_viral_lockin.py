"""Subphase 1.3 Commit 5.3.5 Wave 1 VIRAL lock-in tests.

13 vignettes all anchored to Tyler KL 2018 NEJM viral encephalitis review
(PMID 30089069). Pathogens: 3 HSV1 + 5 enterovirus + 2 HSV2 + 2 dengue +
1 EEE per VIRAL_DISTRIBUTION spec. 2 ambiguity slots (v113 HSV2 first-
episode with prominent meningismus, v117 dengue with prominent CNS arbo
overlap).

Filename convention: vir_NNN_*.json (not viral_NNN_*.json).
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


VIRAL_WAVE1_IDS = [96, 99, 102, 106, 107, 108, 109, 111, 113, 114, 117, 119, 120]
VIRAL_WAVE1_AMBIGUITY_IDS = {113, 117}
VIRAL_WAVE1_PERU_IDS = {99, 117, 119}
WAVE1_DIR = _REPO_ROOT / "data" / "vignettes" / "v2" / "class_03_viral"
TYLER_PMID = "30089069"
PUCCIONI_PMID = "38157877"
VIRAL_WAVE1_PUCCIONI_IDS = {117, 119}
VIRAL_WAVE1_TYLER_IDS = [v for v in VIRAL_WAVE1_IDS if v not in VIRAL_WAVE1_PUCCIONI_IDS]


def _wave1_slot(vid: int) -> dict:
    return next(s for s in VIRAL_DISTRIBUTION if s["vignette_id"] == vid)


def _wave1_json_path(vid: int) -> Path:
    matches = list(WAVE1_DIR.glob(f"vir_{vid:03d}_*.json"))
    assert len(matches) == 1, f"v{vid}: expected 1 match, got {matches!r}"
    return matches[0]


def _load(vid: int) -> dict:
    return json.loads(_wave1_json_path(vid).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Tests 1-5: parametrized over 13 vignettes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", VIRAL_WAVE1_IDS)
def test_viral_wave1_files_exist(vid):
    matches = list(WAVE1_DIR.glob(f"vir_{vid:03d}_*.json"))
    assert len(matches) == 1, f"v{vid} JSON missing or duplicate: {matches}"


@pytest.mark.parametrize("vid", VIRAL_WAVE1_IDS)
def test_viral_wave1_schema_validates(vid):
    VignetteSchema.model_validate(_load(vid))


@pytest.mark.parametrize("vid", VIRAL_WAVE1_IDS)
def test_viral_wave1_demographics_match_spec(vid):
    data = _load(vid)
    spec = _wave1_slot(vid)
    assert data["demographics"]["age_years"] == spec["age_years"], f"v{vid} age"
    assert data["demographics"]["sex"] == spec["sex"], f"v{vid} sex"
    assert (
        data["demographics"]["geography_region"] == spec["geography_region"]
    ), f"v{vid} region"


@pytest.mark.parametrize("vid", VIRAL_WAVE1_IDS)
def test_viral_wave1_anchor_pmid_tyler(vid):
    data = _load(vid)
    expected = PUCCIONI_PMID if vid in VIRAL_WAVE1_PUCCIONI_IDS else TYLER_PMID
    assert data["literature_anchors"][0]["pmid"] == expected, f"v{vid} anchor"


@pytest.mark.parametrize("vid", VIRAL_WAVE1_IDS)
def test_viral_wave1_anchor_type_review(vid):
    data = _load(vid)
    assert data["literature_anchors"][0]["anchor_type"] == "review", (
        f"v{vid} anchor_type must be review"
    )


# ----------------------------------------------------------------------
# Tests 6-13: corpus-level invariants
# ----------------------------------------------------------------------


def test_viral_wave1_count_13():
    """Empirical count from VIRAL_DISTRIBUTION must match hardcoded list."""
    empirical_tyler = sorted(
        s["vignette_id"]
        for s in VIRAL_DISTRIBUTION
        if s.get("vignette_id") in VIRAL_WAVE1_IDS and s.get("pmid") == TYLER_PMID
    )
    empirical_puccioni = sorted(
        s["vignette_id"]
        for s in VIRAL_DISTRIBUTION
        if s.get("vignette_id") in VIRAL_WAVE1_IDS and s.get("pmid") == PUCCIONI_PMID
    )
    assert empirical_tyler == sorted(VIRAL_WAVE1_TYLER_IDS), (
        f"Empirical Tyler {empirical_tyler} != {sorted(VIRAL_WAVE1_TYLER_IDS)}"
    )
    assert empirical_puccioni == sorted(VIRAL_WAVE1_PUCCIONI_IDS), (
        f"Empirical Puccioni {empirical_puccioni} != {sorted(VIRAL_WAVE1_PUCCIONI_IDS)}"
    )
    assert len(VIRAL_WAVE1_IDS) == 13


def test_viral_wave1_freshwater_false():
    """Spec 1.3.10 sanity: viral non-amoebic = no freshwater."""
    for vid in VIRAL_WAVE1_IDS:
        assert (
            _load(vid)["exposure"]["freshwater_exposure_within_14d"] is False
        ), f"v{vid}"


def test_viral_wave1_class_id_3():
    for vid in VIRAL_WAVE1_IDS:
        assert _load(vid)["ground_truth_class"] == 3, f"v{vid}"


def test_viral_wave1_csf_lymphocytic():
    """Spec 1.3.4 Class 3 viral mandate: csf_lymphocyte_pct >= 50."""
    for vid in VIRAL_WAVE1_IDS:
        pct = _load(vid)["csf"]["csf_lymphocyte_pct"]
        assert pct >= 50, f"v{vid} csf_lymphocyte_pct={pct} (must be lymphocytic)"


def test_viral_wave1_csf_neutrophil_low():
    """Viral CSF must NOT be neutrophilic (<50 percent)."""
    for vid in VIRAL_WAVE1_IDS:
        pct = _load(vid)["csf"]["csf_neutrophil_pct"]
        assert pct < 50, f"v{vid} csf_neutrophil_pct={pct} (must NOT be neutrophilic)"


def test_viral_wave1_csf_glucose_normal_or_near():
    """Viral CSF: glucose typically normal or slightly low (>=40)."""
    for vid in VIRAL_WAVE1_IDS:
        glucose = _load(vid)["csf"]["csf_glucose_mg_per_dL"]
        assert glucose >= 40, f"v{vid} csf_glucose={glucose} (viral typically >=40)"


def test_viral_wave1_pre_adjudication_hold():
    for vid in VIRAL_WAVE1_IDS:
        data = _load(vid)
        assert (
            data["adjudication"]["inclusion_decision"] == "hold_for_revision"
        ), f"v{vid}"
        assert (
            "self_review_disposition=hold_for_revision"
            in data["adjudication"]["anchoring_documentation"]
        ), f"v{vid}"


def test_viral_wave1_no_em_dashes():
    """No em-dashes (\\u2014) or en-dashes (\\u2013)."""
    for vid in VIRAL_WAVE1_IDS:
        text = _wave1_json_path(vid).read_text(encoding="utf-8")
        assert chr(0x2014) not in text, f"v{vid} contains em-dash"
        assert chr(0x2013) not in text, f"v{vid} contains en-dash"


def test_viral_wave1_no_ai_tells():
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
    for vid in VIRAL_WAVE1_IDS:
        text = _wave1_json_path(vid).read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, (
                f"v{vid} contains banned AI-tell: {word!r}"
            )


# ----------------------------------------------------------------------
# Pathogen authenticity tests
# ----------------------------------------------------------------------


def test_viral_wave1_hsv1_temporal_lobe_focal():
    """HSV1 temporal lobe encephalitis: at least 50 percent of HSV1 cases
    have focal_neurological_deficit=True or cranial_nerve_palsy != none."""
    hsv1_ids = [
        vid
        for vid in VIRAL_WAVE1_IDS
        if _wave1_slot(vid).get("pathogen") == "HSV1"
    ]
    assert hsv1_ids, "expected at least one HSV1 slot in Tyler set"
    focal = []
    for vid in hsv1_ids:
        exam = _load(vid)["exam"]
        if exam["focal_neurological_deficit"] or exam["cranial_nerve_palsy"] != "none":
            focal.append(vid)
    assert len(focal) >= len(hsv1_ids) // 2 + (len(hsv1_ids) % 2), (
        f"HSV1 temporal lobe authenticity: {focal}/{hsv1_ids} "
        f"(target >=50 percent with focal/CN signs)"
    )


def test_viral_wave1_dengue_thrombocytopenia():
    """Dengue cases must demonstrate platelets <150,000 (WHO 2009 threshold)."""
    dengue_ids = [
        vid
        for vid in VIRAL_WAVE1_IDS
        if _wave1_slot(vid).get("pathogen") == "dengue"
    ]
    assert dengue_ids, "expected at least one dengue slot in Tyler set"
    for vid in dengue_ids:
        platelets = _load(vid)["labs"]["platelets_per_uL"]
        assert platelets < 150000, (
            f"v{vid} dengue platelets={platelets} (must be <150,000)"
        )


def test_viral_wave1_eee_severe_outcome():
    """EEE per Tyler review: severe outcome (fatal or severe sequelae)."""
    eee_ids = [
        vid
        for vid in VIRAL_WAVE1_IDS
        if _wave1_slot(vid).get("pathogen") == "EEE"
    ]
    assert eee_ids, "expected at least one EEE slot in Tyler set"
    for vid in eee_ids:
        spec = _wave1_slot(vid)
        outcome = spec.get("outcome", "")
        assert outcome in ("fatal", "severe_sequelae"), (
            f"v{vid} EEE outcome={outcome!r} (Tyler review epidemiology: "
            f"fatal/severe_sequelae expected)"
        )


def test_viral_wave1_pathogen_distribution():
    """3 HSV1 + 5 enterovirus + 2 HSV2 + 2 dengue + 1 EEE."""
    counts: dict[str, int] = {}
    for vid in VIRAL_WAVE1_IDS:
        p = _wave1_slot(vid)["pathogen"]
        counts[p] = counts.get(p, 0) + 1
    assert counts == {
        "HSV1": 3,
        "enterovirus": 5,
        "HSV2": 2,
        "dengue": 2,
        "EEE": 1,
    }, f"Wave 5.3.5 pathogen distribution: {counts}"


def test_viral_wave1_ambiguity_count():
    """Exactly 2 of 13 (v113 HSV2 first-episode, v117 dengue arbo overlap)."""
    ambiguous = []
    for vid in VIRAL_WAVE1_IDS:
        rationale = (
            _load(vid)["provenance"].get("inclusion_decision_rationale") or ""
        ).lower()
        if "diagnostic_ambiguity=true" in rationale or "type=" in rationale:
            ambiguous.append(vid)
    assert set(ambiguous) == VIRAL_WAVE1_AMBIGUITY_IDS, (
        f"Expected {VIRAL_WAVE1_AMBIGUITY_IDS}, got {set(ambiguous)}"
    )


def test_viral_wave1_peru_anchors():
    """Exactly 3 of 13 Peru-anchored (v99 Lima HSV1, v117 Lima dengue, v119 Tumbes dengue)."""
    peru = [
        vid
        for vid in VIRAL_WAVE1_IDS
        if _load(vid)["demographics"]["geography_region"].startswith("peru")
    ]
    assert set(peru) == VIRAL_WAVE1_PERU_IDS, (
        f"Expected {VIRAL_WAVE1_PERU_IDS}, got {set(peru)}"
    )
