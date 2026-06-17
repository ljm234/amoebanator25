"""Subphase 1.3 Commit 5.3.3 Wave 1 BACT lock-in tests (TDD).

14 vignettes: vignette_id 65, 66, 67, 69, 71, 73, 75, 76, 77, 78, 79, 80, 81, 90.
10 Tunkel anchor (PMID 15494903) + 4 van de Beek anchor (PMID 15509818).
13 SP + 1 GN. 4 of 14 with diagnostic_ambiguity (v75, v76, v79, v80).

Tests are written FIRST (TDD red) and confirmed FAILING before any helper
function lands or any JSON is written. Then helpers + JSONs ship and these
tests pass (TDD green).
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
from scripts.vignettes.generate_pam_vignettes import BACTERIAL_DISTRIBUTION  # noqa: E402


BACT_WAVE1_IDS = [65, 66, 67, 69, 71, 73, 75, 76, 77, 78, 79, 80, 81, 90]
AMBIGUITY_IDS = {75, 76, 79, 80}
TUNKEL_IDS = {65, 67, 69, 71, 75, 76, 78, 79, 80, 90}
VANDEBEEK_IDS = {66, 73, 77, 81}
WAVE1_DIR = _REPO_ROOT / "data" / "vignettes" / "v2" / "class_02_bacterial"


def _wave1_slot(vid: int) -> dict:
    return next(s for s in BACTERIAL_DISTRIBUTION if s["vignette_id"] == vid)


def _wave1_json_path(vid: int) -> Path:
    matches = list(WAVE1_DIR.glob(f"bact_{vid:03d}_*.json"))
    assert len(matches) == 1, (
        f"v{vid}: expected exactly 1 JSON match in {WAVE1_DIR}, got {matches!r}"
    )
    return matches[0]


def _load(vid: int) -> dict:
    return json.loads(_wave1_json_path(vid).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Tests 1-4: parametrized over 14 vignettes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", BACT_WAVE1_IDS)
def test_bact_wave1_files_exist(vid):
    matches = list(WAVE1_DIR.glob(f"bact_{vid:03d}_*.json"))
    assert len(matches) == 1, (
        f"v{vid} JSON missing or duplicate at {WAVE1_DIR}: {matches}"
    )


@pytest.mark.parametrize("vid", BACT_WAVE1_IDS)
def test_bact_wave1_schema_validates(vid):
    VignetteSchema.model_validate(_load(vid))


@pytest.mark.parametrize("vid", BACT_WAVE1_IDS)
def test_bact_wave1_demographics_match_spec(vid):
    data = _load(vid)
    spec = _wave1_slot(vid)
    assert data["demographics"]["age_years"] == spec["age_years"], (
        f"v{vid} age mismatch"
    )
    assert data["demographics"]["sex"] == spec["sex"], f"v{vid} sex mismatch"
    assert (
        data["demographics"]["geography_region"] == spec["geography_region"]
    ), f"v{vid} region mismatch"


@pytest.mark.parametrize("vid", BACT_WAVE1_IDS)
def test_bact_wave1_anchor_pmid_matches(vid):
    data = _load(vid)
    spec = _wave1_slot(vid)
    assert data["literature_anchors"][0]["pmid"] == spec["pmid"], (
        f"v{vid} anchor PMID mismatch"
    )


# ----------------------------------------------------------------------
# Tests 5-10: per-corpus invariants
# ----------------------------------------------------------------------


def test_bact_wave1_freshwater_false():
    """Spec 1.3.10 sanity: zero freshwater exposure for Class 2."""
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        assert (
            data["exposure"]["freshwater_exposure_within_14d"] is False
        ), f"v{vid} freshwater not False"


def test_bact_wave1_class_id_2():
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        assert data["ground_truth_class"] == 2, f"v{vid} not class 2"


def test_bact_wave1_csf_neutrophilic():
    """Spec 1.3.3: CSF neutrophilic (>=50 percent)."""
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        pct = data["csf"]["csf_neutrophil_pct"]
        assert pct >= 50, f"v{vid} csf_neutrophil_pct={pct}"


def test_bact_wave1_csf_glucose_low():
    """Spec 1.3.3: CSF glucose <40."""
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        glucose = data["csf"]["csf_glucose_mg_per_dL"]
        assert glucose <= 40, f"v{vid} csf_glucose_mg_per_dL={glucose}"


def test_bact_wave1_csf_protein_high():
    """Spec 1.3.3: CSF protein >100."""
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        protein = data["csf"]["csf_protein_mg_per_dL"]
        assert protein >= 100, f"v{vid} csf_protein_mg_per_dL={protein}"


def test_bact_wave1_pre_adjudication_hold():
    """Q7 5.3.1 lock: hold_for_revision verbatim."""
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        assert (
            data["adjudication"]["inclusion_decision"] == "hold_for_revision"
        ), f"v{vid} inclusion_decision not hold_for_revision"
        assert (
            "self_review_disposition=hold_for_revision"
            in data["adjudication"]["anchoring_documentation"]
        ), f"v{vid} anchoring_documentation missing verbatim hold_for_revision"


# ----------------------------------------------------------------------
# Tests 11-13: corpus-level
# ----------------------------------------------------------------------


def test_bact_wave1_ambiguity_count():
    """Exactly 4 of 14 (v75, v76, v79, v80) carry ambiguity markers in rationale."""
    ambiguous = []
    for vid in BACT_WAVE1_IDS:
        data = _load(vid)
        rat = (data["provenance"].get("inclusion_decision_rationale") or "").lower()
        if (
            "diagnostic_ambiguity=true" in rat
            or "type=partial_antibiotic" in rat
        ):
            ambiguous.append(vid)
    assert set(ambiguous) == AMBIGUITY_IDS, (
        f"Expected ambiguity IDs {AMBIGUITY_IDS}, got {set(ambiguous)}"
    )


def test_bact_wave1_no_em_dashes():
    """No em-dashes (\\u2014) or en-dashes (\\u2013) in any Wave 1 JSON."""
    for vid in BACT_WAVE1_IDS:
        text = _wave1_json_path(vid).read_text(encoding="utf-8")
        assert chr(0x2014) not in text, f"v{vid} contains em-dash"
        assert chr(0x2013) not in text, f"v{vid} contains en-dash"


def test_bact_wave1_no_ai_tells():
    """No banned AI-tell vocabulary in any Wave 1 JSON."""
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
    for vid in BACT_WAVE1_IDS:
        text = _wave1_json_path(vid).read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, (
                f"v{vid} contains banned AI-tell: {word!r}"
            )
