"""Smoke tests for ml.schemas.vignette VignetteSchema (TDD - red phase).

These tests are EXPECTED TO FAIL initially because vignette.py doesn't exist yet.
They define the contract that the schema implementation must satisfy.

After Step C.4 (writing vignette.py), these should all pass (green phase).
"""
import json
from pathlib import Path

import pytest

# These imports will FAIL until ml/schemas/vignette.py is written
try:
    from ml.schemas import ClassLabel
    from ml.schemas.vignette import (
        VignetteSchema,
        Demographics,
        History,
        ExposureHistory,
        VitalSigns,
        PhysicalExam,
        Labs,
        CSFProfile,
        Imaging,
        DiagnosticTests,
        AdjudicationMetadata,
        LiteratureAnchor,
        Provenance,
    )
    SCHEMA_AVAILABLE = True
except ImportError as e:
    SCHEMA_AVAILABLE = False
    IMPORT_ERROR = str(e)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================================
# Test 1: Module imports
# ============================================================================
def test_schema_module_imports():
    """All schema classes must be importable from ml.schemas.vignette."""
    assert SCHEMA_AVAILABLE, f"Schema not yet implemented: {IMPORT_ERROR if not SCHEMA_AVAILABLE else ''}"


# ============================================================================
# Test 2: ClassLabel basic
# ============================================================================
def test_class_label_has_9_classes():
    """ClassLabel IntEnum must have exactly 9 members for 9-class differential."""
    assert len(list(ClassLabel)) == 9


def test_class_label_pam_is_one():
    """PAM must be class 1 per spec taxonomy."""
    assert ClassLabel.PAM == 1
    assert int(ClassLabel.PAM) == 1


def test_class_label_non_infectious_is_nine():
    """Non-infectious mimics must be class 9 (catch-all last position)."""
    assert ClassLabel.NON_INFECTIOUS_MIMIC == 9


# ============================================================================
# Test 3: Valid PAM fixture loads and validates
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_valid_pam_fixture_loads():
    """Canonical PAM fixture must validate against schema without errors."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path) as f:
        data = json.load(f)

    vignette = VignetteSchema.model_validate(data)
    assert vignette.ground_truth_class == ClassLabel.PAM
    assert vignette.exposure.freshwater_exposure_within_14d is True


# ============================================================================
# Test 4: PAM always-flag rule enforced
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_pam_class_requires_freshwater_exposure():
    """PAM ground_truth_class must require freshwater_exposure_within_14d=True
    per CDC 2017 case definition (always-flag rule)."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    with open(fixture_path) as f:
        data = json.load(f)

    # Modify to violate PAM always-flag rule
    data["exposure"]["freshwater_exposure_within_14d"] = False
    data["exposure"]["freshwater_exposure_type"] = None

    with pytest.raises(ValueError, match="freshwater_exposure_within_14d"):
        VignetteSchema.model_validate(data)


# ============================================================================
# Test 5: CSF differential sums to 100
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_csf_differential_must_sum_to_100():
    """CSF neutrophil + lymphocyte + eosinophil percentages must sum to 100
    when WBC > 5/mm3 (allows acellular CSF)."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    with open(fixture_path) as f:
        data = json.load(f)

    # Break the sum (current: 92+8+0=100, change to 92+5+0=97)
    data["csf"]["csf_lymphocyte_pct"] = 5

    with pytest.raises(ValueError, match="differential"):
        VignetteSchema.model_validate(data)


# ============================================================================
# Test 6: Literature anchor requires PMID or DOI
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_literature_anchor_requires_pmid_or_doi():
    """LiteratureAnchor must have at least one of pmid or doi."""
    with pytest.raises(ValueError, match="pmid|doi"):
        LiteratureAnchor(anchor_type="surveillance", pmid=None, doi=None)


# ============================================================================
# Test 7: Schema version is pinned to 2.0
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_schema_version_pinned_to_2_0():
    """schema_version must be Literal['2.0'] - no other values allowed."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    with open(fixture_path) as f:
        data = json.load(f)

    data["schema_version"] = "1.5"
    with pytest.raises(ValueError):
        VignetteSchema.model_validate(data)


# ============================================================================
# Test 8: Cohen's kappa range
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_cohen_kappa_must_be_in_unit_interval():
    """Cohen's kappa must be in [0.0, 1.0]."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    with open(fixture_path) as f:
        data = json.load(f)

    data["adjudication"]["cohen_kappa"] = 1.5
    with pytest.raises(ValueError):
        VignetteSchema.model_validate(data)


# ============================================================================
# Test 9: Adjudicator IDs require min 2
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_adjudication_requires_min_two_adjudicators():
    """AdjudicationMetadata.adjudicator_ids must have min_length=2 per
    Cohen 1960 / McHugh 2012 inter-rater reliability standard."""
    fixture_path = FIXTURES_DIR / "valid_pam_fixture.json"
    with open(fixture_path) as f:
        data = json.load(f)

    data["adjudication"]["adjudicator_ids"] = ["ADJ-001"]  # only 1
    with pytest.raises(ValueError):
        VignetteSchema.model_validate(data)


# ============================================================================
# Test 10: PMID format validation
# ============================================================================
@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="schema not yet implemented")
def test_pmid_format_validates():
    """PMID must match regex ^\\d{7,9}$ (7-9 digits)."""
    LiteratureAnchor(anchor_type="case_report", pmid="40146665", doi=None)
    LiteratureAnchor(anchor_type="case_report", pmid=None, doi="10.1086/425368")

    with pytest.raises(ValueError):
        LiteratureAnchor(anchor_type="case_report", pmid="abc123", doi=None)
