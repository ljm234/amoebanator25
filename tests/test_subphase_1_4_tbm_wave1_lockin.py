"""Subphase 1.4 Commit 5.4.3 TBM Wave 1 vignette lock-in tests.

14 TBM Wave 1 vignettes (vignette_id 123-136) anchored to Thwaites NEJM 2004
(5 slots), Marais Lancet ID 2010 (3 slots), and Heemskerk NEJM 2016 (6 slots).

Resolution #4 applied: TBM 125 (41yo F, 14wk pregnancy) red_flags_present
updated from [] to ['pregnancy_postpartum'] per schema enum availability.

Slots 130 (Marais 'possible') and 134 (Heemskerk young early-stage) are
Xpert MTB/RIF Ultra NEGATIVE with culture-positive confirmation;
all other 12 slots are Xpert MTB/RIF Ultra POSITIVE.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.schemas.vignette import VignetteSchema  # noqa: E402
from scripts.vignettes.generate_pam_vignettes import (  # noqa: E402
    PMID_REGISTRY,
    TBM_DISTRIBUTION,
)


_W1_DIR = _REPO_ROOT / "data/vignettes/v2/class_04_tb"

WAVE1_PATHS: dict[int, Path] = {
    123: _W1_DIR / "tbm_123_thwaites_hcmc_adult_male_cnpalsy_wave1.json",
    124: _W1_DIR / "tbm_124_thwaites_hcmc_adult_male_fatal_wave1.json",
    125: _W1_DIR / "tbm_125_thwaites_hcmc_pregnancy_female_wave1.json",
    126: _W1_DIR / "tbm_126_thwaites_hcmc_young_male_cnpalsy_wave1.json",
    127: _W1_DIR / "tbm_127_thwaites_hcmc_adult_male_smearpos_wave1.json",
    128: _W1_DIR / "tbm_128_marais_cape_town_adult_female_definite_wave1.json",
    129: _W1_DIR / "tbm_129_marais_cape_town_adult_male_probable_wave1.json",
    130: _W1_DIR / "tbm_130_marais_cape_town_adult_female_possible_wave1.json",
    131: _W1_DIR / "tbm_131_heemskerk_hcmc_adult_male_standard_wave1.json",
    132: _W1_DIR / "tbm_132_heemskerk_hcmc_adult_female_intensified_wave1.json",
    133: _W1_DIR / "tbm_133_heemskerk_hcmc_adult_male_severe_fatal_wave1.json",
    134: _W1_DIR / "tbm_134_heemskerk_hcmc_young_male_xpertneg_wave1.json",
    135: _W1_DIR / "tbm_135_heemskerk_hcmc_elderly_female_sequelae_wave1.json",
    136: _W1_DIR / "tbm_136_heemskerk_hcmc_adult_male_early_intensified_wave1.json",
}

WAVE1_IDS = sorted(WAVE1_PATHS.keys())

ANCHOR_TO_IDS: dict[str, list[int]] = {
    "15496623": [123, 124, 125, 126, 127],          # Thwaites NEJM 2004
    "20822958": [128, 129, 130],                    # Marais Lancet ID 2010
    "26760084": [131, 132, 133, 134, 135, 136],     # Heemskerk NEJM 2016
}

XPERT_NEGATIVE_IDS = {130, 134}
CN_VI_PALSY_TRUE_IDS = {123, 126, 129, 131, 133}


def _load(vid: int) -> dict:
    return json.loads(WAVE1_PATHS[vid].read_text(encoding="utf-8"))


def _spec(vid: int) -> dict:
    return next(s for s in TBM_DISTRIBUTION if s["vignette_id"] == vid)


# ----------------------------------------------------------------------
# Existence + counts
# ----------------------------------------------------------------------


def test_wave1_count_14():
    existing = [p for p in WAVE1_PATHS.values() if p.exists()]
    assert len(existing) == 14, (
        f"Expected 14 wave_1 JSONs, found {len(existing)}: missing="
        f"{[str(p) for p in WAVE1_PATHS.values() if not p.exists()]}"
    )


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_files_exist(vid):
    assert WAVE1_PATHS[vid].exists(), f"vid {vid} missing at {WAVE1_PATHS[vid]}"


# ----------------------------------------------------------------------
# Schema compliance
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_schema_validates(vid):
    VignetteSchema.model_validate(_load(vid))


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_ground_truth_class_4(vid):
    assert _load(vid)["ground_truth_class"] == 4


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_anchor_pmid_matches_registry(vid):
    spec = _spec(vid)
    expected = spec["anchor_pmid"]
    assert expected in PMID_REGISTRY, f"PMID {expected!r} not registered"
    pmid_meta = PMID_REGISTRY[expected]
    anchor = _load(vid)["literature_anchors"][0]
    assert anchor["pmid"] == pmid_meta["pmid"], (
        f"vid {vid} pmid mismatch: {anchor['pmid']!r} vs {pmid_meta['pmid']!r}"
    )


# ----------------------------------------------------------------------
# Cross-class invariants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_freshwater_false_all_14(vid):
    assert _load(vid)["exposure"]["freshwater_exposure_within_14d"] is False


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_wave_assignment_wave_1(vid):
    assert _spec(vid)["wave_assignment"] == "wave_1"


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_inclusion_decision_hold_for_revision(vid):
    assert _load(vid)["adjudication"]["inclusion_decision"] == "hold_for_revision"


_WAVE1_ADJ_PAT = re.compile(r"^WAVE1-TBM-\d+-ADJ-[12]$")


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_adjudicator_ids_wave1_format(vid):
    ids = _load(vid)["adjudication"]["adjudicator_ids"]
    assert len(ids) == 2
    for aid in ids:
        assert _WAVE1_ADJ_PAT.match(aid), f"vid {vid} bad adjudicator id {aid!r}"


# ----------------------------------------------------------------------
# Narrative quality
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_narrative_en_in_band_800_1200(vid):
    text = _load(vid).get("narrative_en") or ""
    assert 800 <= len(text) <= 1200, (
        f"vid {vid} EN narrative len {len(text)} not in [800, 1200]"
    )


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_narrative_es_in_band_700_900(vid):
    text = _load(vid).get("narrative_es") or ""
    assert 700 <= len(text) <= 900, (
        f"vid {vid} ES narrative len {len(text)} not in [700, 900]"
    )


def _full_text(data: dict) -> str:
    return "\n".join([data.get("narrative_en") or "", data.get("narrative_es") or ""])


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_no_em_dashes(vid):
    assert chr(0x2014) not in _full_text(_load(vid)), f"vid {vid} em-dash"


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_no_en_dashes(vid):
    assert chr(0x2013) not in _full_text(_load(vid)), f"vid {vid} en-dash"


AI_TELLS = [
    "delve", "tapestry", "vibrant", "robust", "comprehensive",
    "intricate", "seamlessly", "leverage", "furthermore", "moreover",
    "additionally", "navigate", "in the realm of",
]


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_no_ai_tells(vid):
    txt = _full_text(_load(vid)).lower()
    found = [t for t in AI_TELLS if t in txt]
    assert not found, f"vid {vid} AI-tells: {found}"


# ----------------------------------------------------------------------
# Clinical fidelity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_csf_differential_sums_100(vid):
    csf = _load(vid)["csf"]
    if csf["csf_wbc_per_mm3"] > 5:
        total = (csf["csf_neutrophil_pct"] + csf["csf_lymphocyte_pct"]
                 + csf["csf_eosinophil_pct"])
        assert 98 <= total <= 102, f"vid {vid} diff sum {total}"


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_basal_meningeal_imaging_all_14(vid):
    assert (
        _load(vid)["imaging"]["imaging_pattern"]
        == "basal_meningeal_enhancement_with_hydrocephalus"
    )


@pytest.mark.parametrize("vid", WAVE1_IDS)
def test_wave1_ada_at_least_10_all_14(vid):
    ada = _load(vid)["csf"].get("csf_ada_U_per_L")
    assert ada is not None and ada >= 10, f"vid {vid} ADA {ada}"


def test_wave1_xpert_positive_count_12():
    """12 of 14 Xpert MTB/RIF positive; 2 (slots 130, 134) Xpert NEG culture+."""
    pos = 0
    neg = 0
    for vid in WAVE1_IDS:
        data = _load(vid)
        xpert_test = next(
            (r for r in data["diagnostic_tests"]["results"]
             if "Xpert" in r["test_name"]), None
        )
        assert xpert_test is not None, f"vid {vid} no Xpert test"
        if xpert_test["result"].lower().startswith("positive"):
            pos += 1
        elif xpert_test["result"].lower().startswith("negative"):
            neg += 1
    assert pos == 12, f"Xpert positive count {pos}, expected 12"
    assert neg == 2, f"Xpert negative count {neg}, expected 2"


def test_wave1_cn_vi_palsy_count_5():
    """5 of 14 wave_1 slots have CN VI palsy per locked TBM_DISTRIBUTION
    (vids 123, 126, 129, 131, 133)."""
    cnvi_true = set()
    for vid in WAVE1_IDS:
        if _load(vid)["exam"]["cranial_nerve_palsy"] == "CN_VI":
            cnvi_true.add(vid)
    assert cnvi_true == CN_VI_PALSY_TRUE_IDS, (
        f"CN VI True IDs {sorted(cnvi_true)}, expected {sorted(CN_VI_PALSY_TRUE_IDS)}"
    )


# ----------------------------------------------------------------------
# Anchor distribution per cluster
# ----------------------------------------------------------------------


def test_wave1_thwaites_anchor_count_5():
    matches = [
        vid for vid in WAVE1_IDS
        if _load(vid)["literature_anchors"][0]["pmid"] == "15496623"
    ]
    assert sorted(matches) == ANCHOR_TO_IDS["15496623"], (
        f"Thwaites IDs {matches}, expected {ANCHOR_TO_IDS['15496623']}"
    )


def test_wave1_marais_anchor_count_3():
    matches = [
        vid for vid in WAVE1_IDS
        if _load(vid)["literature_anchors"][0]["pmid"] == "20822958"
    ]
    assert sorted(matches) == ANCHOR_TO_IDS["20822958"], (
        f"Marais IDs {matches}, expected {ANCHOR_TO_IDS['20822958']}"
    )


def test_wave1_heemskerk_anchor_count_6():
    matches = [
        vid for vid in WAVE1_IDS
        if _load(vid)["literature_anchors"][0]["pmid"] == "26760084"
    ]
    assert sorted(matches) == ANCHOR_TO_IDS["26760084"], (
        f"Heemskerk IDs {matches}, expected {ANCHOR_TO_IDS['26760084']}"
    )


# ----------------------------------------------------------------------
# Resolution #4
# ----------------------------------------------------------------------


def test_resolution_4_tbm_125_pregnancy_postpartum_red_flag():
    red_flags = _load(125)["history"]["red_flags_present"]
    assert "pregnancy_postpartum" in red_flags, (
        f"Resolution #4: TBM 125 must include 'pregnancy_postpartum' in "
        f"red_flags_present (got {red_flags!r})"
    )
