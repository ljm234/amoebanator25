"""Subphase 1.4 Commit 5.4.2 pilot vignette lock-in tests.

6 pilot vignettes: TBM 121, 122 + CRYPTO 151, 152 + GAE 181, 182.

Anchored to the 6 PMID_REGISTRY entries verified in commit 5.4.0
(Thwaites, van Toorn, Perfect, Singh, Alvarez/Bravo, Visvesvara).

Resolution #3 applied: TBM 122 Cape Town altitude corrected from 1591m
(empirical error in proposal) to 50m (Cape Town Atlantic coastal city).
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
    CRYPTO_DISTRIBUTION,
    GAE_DISTRIBUTION,
    PMID_REGISTRY,
    TBM_DISTRIBUTION,
)


PILOT_PATHS: dict[int, Path] = {
    121: _REPO_ROOT / "data/vignettes/v2/class_04_tb/tbm_121_thwaites_hcmc_adult_pilot.json",
    122: _REPO_ROOT / "data/vignettes/v2/class_04_tb/tbm_122_vantoorn_cape_town_pediatric_pilot.json",
    151: _REPO_ROOT / "data/vignettes/v2/class_05_fungal/crypto_151_perfect_idsa_hiv_pilot.json",
    152: _REPO_ROOT / "data/vignettes/v2/class_05_fungal/crypto_152_singh_transplant_pilot.json",
    181: _REPO_ROOT / "data/vignettes/v2/class_06_gae/gae_181_alvarez_peru_balamuthia_pilot.json",
    182: _REPO_ROOT / "data/vignettes/v2/class_06_gae/gae_182_visvesvara_acanthamoeba_aids_pilot.json",
}

PILOT_IDS = sorted(PILOT_PATHS.keys())
PILOT_TO_CLASS = {121: 4, 122: 4, 151: 5, 152: 5, 181: 6, 182: 6}
PILOT_TO_CLASS_TOKEN = {121: "TBM", 122: "TBM", 151: "CRYPTO", 152: "CRYPTO",
                       181: "GAE", 182: "GAE"}


def _load(vid: int) -> dict:
    return json.loads(PILOT_PATHS[vid].read_text(encoding="utf-8"))


def _all_dists() -> list[dict]:
    return list(TBM_DISTRIBUTION) + list(CRYPTO_DISTRIBUTION) + list(GAE_DISTRIBUTION)


# ----------------------------------------------------------------------
# Existence and totals
# ----------------------------------------------------------------------


def test_subphase_1_4_pilot_count_6():
    existing = [p for p in PILOT_PATHS.values() if p.exists()]
    assert len(existing) == 6, (
        f"Expected 6 pilot JSONs, found {len(existing)}: missing="
        f"{[str(p) for p in PILOT_PATHS.values() if not p.exists()]}"
    )


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_files_exist(vid):
    assert PILOT_PATHS[vid].exists(), f"Pilot {vid} missing at {PILOT_PATHS[vid]}"


# ----------------------------------------------------------------------
# Schema compliance
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_schema_validates(vid):
    VignetteSchema.model_validate(_load(vid))


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_ground_truth_class(vid):
    assert _load(vid)["ground_truth_class"] == PILOT_TO_CLASS[vid]


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_anchor_pmid_matches_registry(vid):
    spec = next(s for s in _all_dists() if s["vignette_id"] == vid)
    expected_key = spec["anchor_pmid"]
    assert expected_key in PMID_REGISTRY, f"registry key {expected_key!r} missing"
    pmid_meta = PMID_REGISTRY[expected_key]
    anchor = _load(vid)["literature_anchors"][0]
    if pmid_meta.get("pmid"):
        assert anchor["pmid"] == pmid_meta["pmid"], (
            f"vid {vid} pmid mismatch: got {anchor.get('pmid')!r}, "
            f"expected {pmid_meta['pmid']!r}"
        )
    if pmid_meta.get("doi"):
        assert anchor["doi"] == pmid_meta["doi"], (
            f"vid {vid} doi mismatch: got {anchor.get('doi')!r}, "
            f"expected {pmid_meta['doi']!r}"
        )


# ----------------------------------------------------------------------
# Cross-class invariants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_freshwater_false_all_6(vid):
    assert _load(vid)["exposure"]["freshwater_exposure_within_14d"] is False


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_inclusion_decision_hold_for_revision(vid):
    assert _load(vid)["adjudication"]["inclusion_decision"] == "hold_for_revision"


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_adjudicator_ids_sentinel_format(vid):
    ids = _load(vid)["adjudication"]["adjudicator_ids"]
    assert len(ids) == 2, f"vid {vid} adjudicator_ids len {len(ids)}"
    pat = re.compile(r"^PILOT-(TBM|CRYPTO|GAE)-\d+-ADJ-[12]$")
    for aid in ids:
        m = pat.match(aid)
        assert m is not None, f"vid {vid} adjudicator id {aid!r} bad format"
        assert m.group(1) == PILOT_TO_CLASS_TOKEN[vid], (
            f"vid {vid} adjudicator id {aid!r} class token mismatch"
        )


# ----------------------------------------------------------------------
# Narrative quality
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_narrative_en_in_band_800_1200(vid):
    text = _load(vid).get("narrative_en") or ""
    assert 800 <= len(text) <= 1200, (
        f"vid {vid} EN narrative len {len(text)} not in [800,1200]"
    )


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_narrative_es_in_band_700_900(vid):
    text = _load(vid).get("narrative_es") or ""
    assert 700 <= len(text) <= 900, (
        f"vid {vid} ES narrative len {len(text)} not in [700,900]"
    )


def _full_text(data: dict) -> str:
    parts = [data.get("narrative_en") or "", data.get("narrative_es") or ""]
    return "\n".join(parts)


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_no_em_dashes(vid):
    text = _full_text(_load(vid))
    assert chr(0x2014) not in text, f"vid {vid} narrative contains em-dash (U+2014)"


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_no_en_dashes(vid):
    text = _full_text(_load(vid))
    assert chr(0x2013) not in text, f"vid {vid} narrative contains en-dash (U+2013)"


AI_TELLS = [
    "delve", "tapestry", "vibrant", "robust", "comprehensive",
    "intricate", "seamlessly", "leverage", "furthermore", "moreover",
    "additionally", "navigate", "in the realm of",
]


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_no_ai_tells(vid):
    text = _full_text(_load(vid)).lower()
    found = [t for t in AI_TELLS if t in text]
    assert not found, f"vid {vid} narrative contains AI-tells: {found}"


# ----------------------------------------------------------------------
# Clinical fidelity per class
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vid", PILOT_IDS)
def test_pilot_csf_differential_sums_100(vid):
    csf = _load(vid)["csf"]
    if csf["csf_wbc_per_mm3"] > 5:
        total = (
            csf["csf_neutrophil_pct"]
            + csf["csf_lymphocyte_pct"]
            + csf["csf_eosinophil_pct"]
        )
        assert 98 <= total <= 102, f"vid {vid} CSF diff sum {total} not in [98,102]"


@pytest.mark.parametrize("vid", [121, 122])
def test_tbm_pilots_basal_meningeal_imaging(vid):
    assert (
        _load(vid)["imaging"]["imaging_pattern"]
        == "basal_meningeal_enhancement_with_hydrocephalus"
    )


@pytest.mark.parametrize("vid", [151, 152])
def test_crypto_pilots_dilated_vr_imaging(vid):
    assert (
        _load(vid)["imaging"]["imaging_pattern"]
        == "dilated_virchow_robin_with_pseudocysts"
    )


@pytest.mark.parametrize("vid", [181, 182])
def test_gae_pilots_multifocal_ring_enhancing_imaging(vid):
    assert (
        _load(vid)["imaging"]["imaging_pattern"]
        == "multiple_ring_enhancing_lesions"
    )


# ----------------------------------------------------------------------
# Resolution #3
# ----------------------------------------------------------------------


def test_resolution_3_tbm_122_altitude_50m_not_1591m():
    alt = _load(122)["demographics"]["altitude_residence_m"]
    assert alt == 50, (
        f"Resolution #3: TBM 122 altitude must be 50m (Cape Town Atlantic "
        f"coastal city, sea level), got {alt}m"
    )
