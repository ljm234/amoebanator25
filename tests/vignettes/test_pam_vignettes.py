"""Tests for scripts/vignettes/generate_pam_vignettes.py (Subphase 1.2 Day 1).

Validates the 20-vignette PAM corpus end-to-end: schema conformance,
PMID metadata completeness, distribution against the spec, and content
quality (no em-dashes, no AI-tells, Spanish accent integrity).

The DAY1_DISTRIBUTION list and PMID_REGISTRY in
``scripts/vignettes/generate_pam_vignettes.py`` are the source of truth for these
tests. Where the Day 1 spec doc and the actual distribution disagree on
demographic tallies (the spec was drafted before final per-vignette
sex/age assignments), tests assert against the data and call out the
delta inline.
"""
from __future__ import annotations

import re
from typing import Any

import pytest

from ml.schemas.vignette import VignetteSchema
from scripts.vignettes.generate_pam_vignettes import PMID_REGISTRY


pytestmark = pytest.mark.subphase_1_2


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _walk_strings(node: Any):
    """Yield every str leaf in a nested dict/list structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_strings(item)


# ----------------------------------------------------------------------
# 1. Schema validation across all 20 vignettes
# ----------------------------------------------------------------------


def test_all_20_vignettes_load_valid_schema(generated_vignettes):
    assert len(generated_vignettes) == 20
    for vignette in generated_vignettes:
        VignetteSchema.model_validate(vignette)


# ----------------------------------------------------------------------
# 2. PMID_REGISTRY metadata completeness
# ----------------------------------------------------------------------


_REQUIRED_PMID_KEYS = {
    "pmid", "doi", "journal", "journal_short_code", "year", "volume",
    "issue", "pages", "authors_short", "authors_full", "anchor_type",
    "verification_confidence", "last_verified_date",
}
_PMID_DIGIT_RE = re.compile(r"^\d{7,8}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Acceptable verification dates: Day 1 sweep + Day 2 corrections sweep + Day 2
# bonus canonization. See docs/PMID_CORRECTIONS_2026-05-04.md and
# docs/PMID_DAY2_BONUS_CANONIZATION_2026-05-05.md for the audit trails.
_VALID_VERIFICATION_DATES = {
    "2026-05-03", "2026-05-04", "2026-05-05",
    # Subphase 1.3 commit 5.3.1 added 10 Bacterial + Viral consensus anchors
    # (van de Beek, Tunkel, Bijlsma, Mylonakis, Heckenberg, Soeters, Whitley,
    # Tunkel-encephalitis, Granerod, Tyler) at verification_confidence=0.85
    # pending PubMed UI direct fetch in 5.3.2-5.3.4.
    "2026-05-06",
    # Subphase 1.3 commit 5.3.2 added 5 new pilot anchors via the assistant web
    # research v4 PubMed UI verification (Davalos 2016 Lima cohort,
    # Heckenberg 2008 18626301 errata fix from 18626302, Whitley 2006
    # Antiviral Res HSE adult review, Michos 2007 PLoS One enterovirus
    # PMN-predominant cohort, Munayco 2024 MMWR Peru dengue outbreak).
    "2026-05-07",
    # Subphase 1.4 commit 5.4.0 added 18 Class 4 (TBM) + Class 5
    # (Cryptococcal) + Class 6 (GAE) anchor PMIDs via the assistant web PubMed UI
    # verification v5 (Thwaites NEJM 2004, Marais Lancet ID 2010, van Toorn
    # Semin Pediatr Neurol 2014, Heemskerk NEJM 2016, Navarro-Flores J
    # Neurol 2022; Perfect CID 2010, Park AIDS 2009,
    # Singh JID 2007, Boulware NEJM 2014, Jarvis NEJM 2022 AMBITION-cm,
    # Datta EID 2009; Alvarez/Bravo JAAD Int
    # 2022, Cabello-Vilchez Neuropathology 2020, Visvesvara FEMS 2007,
    # Cope CID 2019, Damhorst Lancet ID 2022).
    "2026-05-11",
    # Errata 5.4.3.2 (2026-05-30): dissolved Frankenstein PMID 32935747;
    # the 5 NM/Hib slots (v83-v87) re-anchored to 4 real papers verified
    # via the assistant web PubMed/PMC: MacNeil 2018 CID 29126310, Marcus 2022
    # OFID 35493127, Park 2022 JOGH 35265327, Soeters 2018 CID 29509834.
    "2026-05-30",
    # Errata 5.4.3.3 (2026-05-31): deleted both Mylonakis 2002 Listeria
    # vignettes (v88/v89; full-text verification standard not met) and
    # deep-verified 3 BACT anchors via web PubMed full-text -- Tunkel 2004
    # CID 15494903, Bijlsma 2016 Lancet ID 26652862, Heckenberg 2008
    # Medicine 18626301. (van de Beek 15509818 was deep-verified 2026-05-30.)
    "2026-05-31",
    # 2026-06 verification campaign sweep dates
    "2026-06-06", "2026-06-07", "2026-06-08",
    "2026-06-09", "2026-06-10", "2026-06-11",
}


# Subphase 1.4: non-numeric registry keys (none at present) would be tested
# by the dedicated lock-in suite in
# tests/test_subphase_1_4_pmid_registry_lockin.py rather than here, since
# this completeness test enforces a 7-8 digit pmid field via _PMID_DIGIT_RE.
def _numeric_pmid_keys() -> list[str]:
    return sorted(k for k in PMID_REGISTRY.keys() if k.isdigit())


@pytest.mark.parametrize("pmid", _numeric_pmid_keys())
def test_pmid_metadata_completeness(pmid, pmid_registry):
    meta = pmid_registry[pmid]
    missing = _REQUIRED_PMID_KEYS - set(meta.keys())
    assert not missing, f"PMID {pmid} missing keys: {missing}"
    assert _PMID_DIGIT_RE.match(meta["pmid"]), \
        f"PMID {pmid} has malformed pmid string: {meta['pmid']!r}"
    assert meta["pmid"] == pmid, \
        f"PMID {pmid} self-reference mismatch: {meta['pmid']!r}"
    assert meta["journal_short_code"], \
        f"PMID {pmid} has empty journal_short_code"
    assert isinstance(meta["year"], int) and 1990 <= meta["year"] <= 2030, \
        f"PMID {pmid} year out of range: {meta['year']!r}"
    assert _DATE_RE.match(meta["last_verified_date"]), \
        f"PMID {pmid} last_verified_date not YYYY-MM-DD: " \
        f"{meta['last_verified_date']!r}"
    assert meta["last_verified_date"] in _VALID_VERIFICATION_DATES, (
        f"PMID {pmid} last_verified_date {meta['last_verified_date']!r} "
        f"not in approved verification sweep dates "
        f"{sorted(_VALID_VERIFICATION_DATES)}"
    )
    assert meta["verification_confidence"], \
        f"PMID {pmid} verification_confidence is empty"


# ----------------------------------------------------------------------
# 3. Cluster distribution
# ----------------------------------------------------------------------


_EXPECTED_CLUSTERS: dict[str, set[int]] = {
    "splash_pad": {1, 2, 3, 4},
    "lake_pond": {5, 6, 7, 8, 9},
    "nasal_irrigation": {10, 11, 12},
    "hot_springs": {13, 14},
    "pakistan_ablution": {15, 16},
    "latam": {17, 18},
    "survivor_adult": {19},
    "survivor_pediatric": {20},
}


def test_cluster_distribution_matches_spec(distribution):
    actual: dict[str, set[int]] = {}
    for spec in distribution:
        actual.setdefault(spec["cluster"], set()).add(spec["vignette_id"])
    assert actual == _EXPECTED_CLUSTERS


# ----------------------------------------------------------------------
# 4. Demographic distribution
# ----------------------------------------------------------------------
#
# Spec doc (amoebanator_subphase_1_2_day1_distribution.md) listed
# Female=9 / Male=11 / Pediatric=13 / Adult=7. The final per-vignette
# table in the same doc resolves to Female={2,5,11,12,14}=5 and
# Adult={10,11,12,14,16,19}=6. The data is the source of truth here;
# the summary block in the spec was drafted earlier and is stale.


_FATAL_IDS = set(range(1, 19))
_SURVIVOR_IDS = {19, 20}
_FEMALE_IDS = {2, 5, 11, 12, 14}
_MALE_IDS = set(range(1, 21)) - _FEMALE_IDS
_ADULT_IDS = {10, 11, 12, 14, 16, 19}
_PEDIATRIC_IDS = set(range(1, 21)) - _ADULT_IDS


def test_demographic_distribution_matches_spec(distribution):
    by_id = {s["vignette_id"]: s for s in distribution}
    assert {i for i, s in by_id.items() if s["outcome"] == "fatal"} == _FATAL_IDS
    assert {i for i, s in by_id.items() if s["outcome"] == "survived"} == _SURVIVOR_IDS
    assert {i for i, s in by_id.items() if s["sex"] == "female"} == _FEMALE_IDS
    assert {i for i, s in by_id.items() if s["sex"] == "male"} == _MALE_IDS
    assert {i for i, s in by_id.items() if s["age_years"] >= 18} == _ADULT_IDS
    assert {i for i, s in by_id.items() if s["age_years"] < 18} == _PEDIATRIC_IDS
    # Sanity: counts add to 20
    assert len(_FATAL_IDS) + len(_SURVIVOR_IDS) == 20
    assert len(_FEMALE_IDS) + len(_MALE_IDS) == 20
    assert len(_ADULT_IDS) + len(_PEDIATRIC_IDS) == 20


# ----------------------------------------------------------------------
# 5. No em-dashes (or en-dashes) in generated content
# ----------------------------------------------------------------------


def test_no_em_dashes_in_content(generated_vignettes):
    em = 0
    en = 0
    for vignette in generated_vignettes:
        for s in _walk_strings(vignette):
            em += s.count(chr(0x2014))  # em-dash
            en += s.count(chr(0x2013))  # en-dash
    assert em == 0, f"Found {em} em-dash(es) in generated content"
    assert en == 0, f"Found {en} en-dash(es) in generated content"


# ----------------------------------------------------------------------
# 6. No AI-tell vocabulary in generated content
# ----------------------------------------------------------------------


_AI_TELLS = (
    "leverage", "harness", "delve", "seamless", "comprehensive",
    "exceptional", "robust", "showcase", "elevate", "empower",
    "tapestry", "unleash",
)


def test_no_ai_tells_in_content(generated_vignettes):
    hits: dict[str, int] = {}
    for vignette in generated_vignettes:
        for s in _walk_strings(vignette):
            lower = s.lower()
            for token in _AI_TELLS:
                if token in lower:
                    hits[token] = hits.get(token, 0) + 1
    assert not hits, f"AI-tell tokens found in generated content: {hits}"


# ----------------------------------------------------------------------
# 7. Spanish narratives have proper UTF-8 accents
# ----------------------------------------------------------------------


_SPANISH_ACCENT_CHARS = set("áéíóúñÁÉÍÓÚÑ")
# Tokens universal across the 20 narratives (verified empirically).
# "presentó" and "ingresó" both appear but are mutually exclusive per
# vignette: cases that present comatose use "ingresó en coma" instead
# of "presentó". "años" is absent from the 16-month-old infant case.
# These five tokens cover CSF/imaging language present in every case.
_REQUIRED_SPANISH_TOKENS = (
    "líquido", "presión", "cefalorraquídeo", "días", "mostró",
)


def test_spanish_narratives_have_proper_accents(generated_vignettes):
    for vignette in generated_vignettes:
        case_id = vignette["case_id"]
        narrative_es = vignette["narrative_es"]
        accent_chars = _SPANISH_ACCENT_CHARS & set(narrative_es)
        assert accent_chars, (
            f"{case_id} narrative_es contains no UTF-8 Spanish accents"
        )
        for token in _REQUIRED_SPANISH_TOKENS:
            assert token in narrative_es, (
                f"{case_id} narrative_es missing accented token "
                f"{token!r} (likely an unaccented spelling slipped in)"
            )


# ----------------------------------------------------------------------
# 8. Survivor vs fatal outcome consistency
# ----------------------------------------------------------------------


def test_survivor_vignettes_have_correct_outcome(generated_vignettes):
    by_id = {v["case_id"].split("-")[2]: v for v in generated_vignettes}
    # Survivors: 19, 20
    for vid_str in ("019", "020"):
        v = by_id[vid_str]
        anchoring = v["adjudication"]["anchoring_documentation"].lower()
        assert "outcome=survived" in anchoring, (
            f"vignette {vid_str} adjudication missing outcome=survived "
            f"({anchoring[:120]}...)"
        )
        narrative_en = v["narrative_en"].lower()
        assert "died" not in narrative_en, (
            f"vignette {vid_str} survivor narrative_en contains 'died'"
        )
        assert "survivor" in narrative_en or "discharged" in narrative_en, (
            f"vignette {vid_str} narrative_en lacks survivor/discharged "
            f"language"
        )
        narrative_es = v["narrative_es"].lower()
        assert (
            "sobreviviente" in narrative_es
            or "egresado" in narrative_es
            or "egresada" in narrative_es
        ), (
            f"vignette {vid_str} narrative_es lacks "
            f"sobreviviente/egresado language"
        )
    # Fatal: 1-18
    for i in range(1, 19):
        vid_str = f"{i:03d}"
        v = by_id[vid_str]
        anchoring = v["adjudication"]["anchoring_documentation"].lower()
        assert "outcome=fatal" in anchoring, (
            f"vignette {vid_str} adjudication missing outcome=fatal"
        )


# ----------------------------------------------------------------------
# 9. literature_anchors[0].pmid matches DAY1_DISTRIBUTION assignment
# ----------------------------------------------------------------------


def test_pmid_assignments_match_distribution(distribution, generated_vignettes):
    by_id = {s["vignette_id"]: s for s in distribution}
    for vignette in generated_vignettes:
        vignette_id = int(vignette["case_id"].split("-")[2])
        spec = by_id[vignette_id]
        anchor_pmid = vignette["literature_anchors"][0]["pmid"]
        assert anchor_pmid == spec["pmid"], (
            f"vignette {vignette_id} literature_anchor pmid "
            f"{anchor_pmid!r} != distribution pmid {spec['pmid']!r}"
        )


# ----------------------------------------------------------------------
# 10. case_id format
# ----------------------------------------------------------------------


_ALLOWED_JOURNAL_CODES = {
    "MMWR", "JCM", "CID", "IDCases", "AJTMH", "EID", "IJP",
    # Vancouver MEDLINE-style abbreviations (Day 2 canonization 2026-05-04):
    "Emerg Infect Dis", "Front Microbiol", "Front Med (Lausanne)",
    "Pathogens", "Front Pediatr", "BMC Infect Dis", "J Trop Pediatr",
    "TexMed", "JPIDS", "EpidemiolInfect", "ExpertRevAntiInfect",
    # Day-2 pilot (commit 4 of 5):
    "Diagnostics", "Yonsei Med J", "Pediatrics",
}
# Journal portion may now contain spaces and parentheses (Vancouver style).
# Use a non-greedy capture for the journal segment, terminated by `-NNNN-`
# (a 4-digit year) so the journal can include any chars except newline.
# Day prefix is D1 (v1-v20) or D2 (v21-v60).
_CASE_ID_RE = re.compile(
    r"^PAM-D[12]-(\d{3})-(.+?)-(\d{4})-(.+)$"
)


def test_case_id_format(generated_vignettes):
    seen_ids: set[str] = set()
    for vignette in generated_vignettes:
        case_id = vignette["case_id"]
        assert case_id not in seen_ids, f"duplicate case_id: {case_id}"
        seen_ids.add(case_id)
        m = _CASE_ID_RE.match(case_id)
        assert m, f"case_id {case_id!r} does not match pattern"
        nnn, journal, year, tail = m.groups()
        assert 1 <= int(nnn) <= 20, f"case_id {case_id} NNN out of range"
        assert journal in _ALLOWED_JOURNAL_CODES, (
            f"case_id {case_id} journal {journal!r} not in "
            f"{sorted(_ALLOWED_JOURNAL_CODES)}"
        )
        assert 1990 <= int(year) <= 2030, (
            f"case_id {case_id} year {year} out of range"
        )
        assert tail, f"case_id {case_id} missing region/cluster tail"


# ======================================================================
# Day 2 distribution lock (v21-v60, 40 vignettes)
# ----------------------------------------------------------------------
# Tests below validate the Day-2 distribution data structure only.
# Vignette JSON generation for v21-v60 is deferred to Commits 4-5.
# Source of truth: DAY2_DISTRIBUTION in scripts/vignettes/generate_pam_vignettes.py.
# Rationale doc: docs/DAY2_DISTRIBUTION_RATIONALE.md.
# ======================================================================


_EXPECTED_CLUSTERS_DAY2: dict[str, set[int]] = {
    "splash_pad": {23, 25, 50, 51, 52},
    "lake_pond": {22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
    "river": {21, 41, 42, 43, 44, 45, 46, 47, 48, 49},
    "nasal_irrigation": {53, 54, 55, 56, 57, 58},
    "hot_springs": {59},
    "pakistan_ablution": {60},
}

_DAY2_FILENAME_RE = re.compile(
    r"^pam_d2_(\d{3})_[a-z][a-z0-9_]*\.json$"
)

_REUSE_CAP = 6


def test_day2_distribution_length(day2_distribution):
    assert len(day2_distribution) == 40, (
        f"DAY2_DISTRIBUTION has {len(day2_distribution)} entries, expected 40"
    )


def test_day2_vignette_ids_contiguous(day2_distribution):
    ids = sorted(s["vignette_id"] for s in day2_distribution)
    assert ids == list(range(21, 61)), (
        f"Day-2 vignette_ids not contiguous 21-60: {ids}"
    )


def test_day2_cluster_distribution_matches_spec(day2_distribution):
    actual: dict[str, set[int]] = {}
    for spec in day2_distribution:
        actual.setdefault(spec["cluster"], set()).add(spec["vignette_id"])
    assert actual == _EXPECTED_CLUSTERS_DAY2, (
        f"Day-2 cluster distribution does not match spec.\n"
        f"  expected: {_EXPECTED_CLUSTERS_DAY2}\n"
        f"  actual:   {actual}"
    )


def test_day2_pmids_in_registry(day2_distribution, pmid_registry):
    for spec in day2_distribution:
        assert spec["pmid"] in pmid_registry, (
            f"Day-2 vignette {spec['vignette_id']} pmid {spec['pmid']!r} "
            f"not in PMID_REGISTRY"
        )


def test_combined_corpus_size_60(distribution, day2_distribution):
    assert len(distribution) + len(day2_distribution) == 60, (
        f"Combined corpus size = {len(distribution) + len(day2_distribution)}, "
        f"expected 60"
    )


def test_no_id_collisions(distribution, day2_distribution):
    day1_ids = {s["vignette_id"] for s in distribution}
    day2_ids = {s["vignette_id"] for s in day2_distribution}
    assert day1_ids.isdisjoint(day2_ids), (
        f"Day-1 and Day-2 vignette_ids overlap: "
        f"{sorted(day1_ids & day2_ids)}"
    )


def test_no_filename_collisions(distribution, day2_distribution):
    day1_files = {s["filename"] for s in distribution}
    day2_files = {s["filename"] for s in day2_distribution}
    assert day1_files.isdisjoint(day2_files), (
        f"Day-1 and Day-2 filenames overlap: "
        f"{sorted(day1_files & day2_files)}"
    )


def test_day2_filename_format(day2_distribution):
    for spec in day2_distribution:
        fname = spec["filename"]
        m = _DAY2_FILENAME_RE.match(fname)
        assert m, (
            f"Day-2 vignette {spec['vignette_id']} filename {fname!r} "
            f"does not match pam_d2_NNN_<tag>.json"
        )
        nnn = int(m.group(1))
        assert nnn == spec["vignette_id"], (
            f"filename {fname!r} NNN={nnn} != vignette_id {spec['vignette_id']}"
        )


def test_pmid_reuse_cap(distribution, day2_distribution):
    counts: dict[str, int] = {}
    for spec in distribution:
        counts[spec["pmid"]] = counts.get(spec["pmid"], 0) + 1
    for spec in day2_distribution:
        counts[spec["pmid"]] = counts.get(spec["pmid"], 0) + 1
    over_cap = {p: n for p, n in counts.items() if n > _REUSE_CAP}
    assert not over_cap, (
        f"PMIDs over reuse cap {_REUSE_CAP}x: {over_cap}"
    )


def test_day2_sex_enum(day2_distribution):
    for spec in day2_distribution:
        assert spec["sex"] in {"male", "female"}, (
            f"Day-2 vignette {spec['vignette_id']} has invalid sex "
            f"{spec['sex']!r}"
        )


def test_day2_outcome_enum(day2_distribution):
    for spec in day2_distribution:
        assert spec["outcome"] in {"fatal", "survived"}, (
            f"Day-2 vignette {spec['vignette_id']} has invalid outcome "
            f"{spec['outcome']!r}"
        )


def test_day2_stage_enum(day2_distribution):
    for spec in day2_distribution:
        assert spec["stage"] in {"early", "mid", "late"}, (
            f"Day-2 vignette {spec['vignette_id']} has invalid stage "
            f"{spec['stage']!r}"
        )


def test_combined_demographic_balance(distribution, day2_distribution):
    combined = list(distribution) + list(day2_distribution)
    n = len(combined)
    female = sum(1 for s in combined if s["sex"] == "female")
    adult = sum(1 for s in combined if s["age_years"] >= 18)
    assert female / n >= 0.20, (
        f"Combined female ratio {female}/{n} = {female/n:.2%} < 20% "
        f"(target 22% per locked decisions; floor 20% allowed)"
    )
    assert adult / n >= 0.25, (
        f"Combined adult ratio {adult}/{n} = {adult/n:.2%} < 25%"
    )


def test_combined_outcome_balance(distribution, day2_distribution):
    combined = list(distribution) + list(day2_distribution)
    n = len(combined)
    fatal = sum(1 for s in combined if s["outcome"] == "fatal")
    survived = sum(1 for s in combined if s["outcome"] == "survived")
    assert fatal / n >= 0.90, (
        f"Combined fatal ratio {fatal}/{n} = {fatal/n:.2%} < 90%"
    )
    assert survived / n >= 0.08, (
        f"Combined survivor ratio {survived}/{n} = {survived/n:.2%} < 8%"
    )


def test_combined_geographic_balance(distribution, day2_distribution):
    combined = list(distribution) + list(day2_distribution)
    n = len(combined)
    us_labels = {
        "Arkansas, US", "Florida, US", "Louisiana, US", "Texas, US",
        "Minnesota, US", "Nebraska, US", "California, US",
        "US South region", "Texas (Rio Grande), US",
    }
    non_us = sum(1 for s in combined if s["geography_label"] not in us_labels)
    assert non_us / n >= 0.30, (
        f"Combined non-US ratio {non_us}/{n} = {non_us/n:.2%} < 30%"
    )


def test_day2_special_cases_present(day2_distribution):
    by_pmid = {s["pmid"]: s for s in day2_distribution}
    assert "39795618" in by_pmid, "Phung 2025 cryptic-exposure anchor missing"
    assert "39606118" in by_pmid, "Lin 2024 atypical-myocarditis anchor missing"
    assert "37727924" in by_pmid, "Hong 2023 travel-imported anchor missing"
    assert "25667249" in by_pmid, "Linam 2015 Kali Hardig survivor anchor missing"


# ======================================================================
# Day 2 pilot vignette content tests (v21-v25, commit 4 of 5)
# ----------------------------------------------------------------------
# These tests validate the 5 pilot JSON files generated by Commit 4 of 5.
# Each pilot vignette is anchored 100% in primary-source data verified
# via user PubMed UI direct fetch.
# ======================================================================

import json
from pathlib import Path

_PILOT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "vignettes" / "pam"
_PILOT_IDS = [21, 22, 23, 24, 25]


@pytest.fixture(scope="session")
def pilot_vignettes(day2_distribution):
    """Load the 5 pilot JSON files from disk."""
    out: dict[int, dict[str, Any]] = {}
    by_id = {s["vignette_id"]: s for s in day2_distribution}
    for vid in _PILOT_IDS:
        spec = by_id[vid]
        fpath = _PILOT_DATA_DIR / spec["filename"]
        out[vid] = {
            "spec": spec,
            "path": fpath,
            "data": json.loads(fpath.read_text(encoding="utf-8")),
        }
    return out


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_file_exists(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    assert entry["path"].exists(), f"v{vid} pilot JSON {entry['path']} missing"


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_schema_validates(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    VignetteSchema.model_validate(entry["data"])


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_demographics_match_spec(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    spec = entry["spec"]
    demo = entry["data"]["demographics"]
    assert demo["age_years"] == spec["age_years"], (
        f"v{vid}: JSON age_years={demo['age_years']} != spec {spec['age_years']}"
    )
    assert demo["sex"] == spec["sex"], (
        f"v{vid}: JSON sex={demo['sex']!r} != spec {spec['sex']!r}"
    )
    assert demo["geography_region"] == spec["geography_region"], (
        f"v{vid}: JSON geography_region={demo['geography_region']!r} != spec "
        f"{spec['geography_region']!r}"
    )


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_anchor_pmid_matches(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    anchors = entry["data"]["literature_anchors"]
    assert anchors, f"v{vid}: empty literature_anchors"
    assert anchors[0]["pmid"] == entry["spec"]["pmid"], (
        f"v{vid}: anchor pmid {anchors[0]['pmid']!r} != "
        f"spec pmid {entry['spec']['pmid']!r}"
    )


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_narrative_min_length(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    assert len(en) >= 100, f"v{vid} narrative_en too short ({len(en)} chars)"
    assert len(es) >= 100, f"v{vid} narrative_es too short ({len(es)} chars)"


@pytest.mark.parametrize("vid", _PILOT_IDS)
def test_pilot_narrative_cites_anchor_pmid(vid, pilot_vignettes):
    entry = pilot_vignettes[vid]
    pmid = entry["spec"]["pmid"]
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    needle = f"PMID {pmid}"
    assert needle in en, f"v{vid} narrative_en missing '{needle}'"
    assert needle in es, f"v{vid} narrative_es missing '{needle}'"


def test_v25_outcome_survived(pilot_vignettes):
    v25 = pilot_vignettes[25]
    spec = v25["spec"]
    assert spec["outcome"] == "survived", (
        f"v25 spec outcome {spec['outcome']!r} expected 'survived'"
    )
    anchoring = v25["data"]["adjudication"]["anchoring_documentation"].lower()
    assert "outcome=survived" in anchoring, (
        f"v25 adjudication missing outcome=survived "
        f"(anchoring snippet: {anchoring[:120]}...)"
    )
    en = v25["data"]["narrative_en"].lower()
    assert "survived" in en, "v25 narrative_en missing 'survived'"
    assert "miltefosine" in en, "v25 narrative_en missing miltefosine reference"


def test_v23_atypical_features_in_narrative(pilot_vignettes):
    v23 = pilot_vignettes[23]
    en = v23["data"]["narrative_en"].lower()
    es = v23["data"]["narrative_es"].lower()
    for token in ("myocarditis", "ecmo", "indoor heated"):
        assert token in en, f"v23 narrative_en missing {token!r}"
    for token in ("miocarditis", "ecmo", "piscina"):
        assert token in es, f"v23 narrative_es missing {token!r}"


def test_pilot_no_em_dashes(pilot_vignettes):
    em = chr(0x2014)
    en_dash = chr(0x2013)
    for vid in _PILOT_IDS:
        content = pilot_vignettes[vid]["path"].read_text(encoding="utf-8")
        assert content.count(em) == 0, f"v{vid} contains {em} em-dash(es)"
        assert content.count(en_dash) == 0, f"v{vid} contains {en_dash} en-dash(es)"


def test_pilot_no_ai_tells(pilot_vignettes):
    banned = (
        "delve", "tapestry", "navigate the realm", "in the realm of",
        "vibrant", "robust", "comprehensive", "intricate",
    )
    for vid in _PILOT_IDS:
        content = pilot_vignettes[vid]["path"].read_text(encoding="utf-8").lower()
        for w in banned:
            assert w not in content, f"v{vid} contains AI-tell {w!r}"


# ======================================================================
# Day 2 wave 1 vignette content tests (v26-v40, commit 5.1 of 5)
# ----------------------------------------------------------------------
# These tests validate the 15 wave-1 JSON files generated by Commit
# 5.1 of 5. Each wave-1 vignette uses imputation_within_anchor_
# epidemiology per docs/DAY2_DISTRIBUTION_RATIONALE.md section 9.1.
# ======================================================================

_WAVE1_IDS = list(range(26, 41))


@pytest.fixture(scope="session")
def wave1_vignettes(day2_distribution):
    out: dict[int, dict[str, Any]] = {}
    by_id = {s["vignette_id"]: s for s in day2_distribution}
    for vid in _WAVE1_IDS:
        spec = by_id[vid]
        fpath = _PILOT_DATA_DIR / spec["filename"]
        out[vid] = {
            "spec": spec,
            "path": fpath,
            "data": json.loads(fpath.read_text(encoding="utf-8")),
        }
    return out


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_file_exists(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    assert entry["path"].exists(), f"v{vid} wave-1 JSON {entry['path']} missing"


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_schema_validates(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    VignetteSchema.model_validate(entry["data"])


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_demographics_match_spec(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    spec = entry["spec"]
    demo = entry["data"]["demographics"]
    assert demo["age_years"] == spec["age_years"]
    assert demo["sex"] == spec["sex"]
    assert demo["geography_region"] == spec["geography_region"]


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_anchor_pmid_matches(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    anchors = entry["data"]["literature_anchors"]
    assert anchors and anchors[0]["pmid"] == entry["spec"]["pmid"]


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_narrative_min_length(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    assert len(en) >= 100, f"v{vid} narrative_en too short ({len(en)} chars)"
    assert len(es) >= 100, f"v{vid} narrative_es too short ({len(es)} chars)"


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_narrative_cites_anchor_pmid(vid, wave1_vignettes):
    entry = wave1_vignettes[vid]
    pmid = entry["spec"]["pmid"]
    needle = f"PMID {pmid}"
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    assert needle in en, f"v{vid} narrative_en missing '{needle}'"
    assert needle in es, f"v{vid} narrative_es missing '{needle}'"


@pytest.mark.parametrize("vid", _WAVE1_IDS)
def test_wave1_narrative_imputation_disclosure(vid, wave1_vignettes):
    """Each wave-1 narrative must honestly disclose imputation basis."""
    entry = wave1_vignettes[vid]
    en = (entry["data"].get("narrative_en") or "").lower()
    # Imputation disclosure phrases per RULE 1 of the build spec.
    disclosures = (
        "imputation", "imputed", "within-cohort", "within the anchor",
    )
    assert any(p in en for p in disclosures), (
        f"v{vid} narrative_en missing imputation disclosure "
        f"(expected one of {disclosures})"
    )


def test_wave1_no_em_dashes(wave1_vignettes):
    em = chr(0x2014)
    en_dash = chr(0x2013)
    for vid in _WAVE1_IDS:
        content = wave1_vignettes[vid]["path"].read_text(encoding="utf-8")
        assert content.count(em) == 0, f"v{vid} contains em-dash"
        assert content.count(en_dash) == 0, f"v{vid} contains en-dash"


def test_wave1_no_ai_tells(wave1_vignettes):
    banned = (
        "delve", "tapestry", "navigate the realm", "in the realm of",
        "vibrant", "robust", "comprehensive", "intricate",
    )
    for vid in _WAVE1_IDS:
        content = wave1_vignettes[vid]["path"].read_text(encoding="utf-8").lower()
        for w in banned:
            assert w not in content, f"v{vid} contains AI-tell {w!r}"


# ======================================================================
# Day 2 wave 2 vignette content tests (v41-v60, commit 5.2 of 5)
# ----------------------------------------------------------------------
# These tests validate the 20 wave-2 JSON files generated by Commit
# 5.2 of 5 (FINAL). Wave 2 is a mix of primary-source-anchored
# newcomers (Zhou, Sazzad, Retana, DeNapoli, Wei, Cope), Day-1 PMID
# reuses (Lares-Villa, Rauf, Dulski, Eger, Yoder 2012 x2, Smith,
# Sandi, Burki - different demographics within the same anchor), and
# Tier-3/4 within-cohort imputations (Capewell river, Gharpure x3).
# ======================================================================

_WAVE2_IDS = list(range(41, 61))


@pytest.fixture(scope="session")
def wave2_vignettes(day2_distribution):
    out: dict[int, dict[str, Any]] = {}
    by_id = {s["vignette_id"]: s for s in day2_distribution}
    for vid in _WAVE2_IDS:
        spec = by_id[vid]
        fpath = _PILOT_DATA_DIR / spec["filename"]
        out[vid] = {
            "spec": spec,
            "path": fpath,
            "data": json.loads(fpath.read_text(encoding="utf-8")),
        }
    return out


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_file_exists(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    assert entry["path"].exists(), f"v{vid} wave-2 JSON {entry['path']} missing"


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_schema_validates(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    VignetteSchema.model_validate(entry["data"])


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_demographics_match_spec(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    spec = entry["spec"]
    demo = entry["data"]["demographics"]
    assert demo["age_years"] == spec["age_years"]
    assert demo["sex"] == spec["sex"]
    assert demo["geography_region"] == spec["geography_region"]


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_anchor_pmid_matches(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    anchors = entry["data"]["literature_anchors"]
    assert anchors and anchors[0]["pmid"] == entry["spec"]["pmid"]


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_narrative_min_length(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    assert len(en) >= 100, f"v{vid} narrative_en too short ({len(en)} chars)"
    assert len(es) >= 100, f"v{vid} narrative_es too short ({len(es)} chars)"


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_narrative_cites_anchor_pmid(vid, wave2_vignettes):
    entry = wave2_vignettes[vid]
    pmid = entry["spec"]["pmid"]
    needle = f"PMID {pmid}"
    en = entry["data"].get("narrative_en") or ""
    es = entry["data"].get("narrative_es") or ""
    assert needle in en, f"v{vid} narrative_en missing '{needle}'"
    assert needle in es, f"v{vid} narrative_es missing '{needle}'"


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_narrative_methodology_disclosure(vid, wave2_vignettes):
    """Each wave-2 narrative must honestly disclose its methodology.

    Acceptable disclosures: imputation/reuse phrasing for imputed and
    Day-1 reuse vignettes; "inferred from PAM-cohort epidemiology" or
    similar for primary-source-anchored newcomers where exact source
    values were not directly reported.
    """
    entry = wave2_vignettes[vid]
    en = (entry["data"].get("narrative_en") or "").lower()
    disclosures = (
        "imputation",
        "imputed",
        "within-cohort",
        "within the anchor",
        "day-1 used",
        "inferred from pam-cohort",
        "pam-cohort epidemiology",
        "case context",
    )
    assert any(p in en for p in disclosures), (
        f"v{vid} narrative_en missing methodology disclosure "
        f"(expected one of {disclosures})"
    )


def test_wave2_no_em_dashes(wave2_vignettes):
    em = chr(0x2014)
    en_dash = chr(0x2013)
    for vid in _WAVE2_IDS:
        content = wave2_vignettes[vid]["path"].read_text(encoding="utf-8")
        assert content.count(em) == 0, f"v{vid} contains em-dash"
        assert content.count(en_dash) == 0, f"v{vid} contains en-dash"


def test_wave2_no_ai_tells(wave2_vignettes):
    banned = (
        "delve", "tapestry", "navigate the realm", "in the realm of",
        "vibrant", "robust", "comprehensive", "intricate",
    )
    for vid in _WAVE2_IDS:
        content = wave2_vignettes[vid]["path"].read_text(encoding="utf-8").lower()
        for w in banned:
            assert w not in content, f"v{vid} contains AI-tell {w!r}"


def test_v49_outcome_survived(wave2_vignettes):
    v49 = wave2_vignettes[49]
    spec = v49["spec"]
    assert spec["outcome"] == "survived"
    anchoring = v49["data"]["adjudication"]["anchoring_documentation"].lower()
    assert "outcome=survived" in anchoring
    en = v49["data"]["narrative_en"].lower()
    assert "miltefosine" in en, "v49 narrative_en missing miltefosine"
    assert "survivor" in en or "discharged" in en
    es = v49["data"]["narrative_es"].lower()
    assert "miltefosina" in es, "v49 narrative_es missing miltefosina"


def test_v60_outcome_survived(wave2_vignettes):
    v60 = wave2_vignettes[60]
    spec = v60["spec"]
    assert spec["outcome"] == "survived"
    anchoring = v60["data"]["adjudication"]["anchoring_documentation"].lower()
    assert "outcome=survived" in anchoring
    en = v60["data"]["narrative_en"].lower()
    assert "miltefosine" in en, "v60 narrative_en missing miltefosine"
    assert "survivor" in en or "discharged" in en
    es = v60["data"]["narrative_es"].lower()
    assert "miltefosina" in es, "v60 narrative_es missing miltefosina"


# ======================================================================
# Day 2 wave 2 polish lock-in tests (commit 5.2.1, tag v2.2.1)
# ----------------------------------------------------------------------
# These six tests lock in the wave-2 quality dimensions that earlier
# rated Excellent (jitter heterogeneity, stage-state consistency,
# cluster-exposure mapping, bilingual narrative coverage, survivor
# treatment completeness, case_id format) so any future regression
# fails CI before merge. Combined with the schema/disclosure/em-dash/
# AI-tell tests, every wave-2 quality dimension is now test-locked.
# ======================================================================


def test_wave2_jitter_uniqueness(wave2_vignettes):
    """No two wave-2 entries share the same (CSF WBC, protein, glucose, CRP, PCT)."""
    seen: dict[tuple, int] = {}
    for vid in _WAVE2_IDS:
        d = wave2_vignettes[vid]["data"]
        tup = (
            d["csf"]["csf_wbc_per_mm3"],
            d["csf"]["csf_protein_mg_per_dL"],
            d["csf"]["csf_glucose_mg_per_dL"],
            d["labs"]["crp_mg_per_L"],
            d["labs"]["procalcitonin_ng_per_mL"],
        )
        if tup in seen:
            raise AssertionError(
                f"Wave-2 jitter collision: v{vid} and v{seen[tup]} share "
                f"identical (CSF_WBC, protein, glucose, CRP, PCT) tuple {tup}"
            )
        seen[tup] = vid


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_stage_state_consistency(vid, wave2_vignettes, day2_distribution):
    """GCS and mental_status_grade must match the spec's stage classification."""
    spec = next(s for s in day2_distribution if s["vignette_id"] == vid)
    stage = spec["stage"]
    d = wave2_vignettes[vid]["data"]
    gcs = d["vitals"]["glasgow_coma_scale"]
    ms = d["exam"]["mental_status_grade"]
    if stage == "early":
        assert gcs >= 14, f"v{vid} early stage requires GCS>=14, got {gcs}"
        assert ms == "alert", f"v{vid} early stage requires ms=alert, got {ms!r}"
    elif stage == "mid":
        # Survivors may sit at GCS 12-14 (e.g., v49=13, v60=12) and still
        # be classified mid-stage by the rapid-recognition convention.
        assert 9 <= gcs <= 14, f"v{vid} mid stage requires GCS 9-14, got {gcs}"
        assert ms in {"somnolent", "confused"}, (
            f"v{vid} mid stage requires somnolent/confused, got {ms!r}"
        )
    elif stage == "late":
        assert gcs <= 8, f"v{vid} late stage requires GCS<=8, got {gcs}"
        assert ms in {"stuporous", "comatose"}, (
            f"v{vid} late stage requires stuporous/comatose, got {ms!r}"
        )


_WAVE2_CLUSTER_EXPOSURE_MAP: dict[str, set[str]] = {
    "splash_pad": {"splash_pad"},
    "lake_pond": {"lake", "river", "swimming_pool_unchlorinated", "none"},
    "river": {"river"},
    "nasal_irrigation": {"neti_pot_tap_water"},
    "hot_springs": {"hot_spring"},
    "pakistan_ablution": {"ritual_ablution_wudu"},
}


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_cluster_exposure_mapping(vid, wave2_vignettes, day2_distribution):
    """Each cluster value must enforce a specific freshwater_exposure_type subset."""
    spec = next(s for s in day2_distribution if s["vignette_id"] == vid)
    cluster = spec["cluster"]
    expo = wave2_vignettes[vid]["data"]["exposure"]["freshwater_exposure_type"]
    allowed = _WAVE2_CLUSTER_EXPOSURE_MAP.get(cluster)
    assert allowed is not None, (
        f"v{vid} cluster {cluster!r} not in _WAVE2_CLUSTER_EXPOSURE_MAP; "
        f"update the map if new clusters were added"
    )
    assert expo in allowed, (
        f"v{vid} cluster={cluster} has freshwater_exposure_type={expo!r}, "
        f"expected one of {sorted(allowed)}"
    )


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_spanish_required_tokens(vid, wave2_vignettes):
    """Each wave-2 ES narrative carries the universal Spanish-accent token set."""
    es = wave2_vignettes[vid]["data"].get("narrative_es") or ""
    accent_chars = _SPANISH_ACCENT_CHARS & set(es)
    assert accent_chars, f"v{vid} narrative_es contains no UTF-8 Spanish accents"
    for token in _REQUIRED_SPANISH_TOKENS:
        assert token in es, (
            f"v{vid} narrative_es missing accented token {token!r}"
        )


def test_wave2_survivor_completeness(wave2_vignettes):
    """v49 and v60 survivor narratives must include miltefosine + ICP control +
    cooling protocol + ICU + discharge in both languages."""
    for vid in (49, 60):
        d = wave2_vignettes[vid]["data"]
        en = d["narrative_en"].lower()
        es = d["narrative_es"].lower()
        assert "miltefosine" in en, f"v{vid} EN missing miltefosine"
        assert "miltefosina" in es, f"v{vid} ES missing miltefosina"
        assert "intracranial pressure" in en, f"v{vid} EN missing ICP control"
        assert "presión intracraneal" in es, f"v{vid} ES missing ICP control"
        # Cooling protocol: explicit hypothermia OR targeted temperature management.
        assert ("hypothermia" in en or "temperature management" in en), (
            f"v{vid} EN missing hypothermia/targeted temperature management"
        )
        assert ("hipotermia" in es or "manejo dirigido de temperatura" in es), (
            f"v{vid} ES missing hipotermia/manejo dirigido de temperatura"
        )
        assert ("intensive care" in en or "icu" in en), (
            f"v{vid} EN missing intensive-care/ICU language"
        )
        assert "discharged" in en, f"v{vid} EN missing 'discharged'"
        assert "egresado" in es, f"v{vid} ES missing 'egresado'"
        assert "outcome=survived" in (
            d["adjudication"]["anchoring_documentation"].lower()
        ), f"v{vid} adjudication missing outcome=survived"


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_case_id_format(vid, wave2_vignettes):
    """Wave-2 case_id must follow PAM-D2-NNN-<journal_short_code>-<year>-..."""
    case_id = wave2_vignettes[vid]["data"]["case_id"]
    assert case_id.startswith(f"PAM-D2-{vid:03d}-"), (
        f"v{vid} case_id {case_id!r} does not follow PAM-D2-NNN- pattern"
    )
    pmid = wave2_vignettes[vid]["data"]["literature_anchors"][0]["pmid"]
    journal_short = PMID_REGISTRY[pmid]["journal_short_code"]
    assert journal_short in case_id, (
        f"v{vid} case_id {case_id!r} missing journal_short_code "
        f"{journal_short!r}"
    )


# ======================================================================
# Day 2 wave 2 commit 5.2.2 lock-in tests (tag v2.2.2)
# ----------------------------------------------------------------------
# Three tests added by Commit 5.2.2 in response to the forensic audit
# verdict (methodology tag NOT APPLIED, GCS variance NOT APPLIED, CSF
# WBC PARTIAL). These lock the three spec items into CI so any future
# regression in methodology classification, stage-GCS spread, or CSF
# WBC tail coverage fails before merge.
# ======================================================================

_WAVE2_VALID_METHODOLOGY_CLASSES = {
    "primary_source_direct",
    "day1_pmid_reuse",
    "tier_3_imputation",
    "tier_4_imputation",
}


@pytest.mark.parametrize("vid", _WAVE2_IDS)
def test_wave2_methodology_tag_present(vid, wave2_vignettes):
    """Each wave-2 entry must carry a methodology=<class>; prefix at the
    start of adjudication.anchoring_documentation, with class drawn from
    the four canonical methodology categories."""
    anchoring = wave2_vignettes[vid]["data"]["adjudication"][
        "anchoring_documentation"
    ]
    m = re.match(r"^methodology=([a-z_0-9]+);\s+", anchoring)
    assert m, (
        f"v{vid} adjudication missing leading 'methodology=<class>; ' prefix; "
        f"first 80 chars: {anchoring[:80]!r}"
    )
    cls = m.group(1)
    assert cls in _WAVE2_VALID_METHODOLOGY_CLASSES, (
        f"v{vid} methodology={cls!r} not in valid classes "
        f"{sorted(_WAVE2_VALID_METHODOLOGY_CLASSES)}"
    )


def test_wave2_gcs_distribution_spread(wave2_vignettes, day2_distribution):
    """Mid-stage wave-2 GCS values must span >= 5 distinct levels across
    {9, 10, 11, 12, 13}, and late-stage must span >= 5 distinct levels
    across {4, 5, 6, 7, 8}, per Commit 5.2.2 spec maximizing variance."""
    by_id = {s["vignette_id"]: s for s in day2_distribution}
    mid: list[int] = []
    late: list[int] = []
    for vid in _WAVE2_IDS:
        spec = by_id[vid]
        gcs = wave2_vignettes[vid]["data"]["vitals"]["glasgow_coma_scale"]
        if spec["stage"] == "mid":
            mid.append(gcs)
        elif spec["stage"] == "late":
            late.append(gcs)
    assert len(set(mid)) >= 5, (
        f"Mid-stage wave-2 GCS distribution has only {len(set(mid))} "
        f"distinct values: {sorted(set(mid))}; spec requires >= 5 across "
        f"{{9,10,11,12,13}}"
    )
    assert len(set(late)) >= 5, (
        f"Late-stage wave-2 GCS distribution has only {len(set(late))} "
        f"distinct values: {sorted(set(late))}; spec requires >= 5 across "
        f"{{4,5,6,7,8}}"
    )


def test_wave2_csf_wbc_range_extremes(wave2_vignettes):
    """Wave-2 CSF WBC must include >= 3 entries below 2,000 AND >= 3
    entries at or above 4,500, per Commit 5.2.2 spec maximizing the
    low- and high-tail extremes documented in PAM cohort epidemiology."""
    wbcs = [
        wave2_vignettes[vid]["data"]["csf"]["csf_wbc_per_mm3"]
        for vid in _WAVE2_IDS
    ]
    below_2000 = sum(1 for w in wbcs if w < 2000)
    at_or_above_4500 = sum(1 for w in wbcs if w >= 4500)
    assert below_2000 >= 3, (
        f"Wave-2 CSF WBC has only {below_2000} entries below 2,000; "
        f"spec requires >= 3 (extreme low tail). All values: "
        f"{sorted(wbcs)}"
    )
    assert at_or_above_4500 >= 3, (
        f"Wave-2 CSF WBC has only {at_or_above_4500} entries at or above "
        f"4,500; spec requires >= 3 (extreme high tail). All values: "
        f"{sorted(wbcs)}"
    )


# ======================================================================
# Subphase 1.3 commit 5.3.1 distribution-lock tests
# ----------------------------------------------------------------------
# These twelve tests assert structural correctness of the BACTERIAL_
# DISTRIBUTION (n=30) and VIRAL_DISTRIBUTION (n=30) lists locked in
# scripts/vignettes/generate_pam_vignettes.py against spec L1413-1414
# mandates and the marginals.json design artifacts at
# data/vignettes/v2/class_02_bacterial/marginals.json and
# data/vignettes/v2/class_03_viral/marginals.json.
#
# No vignette JSONs are generated in commit 5.3.1; runtime-encoding
# tests for the per-vignette JSONs will fire in commits 5.3.2-5.3.4.
# ======================================================================

import collections as _collections
import json as _json
from pathlib import Path as _Path
from scripts.vignettes.generate_pam_vignettes import (  # noqa: E402
    BACTERIAL_DISTRIBUTION,
    VIRAL_DISTRIBUTION,
)


_PERU_GEOGRAPHY_REGIONS = {
    "peru_lima_coast", "peru_loreto_amazon", "peru_cusco_altitude",
    "peru_puno_altitude", "peru_tumbes", "peru_madre_de_dios",
}


def test_bacterial_distribution_count():
    assert len(BACTERIAL_DISTRIBUTION) == 28


def test_viral_distribution_count():
    assert len(VIRAL_DISTRIBUTION) == 30


def test_bacterial_pathogen_counts():
    """Spec L1413: 21 SP / 4 NM / 2 Hib / 0 Listeria / 1 GN
    (2 Listeria slots removed in errata 5.4.3.3)."""
    counts = _collections.Counter(
        s["pathogen"] for s in BACTERIAL_DISTRIBUTION
    )
    assert counts["S_pneumoniae"] == 21, counts
    assert counts["N_meningitidis"] == 4, counts
    assert counts["H_influenzae"] == 2, counts
    assert counts["Listeria_monocytogenes"] == 0, counts
    assert counts["gram_negative"] == 1, counts


def test_viral_pathogen_counts():
    """Spec L1414: 12 HSV-1 / 8 enterovirus / 4 HSV-2-VZV /
    4 arboviral (3 dengue + 1 EEE) / 2 HSV-PCR-negative-at-72h."""
    counts = _collections.Counter(
        s["pathogen"] for s in VIRAL_DISTRIBUTION
    )
    assert counts["HSV1"] == 12, counts
    assert counts["enterovirus"] == 8, counts
    assert (counts["HSV2"] + counts["VZV"]) == 4, counts
    assert counts["HSV2"] == 2, counts
    assert counts["VZV"] == 2, counts
    assert counts["dengue"] == 3, counts
    assert counts["EEE"] == 1, counts
    assert counts["HSV_PCR_negative_72h"] == 2, counts


def test_bacterial_peru_anchor_share():
    """4/28 Peru-anchored: 2 Lima SP + 1 Loreto NM + 1 Cusco Hib
    (1 Tumbes Listeria removed in errata 5.4.3.3)."""
    peru = sum(
        1 for s in BACTERIAL_DISTRIBUTION
        if s["geography_region"] in _PERU_GEOGRAPHY_REGIONS
    )
    assert peru == 4


def test_viral_dengue_peru_anchor():
    """All 3 dengue cases must be Peru-anchored per spec L1408."""
    dengue_peru = sum(
        1 for s in VIRAL_DISTRIBUTION
        if s["pathogen"] == "dengue"
        and s["geography_region"] in _PERU_GEOGRAPHY_REGIONS
    )
    assert dengue_peru == 3


def test_subphase_1_3_freshwater_false():
    """Spec 1.3.10 sanity: ALL 60 Class-2/3 specs must have
    freshwater_exposure_within_14d=False."""
    for spec in BACTERIAL_DISTRIBUTION + VIRAL_DISTRIBUTION:
        assert spec["freshwater_exposure_within_14d"] is False, spec[
            "vignette_id"
        ]


def test_diagnostic_ambiguity_count():
    """Spec 1.3.5: 5 ambiguity cases per class."""
    bact_amb = sum(
        1 for s in BACTERIAL_DISTRIBUTION if s.get("diagnostic_ambiguity")
    )
    viral_amb = sum(
        1 for s in VIRAL_DISTRIBUTION if s.get("diagnostic_ambiguity")
    )
    assert bact_amb == 5, bact_amb
    assert viral_amb == 5, viral_amb


def test_hsv1_imaging_mandate_present_in_specs():
    """Each of the 12 HSV1 specs carries imaging_mandate field per master
    prompt test L1469."""
    hsv1 = [s for s in VIRAL_DISTRIBUTION if s["pathogen"] == "HSV1"]
    assert len(hsv1) == 12
    for s in hsv1:
        assert s.get("imaging_mandate") == (
            "mesial_temporal_t2_flair_hyperintensity"
        ), s["vignette_id"]


def test_dengue_platelet_mandate_present_in_specs():
    """Each of the 3 dengue specs carries platelet_mandate field per master
    prompt test L1470."""
    dengue = [s for s in VIRAL_DISTRIBUTION if s["pathogen"] == "dengue"]
    assert len(dengue) == 3
    for s in dengue:
        assert s.get("platelet_mandate_below_per_uL") == 150000, s[
            "vignette_id"
        ]


def test_subphase_1_3_vignette_ids_contiguous():
    """Class 2 occupies 61-90 minus 88/89 (the 2 Listeria slots removed in
    errata 5.4.3.3); Class 3 occupies 91-120; specs are disjoint from Day-1
    (1-20) and Day-2 (21-60)."""
    bact_ids = sorted(s["vignette_id"] for s in BACTERIAL_DISTRIBUTION)
    viral_ids = sorted(s["vignette_id"] for s in VIRAL_DISTRIBUTION)
    assert bact_ids == [i for i in range(61, 91) if i not in (88, 89)]
    assert viral_ids == list(range(91, 121))


def test_marginals_files_exist_and_valid():
    """marginals.json artifacts present per spec 1.3.1 / 1.3.2."""
    cases = [
        ("data/vignettes/v2/class_02_bacterial/marginals.json", 2, 28),
        ("data/vignettes/v2/class_03_viral/marginals.json", 3, 30),
    ]
    for path, class_id, total_n in cases:
        p = _Path(path)
        assert p.exists(), f"{path} missing"
        data = _json.loads(p.read_text(encoding="utf-8"))
        assert data["class_id"] == class_id, path
        assert data["total_n"] == total_n, path
        assert "pathogen_distribution" in data, path
        assert "csf_profile_ranges" in data, path
        assert "cited_anchors" in data, path
        assert isinstance(data["cited_anchors"], list) and len(
            data["cited_anchors"]
        ) >= 3, path
        # Pathogen distribution must match spec targets exactly.
        assert (
            data["pathogen_distribution"]
            == data["pathogen_distribution_target_per_spec"]
        ), f"{path} pathogen distribution drifted from spec"


def test_marginals_freshwater_sanity_and_adjudication_state():
    """marginals.json artifacts disclose pre-adjudication state and
    freshwater=False sanity for downstream auditors."""
    for path in (
        "data/vignettes/v2/class_02_bacterial/marginals.json",
        "data/vignettes/v2/class_03_viral/marginals.json",
    ):
        data = _json.loads(_Path(path).read_text(encoding="utf-8"))
        assert (
            data.get("freshwater_exposure_within_14d_for_all") is False
        ), path
        assert data.get("adjudication_state") == (
            "pre_adjudication_hold_for_revision"
        ), path


# ======================================================================
# Subphase 1.2.x metadata lock (commit v2.2.4-metadata-locked, 2026-05-08)
# ----------------------------------------------------------------------
# Single scoped lock-in: PMID 29462145 (Jiang YH 2018 PLoS One urology
# paper, ZERO meningitis/encephalitis/Zika content) must NOT be in
# PMID_REGISTRY. Permanent guard against the committee-hint catastrophic
# miss caught via manual PMC verification 2026-05-07. The eight author
# corrections targeting PMIDs not currently in registry are deferred
# pending ADD scope authorization. See Day 2 Corrections section in
# docs/PMID_CORRECTIONS_2026-05-04.md.
# ======================================================================


def test_pmid_29462145_excluded_from_registry():
    """Lock-in: PMID 29462145 (Jiang YH urology paper) MUST NOT be in registry.

    Verified PMID = "Videourodynamic findings of lower urinary tract
    dysfunctions in men with persistent storage lower urinary tract
    symptoms after medical treatment" (Jiang YH, Wang CC, Kuo HC. PLoS
    One 2018;13(2):e0190704). Topic is benign prostatic hyperplasia /
    bladder outlet obstruction, which has no Zika / meningitis /
    encephalitis content. Originally hinted as VIRAL_W1_07 companion
    (Mehta R Zika systematic review); manual PMC verification on
    2026-05-07 confirmed it is unrelated.
    """
    assert "29462145" not in PMID_REGISTRY, (
        "PMID 29462145 (Jiang YH urology paper) must not be present. "
        "See docs/PMID_CORRECTIONS_2026-05-04.md Day 2 Corrections."
    )


# =========================================================================
# Subphase 1.3.x errata: registry-coverage guard for BACT/VIRAL distributions
# -------------------------------------------------------------------------
# Subphase 1.2.x metadata lock + Commit 5.3.2 errata fix removed PMID
# 18626302 (typo) from PMID_REGISTRY but slot v83 in BACTERIAL_DISTRIBUTION
# still referenced it. test_day2_pmids_in_registry parametrizes over PAM
# corpus only (DAY2_DISTRIBUTION, IDs 21-60) and missed the leak. This
# lock-in extends coverage to all 60 BACT + VIRAL slots (IDs 61-120) and
# any future ADD slots in those distributions.
# =========================================================================


def test_bacterial_viral_distribution_pmids_in_registry():
    """All BACT + VIRAL slot anchor PMIDs must resolve in PMID_REGISTRY.

    Prevents the v83 18626302 typo regression and guards Subphase 1.3
    Commits 5.3.3 + 5.3.4 against introducing slots whose anchor PMID was
    typo'd or removed.
    """
    from scripts.vignettes.generate_pam_vignettes import (
        BACTERIAL_DISTRIBUTION,
        VIRAL_DISTRIBUTION,
    )
    all_slots = list(BACTERIAL_DISTRIBUTION) + list(VIRAL_DISTRIBUTION)
    broken: list[tuple[object, object]] = []
    for slot in all_slots:
        vid = slot.get("vignette_id", "?")
        pmid = slot.get("anchor_pmid") or slot.get("pmid")
        if pmid not in PMID_REGISTRY:
            broken.append((vid, pmid))
    assert not broken, (
        f"BACT/VIRAL distribution slots reference PMIDs not in registry: "
        f"{broken}. See docs/PMID_CORRECTIONS_2026-05-04.md Subphase 1.3.x "
        f"errata section."
    )
