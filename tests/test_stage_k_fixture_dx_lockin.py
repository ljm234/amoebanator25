"""Stage K lock-in: schema-fixture diagnostic sens/spec are nulled/verified.

The 8 subphase-1.1 schema validation fixtures (tests/schemas/fixtures/valid_*_fixture.json)
carried diagnostic sensitivity/specificity attributed to real PMIDs that are NOT
diagnostic-accuracy studies (case reports, cohorts, surveillance, treatment RCTs),
so those numbers were templated/misattributed. Stage K nulls all of them except the
single verified figure: van de Beek 2004 (PMID 15509818) CSF Gram stain at 80/97
(the same figure pinned by the Stage H corpus lock-in). valid_pam_fixture.json is a
hand-authored orphan (no builder generates it); the other 7 are emitted by
scripts/vignettes/generate_subphase11_fixtures.py. This pins both the committed JSON and the
builder dicts so a future regen or manual edit cannot silently restore the numbers.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FIX = _REPO_ROOT / "tests" / "schemas" / "fixtures"

# fixture -> {test_name: (sensitivity_pct, specificity_pct)}; None means null.
_EXPECTED = {
    "valid_bacterial_fixture.json": {
        "CSF Gram stain": (80.0, 97.0),
        "CSF culture": (None, None),
    },
    "valid_viral_fixture.json": {"CSF HSV-1 PCR": (None, None)},
    "valid_tbm_fixture.json": {
        "Xpert MTB/RIF Ultra (CSF)": (None, None),
        "CSF AFB smear": (None, None),
    },
    "valid_cryptococcal_fixture.json": {
        "CSF cryptococcal antigen LFA (titer)": (None, None),
        "CSF India ink": (None, None),
    },
    "valid_ncc_fixture.json": {"Serum EITB cysticercosis": (None, None)},
    "valid_cerebral_malaria_fixture.json": {
        "Peripheral blood thick smear (Giemsa)": (None, None),
        "Plasmodium RDT (HRP-2 / pLDH)": (None, None),
    },
    "valid_nmdar_fixture.json": {
        "CSF anti-NMDA receptor antibodies (cell-based assay)": (None, None),
    },
    "valid_pam_fixture.json": {"CSF PCR Naegleria fowleri": (None, None)},
}


def _results(path):
    obj = json.load(io.open(path, encoding="utf-8"))
    return obj["diagnostic_tests"]["results"]


def test_stage_k_committed_json_fixtures():
    for fn, expected in _EXPECTED.items():
        by_name = {t["test_name"]: t for t in _results(_FIX / fn)}
        for tname, (s, sp) in expected.items():
            assert tname in by_name, f"{fn}: missing test {tname!r}"
            t = by_name[tname]
            assert (t["sensitivity_pct"], t["specificity_pct"]) == (s, sp), (
                f"{fn} {tname!r}: got "
                f"{t['sensitivity_pct']}/{t['specificity_pct']}, expected {s}/{sp}"
            )


def test_stage_k_gae_fixture_all_null():
    for t in _results(_FIX / "valid_gae_fixture.json"):
        assert t["sensitivity_pct"] is None and t["specificity_pct"] is None, (
            f"gae {t['test_name']!r}: {t['sensitivity_pct']}/{t['specificity_pct']}"
        )


def test_stage_k_builder_dicts_in_sync():
    import scripts.vignettes.generate_subphase11_fixtures as g

    builder_backed = {fn: data for fn, data in g.FIXTURES}
    for fn, expected in _EXPECTED.items():
        if fn == "valid_pam_fixture.json":
            continue  # orphan: no builder emits it
        assert fn in builder_backed, f"{fn} not in builder FIXTURES"
        by_name = {
            t["test_name"]: t
            for t in builder_backed[fn]["diagnostic_tests"]["results"]
        }
        for tname, (s, sp) in expected.items():
            assert tname in by_name, f"builder {fn}: missing {tname!r}"
            t = by_name[tname]
            assert (t["sensitivity_pct"], t["specificity_pct"]) == (s, sp), (
                f"builder {fn} {tname!r}: got "
                f"{t['sensitivity_pct']}/{t['specificity_pct']}, expected {s}/{sp}"
            )
