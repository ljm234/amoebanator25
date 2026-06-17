"""Stage J lock-in: bact_064 (Davalos 2016, PMID 27831604) csf_culture is nulled.

The csf_culture sens/spec was 80/100 cited to Davalos 2016 RPMESP -- a pediatric
pneumococcal-meningitis epidemiology/outcomes cohort, NOT a diagnostic-accuracy
study; 80/100 was the templated bacterial-culture default (the same value nulled
for gn_pseudomonas / hib in Stage H), not a Davalos-measured figure. bact_064 is
a frozen pilot JSON (no builder/writer regenerates it), so this pins the
committed JSON directly and guards that no generator literal can restore 80/100.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACT_064 = (
    _REPO_ROOT
    / "data/vignettes/v2/class_02_bacterial/bact_064_sp_lima_pediatric.json"
)


def test_davalos_csf_culture_nulled_in_bact_064():
    obj = json.load(io.open(_BACT_064, encoding="utf-8"))
    results = obj["diagnostic_tests"]["results"]
    culture = [t for t in results if t["test_name"] == "csf_culture"]
    assert len(culture) == 1, f"expected 1 csf_culture, got {len(culture)}"
    t = culture[0]
    assert t["citation_pmid_or_doi"] == "PMID:27831604"
    assert t["sensitivity_pct"] is None and t["specificity_pct"] is None, (
        f"Davalos csf_culture must be None/None, got "
        f"{t['sensitivity_pct']}/{t['specificity_pct']}"
    )


def test_no_culture_80_literal_in_generator():
    # bact_064 is a frozen pilot JSON; ensure no builder JSON-literal could re-emit 80/100.
    src = (_REPO_ROOT / "scripts/vignettes/generate_pam_vignettes.py").read_text(encoding="utf-8")
    assert '"sensitivity_pct": 80.0' not in src
