"""Stage H lock-in: bacterial dx-test builders must emit nulled/verified sens/spec.

After the Stage-H audit (commit-pending), the 6 bacterial dx-test builders carry
no unsupported sens/spec numbers. The single verified figure is van de Beek 2004
(PMID 15509818) CSF gram-stain at 80/97; every other bacterial dx sens/spec is
null (numbers were templated/misattributed to guideline/surveillance/case-report
anchors). This test pins that so a future builder regen can't silently restore
the old numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_bacterial_dx_builders_match_stage_h():
    from scripts.vignettes.generate_pam_vignettes import (
        _bact_wave1_dx_tests_sp_culture_positive as sp,
        _bact_wave1_dx_tests_sp_pretreated as spt,
        _bact_wave1_dx_tests_gn_pseudomonas as gn,
        _bact_wave2_dx_tests_nm_culture_positive as nm,
        _bact_wave2_dx_tests_nm_pretreated as nmp,
        _bact_wave2_dx_tests_hib as hib,
    )
    # sp_culture_positive: only van de Beek (15509818) gram-stain keeps 80/97
    for p in ("15494903", "15509818", "26652862"):
        for t in sp(p):
            if t["test_name"] == "csf_gram_stain" and p == "15509818":
                assert (t["sensitivity_pct"], t["specificity_pct"]) == (80.0, 97.0)
            else:
                assert t["sensitivity_pct"] is None and t["specificity_pct"] is None
    # the other 5 builders: everything null
    for fn in (spt, gn, nm, nmp, hib):
        for t in fn("15494903"):
            assert t["sensitivity_pct"] is None and t["specificity_pct"] is None
