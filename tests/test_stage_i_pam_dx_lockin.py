"""Stage I lock-in: every PAM (Naegleria) builder emits a NULL Naegleria-fowleri PCR.

The Naegleria-fowleri PCR sens/spec was templated 95/99 across all 30 PAM
case-report anchors (none of which is a diagnostic-accuracy study), so it was
nulled in Stage I. This pins the builder outputs - including the 3 reused
imputation helpers (anjum/capewell/kemble), exercised transitively through the
60 per-vignette builders via generate_vignette - so a future regen cannot
restore the unsupported 95/99.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _naegleria_pcr_sensspec(obj):
    out = []

    def walk(x):
        if isinstance(x, dict):
            tn = str(x.get("test_name", "")).lower()
            if "naegleria" in tn and "pcr" in tn and (
                "sensitivity_pct" in x or "specificity_pct" in x
            ):
                out.append((x.get("test_name"), x.get("sensitivity_pct"), x.get("specificity_pct")))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


def test_pam_naegleria_pcr_nulled_in_all_60_builders():
    import scripts.vignettes.generate_pam_vignettes as g

    specs = list(g.DAY1_DISTRIBUTION) + list(g.DAY2_DISTRIBUTION)
    assert len(specs) == 60, f"expected 60 PAM specs, got {len(specs)}"
    total = 0
    for spec in specs:
        v = g.generate_vignette(spec, g.load_pmid_metadata(spec["pmid"]))
        for tn, s, sp in _naegleria_pcr_sensspec(v):
            total += 1
            assert s is None and sp is None, (
                f"{spec['filename']} {tn!r}: sens={s} spec={sp} (expected None/None)"
            )
    # 56 numeric CSF PCR (46 inline + 10 helper-reuse) + environmental PCR entries
    assert total >= 56, f"expected >=56 Naegleria-PCR dx-tests across builders, found {total}"
