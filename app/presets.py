"""Locked clinical presets for the Phase 4.5 predict page.

Three presets cover the demo's discrimination story:

1. ``high_risk_pam``                    - positive control (PAM-likely
                                          pediatric patient).
2. ``bacterial_meningitis_limitation``  - D18 honesty demo: model
                                          cannot distinguish bacterial-
                                          NOT-PAM from PAM at n=30.
                                          UI renders red banner adjacent
                                          to result.
3. ``normal_csf``                       - negative control (adult, no
                                          PAM risk factors).

The page-load NEUTRAL state functions as a fourth implicit scenario.

Per Q12.B: the field name ``current_behavior`` (NOT ``expected``) is
mandatory. ``current_behavior`` is descriptive - it logs what
``infer_one`` returns at ``snapshot_date``. ``expected`` would carry
normative ML connotation that conflicts with the D18 trajectory (Phase
6 will flip the bacterial preset's behavior; we don't *expect* the
current behavior to persist).

The ``limitation_banner`` flag is explicit (not omitted) on every
preset so the UI's render logic doesn't have to handle missing-key
cases.
"""
from __future__ import annotations

from typing import Any


# Snapshot date for current_behavior values. Locked Q12.B + Q15 pre-flip
# re-verification at commit b8f62e3 (post-flip preset coverage table).
_SNAPSHOT_DATE: str = "2026-04-26"


PRESETS: dict[str, dict[str, Any]] = {
    # -- Preset 1: positive control (PAM-likely) -------------------------
    "high_risk_pam": {
        "label": "Load PAM-likely example",
        "description": (
            "Pediatric patient with classic PAM presentation: low CSF "
            "glucose, high protein, high WBC, recent freshwater exposure, "
            "positive PCR and microscopy, full symptom triad. Expected: "
            "High risk prediction."
        ),
        "inputs": {
            "age": 12,
            "csf_glucose": 18.0,
            "csf_protein": 420.0,
            "csf_wbc": 2100,
            "pcr": True,
            "microscopy": True,
            "exposure": True,
            "symptoms": ["fever", "headache", "nuchal_rigidity"],
        },
        "current_behavior": {
            "prediction": "High",
            "p_high_approx": 1.0,
            "snapshot_date": _SNAPSHOT_DATE,
        },
        "limitation_banner": False,
    },

    # -- Preset 2: D18 honesty demo (bacterial NOT PAM) ------------------
    # The model returns prediction="High" because the n=30 training set
    # has zero non-PAM bacterial cases. We surface this preset
    # deliberately as an honesty signal (Q12.C). The corresponding test
    # in tests/test_app_presets.py uses @pytest.mark.xfail(strict=False)
    # so Phase 6 (MIMIC-IV) success -> XPASS as a "fix this" signal
    # without breaking CI.
    "bacterial_meningitis_limitation": {
        "label": "Load bacterial meningitis (limitation demo)",
        "description": (
            "This preset is a known model limitation. Training data "
            "(n=30) contains zero non-PAM bacterial meningitis cases, so "
            "the model cannot distinguish bacterial-NOT-PAM from PAM. "
            "The Phase 6 MIMIC-IV cohort (target n >= 200, includes "
            "bacterial vs viral meningitis labels) will fix this. We "
            "surface this preset deliberately as an honesty signal - "
            "every model has limits, and showing them where they bite is "
            "more useful than hiding them. Try the other 2 presets to "
            "see the model's working regime."
        ),
        "inputs": {
            "age": 45,
            "csf_glucose": 38.0,
            "csf_protein": 180.0,
            "csf_wbc": 2500,
            "pcr": False,
            "microscopy": False,
            "exposure": False,
            "symptoms": ["fever", "headache", "nuchal_rigidity"],
        },
        "current_behavior": {
            "prediction": "High",
            "p_high_approx": 1.0,
            "snapshot_date": _SNAPSHOT_DATE,
        },
        # UI renders description as red banner adjacent to result panel
        # (Q12.C lock: NOT before "Run inference" - co-located).
        "limitation_banner": True,
    },

    # -- Preset 3: negative control (normal CSF) -------------------------
    "normal_csf": {
        "label": "Load normal CSF example",
        "description": (
            "Adult patient with normal CSF profile and no PAM risk "
            "factors. Expected: Low risk prediction."
        ),
        "inputs": {
            "age": 35,
            "csf_glucose": 65.0,
            "csf_protein": 30.0,
            "csf_wbc": 3,
            "pcr": False,
            "microscopy": False,
            "exposure": False,
            "symptoms": [],
        },
        "current_behavior": {
            "prediction": "Low",
            # 1.89e-13 verified live at b8f62e3 post-flip re-verification.
            "p_high_approx": 1.89e-13,
            "snapshot_date": _SNAPSHOT_DATE,
        },
        "limitation_banner": False,
    },
}


def load_preset(key: str) -> dict[str, Any]:
    """Return the preset dict for ``key``.

    Raises ``KeyError`` if ``key`` not in :data:`PRESETS` - fail-loud
    over silent default so an upstream typo surfaces immediately rather
    than rendering an empty form.
    """
    return PRESETS[key]
