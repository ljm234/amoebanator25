"""Utility functions for the Phase 4.5 Streamlit web layer.

Three public helpers consumed by `pages/01_predict.py` and the test
suite:

- ``build_row``     - coerce 8 form widget values into the dict shape
                      ``ml.infer.infer_one`` accepts.
- ``decision_badge`` - render a Streamlit-markdown badge for the
                       prediction state. Icon + bold weight + color
                       tag; meaning preserved when color is stripped
                       (Q15.5.A color-blind safety).
- ``_fmt_metric``    - tolerant numeric formatter; returns ``"-"`` for
                       missing / None / non-numeric values so partial
                       inference output dicts never crash the page.

Plus the module-level constant ``KNOWN_SYMPTOMS`` - the exact 3 symptoms
the n=30 model was trained on (Q11.B). The 4 currently-dropped symptoms
(altered_mental_status, photophobia, nausea_vomiting, seizure) are
deferred to Phase 6 with the MIMIC-IV retrain.
"""
from __future__ import annotations

from typing import Any

# Q11.B locked: only the 3 symptoms the model scores. Surfacing more in
# the UI than the model handles would be a defensible-feature mismatch.
KNOWN_SYMPTOMS: tuple[str, str, str] = ("fever", "headache", "nuchal_rigidity")


# Q15.5.A locked bold-label + color mapping. Stripping the color
# tags MUST leave the bold label legible (achromatopsia + low-vision
# accessibility). The mapping is the single source of truth for badge
# rendering across pages.
_BADGE_MAP: dict[str, tuple[str, str]] = {
    "High":     ("HIGH",     "red"),
    "Low":      ("LOW",      "green"),
    "Moderate": ("MODERATE", "blue"),
    "ABSTAIN":  ("ABSTAIN",  "orange"),
}


def build_row(
    age: int,
    csf_glucose: float,
    csf_protein: float,
    csf_wbc: int,
    pcr: bool,
    microscopy: bool,
    exposure: bool,
    symptoms: list[str],
) -> dict[str, Any]:
    """Convert form widget values to ``ml.infer.infer_one``'s input dict.

    Coerces booleans to int 0/1 (model expects numeric flags), joins the
    symptoms multiselect to a semicolon-delimited string, and strips
    blank tokens so an empty multiselect produces ``""`` rather than
    ``";;"``.

    The 8 keys returned match the form widget names exactly so a fresh
    reader of the page can grep keys -> widgets without indirection.
    """
    cleaned_symptoms = [s for s in symptoms if s and s.strip()]
    return {
        "age": int(age),
        "csf_glucose": float(csf_glucose),
        "csf_protein": float(csf_protein),
        "csf_wbc": int(csf_wbc),
        "pcr": int(bool(pcr)),
        "microscopy": int(bool(microscopy)),
        "exposure": int(bool(exposure)),
        "symptoms": ";".join(cleaned_symptoms),
    }


def decision_badge(prediction: str, reason: str | None = None) -> str:
    """Return a Streamlit-markdown badge for the prediction state.

    Format: ``:<color>[**<LABEL>**]`` for High/Low/Moderate;
    ``:<color>[**ABSTAIN - <reason>**]`` for ABSTAIN. Empty or
    unrecognised prediction returns the literal ``"unknown"``.

    The bold label conveys the prediction state
    without relying on color, satisfying the Q15.5.A color-blind safety
    contract: stripping the ``:<color>[...]`` wrapper still leaves the
    semantic content intact.
    """
    if not prediction:
        return "unknown"

    entry = _BADGE_MAP.get(prediction)
    if entry is None:
        return "unknown"

    label, color = entry
    if prediction == "ABSTAIN":
        suffix = reason if reason else "unspecified"
        return f":{color}[**{label} - {suffix}**]"
    return f":{color}[**{label}**]"


def _fmt_metric(out: dict[str, Any], key: str, fmt: str = "{:.3f}") -> str:
    """Format ``out[key]`` via ``fmt``; return ``"-"`` for missing values.

    Tolerates the three failure modes that occur in real inference output
    dicts: (1) the key is absent (some inference branches don't populate
    every field), (2) the value is ``None``, (3) the value is
    non-numeric (string sentinel, garbage). All three render as
    ``"-"`` (em-dash) so the page renders cleanly without ``KeyError``
    or ``TypeError``.
    """
    value = out.get(key)
    if value is None:
        return "-"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return "-"
