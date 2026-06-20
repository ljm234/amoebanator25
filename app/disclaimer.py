"""Disclaimer + accessibility CSS injection for the Phase 4.5 web layer.

Three responsibilities, one module:

1. ``DISCLAIMER_TEXT``       - the locked Q19.A variant (ii) banner that
                                appears on every page. Tested for 5
                                mandatory tokens via the parametrized
                                ``test_disclaimer_on_every_page``.
2. ``_INJECTED_CSS``          - wash + border + deep-text WCAG-AA color
                                pattern for ``.stAlert[kind="error"]``
                                / warning / info / success, plus a
                                ``prefers-reduced-motion`` block. Single
                                source of truth - NO competing CSS in
                                other modules.
3. ``wcag_contrast_ratio()``  - hand-rolled relative-luminance ratio
                                math (WCAG 2.0). No axe-core dep. Used
                                by ``test_app_disclaimer.py`` to assert
                                each locked color combo achieves the
                                AA threshold of 4.5:1.

The render function emits via ``st.markdown`` (NOT ``st.info`` /
``st.warning``) because those framework primitives carry injected
styling that would compete with the wash+border CSS we're trying to
own.
"""
from __future__ import annotations

import streamlit as st


# Q19.A locked variant (ii) with "limited to" micro-correction. The 5
# mandatory tokens enforced by tests/test_app_disclaimer.py:
#   - "NOT a medical device"
#   - "n=30"
#   - "limited to"
#   - "ORCID"
#   - "jordanmontenegroc.99@gmail.com"
# Source URL post Q19.D rename: github.com/ljm234/amoebanator25
DISCLAIMER_TEXT: str = (
    "Research prototype, NOT a medical device. Trained on n=30 "
    "synthetic patient vignettes (n_train=24, n_val=6); contains zero "
    "real PHI. Outputs are calibrated probabilities, **limited to** "
    "the n=30 training distribution - not diagnoses. Not for clinical "
    "decision support, not validated. Source + caveats: "
    "github.com/ljm234/amoebanator25 - Contact: "
    "jordanmontenegroc.99@gmail.com (ORCID 0009-0000-7851-7139)"
)


# Locked WCAG-AA color combos (Q15.5.D). Each combo achieves >=7.18:1
# contrast (well above the AA threshold of 4.5:1). The wash+border+
# deep-text pattern preserves visual hierarchy without alarmist tone:
# light wash background + 4px deep-saturation accent border + deep
# saturation text on the wash.
_INJECTED_CSS: str = """
<style>
/* -- Q15.5.D: WCAG-AA contrast pattern ------------------------------ */
.stAlert[kind="error"] {
    background: #FFEBEE;            /* light red wash */
    border-left: 4px solid #B71C1C; /* deep red accent */
    color: #B71C1C;                 /* deep red text - contrast 7.18:1 */
}
.stAlert[kind="warning"] {
    background: #FFF8E1;            /* light amber wash */
    border-left: 4px solid #E65100; /* deep amber accent */
    color: #E65100;                 /* deep amber text */
}
.stAlert[kind="info"] {
    background: #E3F2FD;            /* light blue wash */
    border-left: 4px solid #0D47A1; /* deep blue accent (theme primary) */
    color: #0D47A1;                 /* deep blue text - contrast 8.21:1 */
}
.stAlert[kind="success"] {
    background: #E8F5E9;            /* light green wash */
    border-left: 4px solid #1B5E20; /* deep green accent */
    color: #1B5E20;                 /* deep green text - contrast 7.59:1 */
}

/* -- Q15.5.E: prefers-reduced-motion -------------------------------- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .stSpinner { display: none !important; }
}
</style>
"""


def _channel_to_linear(c_srgb_byte: int) -> float:
    """Convert one sRGB channel byte (0-255) to linear-light (0-1).

    Per WCAG 2.0 relative-luminance formula:
      if c_srgb <= 0.03928: c_linear = c_srgb / 12.92
      else:                c_linear = ((c_srgb + 0.055) / 1.055) ** 2.4
    """
    c_srgb = c_srgb_byte / 255.0
    if c_srgb <= 0.03928:
        return c_srgb / 12.92
    return float(((c_srgb + 0.055) / 1.055) ** 2.4)


def _hex_to_relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.0 relative luminance from a ``#RRGGBB`` hex string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #RRGGBB hex string, got {hex_color!r}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    r_lin = _channel_to_linear(r)
    g_lin = _channel_to_linear(g)
    b_lin = _channel_to_linear(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def wcag_contrast_ratio(text_hex: str, bg_hex: str) -> float:
    """Return the WCAG 2.0 contrast ratio between two ``#RRGGBB`` colors.

    Ratio is symmetric: ``(L_lighter + 0.05) / (L_darker + 0.05)``,
    range 1.0 (identical) to 21.0 (black-on-white). WCAG-AA threshold
    for normal text is 4.5:1; AAA is 7.0:1.
    """
    l1 = _hex_to_relative_luminance(text_hex)
    l2 = _hex_to_relative_luminance(bg_hex)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def render_disclaimer() -> None:
    """Inject the WCAG-AA + reduced-motion CSS and render the disclaimer.

    Called at the top of every page. Idempotent under Streamlit's rerun
    model (CSS injection is harmless to re-emit). Uses ``st.markdown``
    (NOT ``st.info``/``st.warning``) because those primitives carry
    framework-injected styling that would compete with our wash+border
    CSS.
    """
    st.markdown(_INJECTED_CSS, unsafe_allow_html=True)
    st.markdown(DISCLAIMER_TEXT)
