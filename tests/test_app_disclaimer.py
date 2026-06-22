"""WCAG-AA contrast and disclaimer-token assertions for ``app.disclaimer``.

These tests parse the colors out of the actual injected CSS (rather than
hard-coding a copy), so they stay in sync with ``_INJECTED_CSS``: if a color
is changed, the ratio is re-derived and re-checked. They back the claim in the
module docstring that every alert combo clears the WCAG-AA threshold of 4.5:1,
and that the disclaimer banner carries its mandatory tokens.
"""
from __future__ import annotations

import re

import pytest

from app.disclaimer import DISCLAIMER_TEXT, _INJECTED_CSS, wcag_contrast_ratio

AA_THRESHOLD = 4.5

# Pull (kind, foreground, background) out of each ``.stAlert[kind="..."]`` block.
_ALERT_BLOCK = re.compile(
    r'\.stAlert\[kind="(?P<kind>\w+)"\]\s*\{'
    r"[^}]*?background:\s*(?P<bg>#[0-9A-Fa-f]{6})"
    r"[^}]*?color:\s*(?P<fg>#[0-9A-Fa-f]{6})",
    re.DOTALL,
)

ALERT_COMBOS: list[tuple[str, str, str]] = [
    (m.group("kind"), m.group("fg"), m.group("bg"))
    for m in _ALERT_BLOCK.finditer(_INJECTED_CSS)
]

MANDATORY_TOKENS = (
    "NOT a medical device",
    "n=30",
    "limited to",
    "ORCID",
    "jordanmontenegroc.99@gmail.com",
)


def test_four_alert_combos_parsed() -> None:
    """The injected CSS defines exactly the four expected alert kinds."""
    kinds = {kind for kind, _, _ in ALERT_COMBOS}
    assert kinds == {"error", "warning", "info", "success"}


@pytest.mark.parametrize("kind, fg, bg", ALERT_COMBOS)
def test_alert_combo_meets_aa(kind: str, fg: str, bg: str) -> None:
    """Each alert's text-on-wash contrast clears the AA threshold of 4.5:1."""
    ratio = wcag_contrast_ratio(fg, bg)
    assert ratio >= AA_THRESHOLD, (
        f"{kind}: {fg} on {bg} = {ratio:.2f}:1 < {AA_THRESHOLD}"
    )


def test_reference_black_on_white_is_21() -> None:
    """Maximum contrast (black on white) is 21:1."""
    assert abs(wcag_contrast_ratio("#000000", "#FFFFFF") - 21.0) < 0.01


def test_reference_identical_colors_is_1() -> None:
    """Identical colors have the minimum contrast ratio of 1:1."""
    assert abs(wcag_contrast_ratio("#777777", "#777777") - 1.0) < 1e-9


@pytest.mark.parametrize("token", MANDATORY_TOKENS)
def test_disclaimer_contains_mandatory_token(token: str) -> None:
    """The disclaimer banner carries every mandatory safety/contact token."""
    assert token in DISCLAIMER_TEXT
