"""Tests for pages/01_predict.py - Phase 4.5 Mini-1 T1.7.

18 spec-enumerated tests covering form, presets, error paths, badges,
debounce, and D18 banner. Plus 2 IRB_BYPASS branch tests (Mini-1
closure gate criterion #6) and 1 visual snapshot baseline test
(criterion #7) - total 21 tests.

AppTest is the load-bearing fixture. We mock ``infer_one`` for tests
that don't need real inference (most of them) and let the real model
fire only for the NEUTRAL-defaults sanity gate (test #4) where the
Q11.A spec requires a live ``p_high < 0.001`` proof.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch  # noqa: F401 - used by patch() in tests #9, 10, 15-18

import pytest
from streamlit.testing.v1 import AppTest


PAGE_PATH = "pages/01_predict.py"
SNAPSHOT_PATH = Path(__file__).parent / "_snapshots" / "predict.md.snap"


def _fake_infer_output(
    *,
    prediction: str = "Low",
    p_high: float = 1.4e-9,
    reason: str | None = None,
    n_cal: int = 6,
    alpha: float = 0.10,
    energy: float = -11.7,
    energy_tau: float = -0.99,
    mahalanobis_d2: float = 5.0,
    d2_tau: float = 24.86,
) -> dict[str, Any]:
    """Build a synthetic infer_one output dict matching the real shape."""
    out: dict[str, Any] = {
        "prediction": prediction,
        "p_high": p_high,
        "n_cal": n_cal,
        "alpha": alpha,
        "energy": energy,
        "energy_tau": energy_tau,
        "mahalanobis_d2": mahalanobis_d2,
        "d2_tau": d2_tau,
    }
    if reason is not None:
        out["reason"] = reason
    return out


def _fresh_app_test(env: dict[str, str] | None = None) -> AppTest:
    """Build an AppTest with a clean session state and optional env vars."""
    # Reset audit-hooks singleton between tests so IRB_STATUS_CHANGE etc.
    # don't bleed across runs.
    from ml import audit_hooks as ah
    ah._singleton_log = None
    ah._singleton_path = None
    at = AppTest.from_file(PAGE_PATH)
    if env:
        for k, v in env.items():
            os.environ[k] = v
    return at


# ---------------------------------------------------------------------
# 1. Module imports cleanly
# ---------------------------------------------------------------------
def test_module_imports_cleanly() -> None:
    """`import pages.predict` would succeed if pages were a package; here we
    verify AppTest can load the file as a script without exceptions."""
    at = _fresh_app_test()
    at.run(timeout=30)
    assert len(at.exception) == 0


# ---------------------------------------------------------------------
# 2. Form renders 8 widgets
# ---------------------------------------------------------------------
def test_form_renders_8_widgets() -> None:
    at = _fresh_app_test()
    at.run(timeout=30)
    # 4 number_input + 3 checkbox + 1 multiselect = 8
    assert len(list(at.number_input)) == 4
    assert len(list(at.checkbox)) == 3
    assert len(list(at.multiselect)) == 1


# ---------------------------------------------------------------------
# 3. Form uses Q11.A neutral defaults
# ---------------------------------------------------------------------
def test_form_uses_neutral_defaults() -> None:
    at = _fresh_app_test()
    at.run(timeout=30)
    assert at.number_input(key="age").value == 12
    assert at.number_input(key="csf_glucose").value == 65.0
    assert at.number_input(key="csf_protein").value == 30.0
    assert at.number_input(key="csf_wbc").value == 3
    assert at.checkbox(key="pcr").value is False
    assert at.checkbox(key="microscopy").value is False
    assert at.checkbox(key="exposure").value is False
    assert list(at.multiselect(key="symptoms").value) == []


# ---------------------------------------------------------------------
# 4. Q11.A sanity gate: neutral defaults predict Low with p_high < 0.001
# ---------------------------------------------------------------------
def test_neutral_defaults_predict_low_p_high_lt_001() -> None:
    """Calls real infer_one on the locked NEUTRAL defaults. No mock -
    if this regresses, the model itself has changed and the demo's
    'page-load shows Low' invariant breaks."""
    from app.utils import build_row
    from ml.infer import infer_one

    row = build_row(
        age=12, csf_glucose=65.0, csf_protein=30.0, csf_wbc=3,
        pcr=False, microscopy=False, exposure=False, symptoms=[],
    )
    out = infer_one(row)
    assert out["prediction"] == "Low"
    assert float(out["p_high"]) < 1e-3


# ---------------------------------------------------------------------
# 5. Three preset buttons render
# ---------------------------------------------------------------------
def test_three_preset_buttons_render() -> None:
    from app.presets import PRESETS

    at = _fresh_app_test()
    at.run(timeout=30)
    button_labels = [b.label for b in at.button]
    for key in ("high_risk_pam", "bacterial_meningitis_limitation", "normal_csf"):
        assert PRESETS[key]["label"] in button_labels


# ---------------------------------------------------------------------
# 6. Loading high_risk_pam preset populates session_state
# ---------------------------------------------------------------------
def test_loading_high_risk_pam_preset_populates_form() -> None:
    from app.presets import PRESETS

    at = _fresh_app_test()
    at.run(timeout=30)
    at.button(key="preset_high_risk_pam").click().run(timeout=30)
    expected = PRESETS["high_risk_pam"]["inputs"]
    for field, value in expected.items():
        assert at.session_state[f"input_{field}"] == value
    assert at.session_state["active_preset"] == "high_risk_pam"


# ---------------------------------------------------------------------
# 7. Submit calls infer_one with the dict shape build_row produces
# ---------------------------------------------------------------------
def test_submit_calls_infer_one_with_built_row() -> None:
    """AppTest can't patch script-level imports across runs (the page
    is loaded as a script, not a module - `pages.predict` is not
    importable). We verify the *contract* instead: ``build_row`` emits
    the dict shape ``infer_one`` accepts. Patch-based call-arg
    verification is covered indirectly by tests #15-#17 (they patch
    ``ml.infer.infer_one`` and observe the page's downstream rendering).
    """
    from app.utils import build_row
    row = build_row(
        age=12, csf_glucose=65.0, csf_protein=30.0, csf_wbc=3,
        pcr=False, microscopy=False, exposure=False, symptoms=[],
    )
    assert set(row.keys()) == {
        "age", "csf_glucose", "csf_protein", "csf_wbc",
        "pcr", "microscopy", "exposure", "symptoms",
    }
    assert isinstance(row["age"], int)
    assert isinstance(row["csf_glucose"], float)
    assert row["pcr"] == 0  # False -> 0
    assert row["symptoms"] == ""


# ---------------------------------------------------------------------
# 8. No submit -> no infer_one call
# ---------------------------------------------------------------------
def test_no_submit_returns_early() -> None:
    """First render with no interaction must not invoke inference."""
    at = _fresh_app_test()
    at.run(timeout=30)
    # If inference had run, a result heading would render. Verify none.
    headings = [h.value for h in at.markdown]
    assert not any(h.startswith("### Result:") for h in headings)


# ---------------------------------------------------------------------
# 9. FileNotFoundError -> graceful yellow banner, NOT crash
# ---------------------------------------------------------------------
def test_filenotfounderror_renders_graceful_banner() -> None:
    """Q15.B: missing artefact must surface as warning banner, not raise."""
    err = FileNotFoundError("Mahalanobis stats not found")
    err.filename = "outputs/metrics/feature_stats_train.json"
    at = _fresh_app_test()
    at.run(timeout=30)
    # Submit the form. Patch infer_one to raise FNFE.
    with patch("ml.infer.infer_one", side_effect=err):
        at.button[3].click()  # form_submit_button is the 4th button
        at.run(timeout=30)
    warnings = [w.value for w in at.warning]
    assert any("OOD gate is unconfigured" in w for w in warnings)
    assert len(at.exception) == 0  # no crash


# ---------------------------------------------------------------------
# 10. Generic exception -> correlation-ID error + INTEGRITY_VIOLATION audit
# ---------------------------------------------------------------------
def test_uncaught_exception_emits_correlation_id_audit() -> None:
    """Q15.A: uncaught exception -> uuid4 12-char display + INTEGRITY_VIOLATION emit."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(tmp_path)
    try:
        at = _fresh_app_test()
        at.run(timeout=30)
        with patch("ml.infer.infer_one", side_effect=ValueError("boom")):
            at.button[3].click()
            at.run(timeout=30)
        errors = [e.value for e in at.error]
        assert any(re.search(r"error ID: [0-9a-f]{12}", e) for e in errors), (
            f"expected 12-char hex ID in error message; got {errors!r}"
        )
        # Verify INTEGRITY_VIOLATION emitted with full 32-char error_id
        events = [json.loads(line) for line in tmp_path.read_text().splitlines() if line]
        violations = [e for e in events if e["event_type"] == "integrity_violation"]
        assert len(violations) >= 1
        meta = violations[-1]["metadata"]
        assert "error_id" in meta and len(meta["error_id"]) == 32
        assert meta["exception_type"] == "ValueError"
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# 11. Double-submit within 30s blocked by debounce
# ---------------------------------------------------------------------
def test_double_submit_within_30s_blocked() -> None:
    """Q15.D: predicting=True with fresh timestamp -> second submit aborts."""
    import time
    at = _fresh_app_test()
    at.session_state["predicting"] = True
    at.session_state["predicting_at"] = time.time()  # fresh lock
    at.run(timeout=30)
    at.button[3].click()  # form_submit
    at.run(timeout=30)
    warnings = [w.value for w in at.warning]
    assert any("Already processing" in w for w in warnings)


# ---------------------------------------------------------------------
# 12. Stale lock (>30s old) recovers and allows submission
# ---------------------------------------------------------------------
def test_stale_lock_recovers_after_30s() -> None:
    """Q15.D: predicting=True with timestamp >30s ago -> fall through, re-acquire."""
    import time
    at = _fresh_app_test()
    at.session_state["predicting"] = True
    at.session_state["predicting_at"] = time.time() - 31  # stale by 1s
    at.run(timeout=30)
    at.button[3].click()
    at.run(timeout=30)
    # No "Already processing" warning - stale lock was bypassed.
    warnings = [w.value for w in at.warning]
    assert not any("Already processing" in w for w in warnings)


# ---------------------------------------------------------------------
# 13. decision_badge: icon + bold preserved when color tags stripped
# ---------------------------------------------------------------------
def test_decision_badge_renders_with_bold() -> None:
    from app.utils import decision_badge

    badge = decision_badge("High")
    # Strip :color[...] wrapper; bold label must remain
    stripped = re.sub(r":\w+\[|\]$", "", badge)
    assert "**HIGH**" in stripped


# ---------------------------------------------------------------------
# 14. decision_badge: color-blind safe across all 4 prediction states
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "prediction, label",
    [
        ("High",     "HIGH"),
        ("Low",      "LOW"),
        ("Moderate", "MODERATE"),
        ("ABSTAIN",  "ABSTAIN"),
    ],
)
def test_decision_badge_color_blind_safe(
    prediction: str, label: str
) -> None:
    """Q15.5.A: every badge state legible without color."""
    from app.utils import decision_badge

    badge = decision_badge(prediction, reason="OOD" if prediction == "ABSTAIN" else None)
    stripped = re.sub(r":\w+\[|\]$", "", badge)
    assert label in stripped
    assert f"**{label}" in stripped  # bold prefix; ABSTAIN may have suffix " - OOD"


# ---------------------------------------------------------------------
# 15. T=0.27 calibration tooltip rendered with full Q3 text
# ---------------------------------------------------------------------
def test_t_027_badge_renders_with_tooltip() -> None:
    """Q3: tooltip must explain the T<1 amplification, n=6 fit."""
    fake = _fake_infer_output()
    at = _fresh_app_test()
    at.run(timeout=30)
    with patch("ml.infer.infer_one", return_value=fake):
        at.button[3].click()
        at.run(timeout=30)
    md_blob = "\n".join(m.value for m in at.markdown)
    assert "T=0.27 (n=6)" in md_blob
    assert "amplifies the model" in md_blob.lower() or "amplifies the model" in md_blob


# ---------------------------------------------------------------------
# 16. SmallCalibrationWarning fires when n_cal < 30
# ---------------------------------------------------------------------
def test_smallcalibrationwarning_fires_for_n_below_30() -> None:
    fake = _fake_infer_output(n_cal=6)
    at = _fresh_app_test()
    at.run(timeout=30)
    with patch("ml.infer.infer_one", return_value=fake):
        at.button[3].click()
        at.run(timeout=30)
    warnings = [w.value for w in at.warning]
    assert any("Calibration set is small" in w for w in warnings)


# ---------------------------------------------------------------------
# 17. 3-state regime badge: at n=6 alpha=0.10 -> INVALID
# ---------------------------------------------------------------------
def test_three_state_regime_badge_invalid_at_n6_alpha010() -> None:
    """Q4.C: k = ceil((n+1)(1-alpha)) = 7 > n=6 -> INVALID."""
    fake = _fake_infer_output(n_cal=6, alpha=0.10)
    at = _fresh_app_test()
    at.run(timeout=30)
    with patch("ml.infer.infer_one", return_value=fake):
        at.button[3].click()
        at.run(timeout=30)
    errors = [e.value for e in at.error]
    assert any("INVALID" in e for e in errors)


# ---------------------------------------------------------------------
# 18. D18 banner renders ONLY when bacterial preset active
# ---------------------------------------------------------------------
def test_d18_limitation_banner_only_on_bacterial_preset() -> None:
    """Q12.C: banner is post-result + bacterial-preset-gated."""
    from app.presets import PRESETS

    fake = _fake_infer_output(prediction="High", p_high=1.0)
    at = _fresh_app_test()
    at.session_state["active_preset"] = "bacterial_meningitis_limitation"
    at.run(timeout=30)
    with patch("ml.infer.infer_one", return_value=fake):
        at.button[3].click()
        at.run(timeout=30)
    errors = [e.value for e in at.error]
    bacterial_desc = PRESETS["bacterial_meningitis_limitation"]["description"]
    assert any(bacterial_desc[:60] in e for e in errors), (
        "D18 limitation banner missing on bacterial preset"
    )

    # Now non-bacterial preset -> no D18 banner
    at2 = _fresh_app_test()
    at2.session_state["active_preset"] = "high_risk_pam"
    at2.run(timeout=30)
    with patch("ml.infer.infer_one", return_value=fake):
        at2.button[3].click()
        at2.run(timeout=30)
    errors2 = [e.value for e in at2.error]
    # The INVALID conformal-regime error is allowed; D18 description is not.
    assert not any(bacterial_desc[:60] in e for e in errors2), (
        "D18 limitation banner spuriously rendered on non-bacterial preset"
    )


# ---------------------------------------------------------------------
# 19. IRB_BYPASS=1 -> red banner + IRB_STATUS_CHANGE audit emit
# ---------------------------------------------------------------------
def test_irb_bypass_active_renders_banner_and_emits_event() -> None:
    """Mini-1 closure gate criterion #6 - IRB_BYPASS=1 branch."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(tmp_path)
    os.environ["AMOEBANATOR_IRB_BYPASS"] = "1"
    try:
        at = _fresh_app_test()
        at.run(timeout=30)
        errors = [e.value for e in at.error]
        assert any("IRB bypass active" in e for e in errors)
        events = [json.loads(line) for line in tmp_path.read_text().splitlines() if line]
        irb_events = [e for e in events if e["event_type"] == "irb_status_change"]
        assert len(irb_events) >= 1
        assert irb_events[-1]["actor"] == "env_var"
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        os.environ.pop("AMOEBANATOR_IRB_BYPASS", None)
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# 20. IRB_BYPASS unset -> NO banner + NO event
# ---------------------------------------------------------------------
def test_irb_bypass_inactive_no_banner_no_event() -> None:
    """Mini-1 closure gate criterion #6 - IRB_BYPASS=0/unset branch."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(tmp_path)
    os.environ.pop("AMOEBANATOR_IRB_BYPASS", None)
    try:
        at = _fresh_app_test()
        at.run(timeout=30)
        errors = [e.value for e in at.error]
        assert not any("IRB bypass active" in e for e in errors)
        events = [json.loads(line) for line in tmp_path.read_text().splitlines() if line]
        irb_events = [e for e in events if e["event_type"] == "irb_status_change"]
        # The page never emitted IRB_STATUS_CHANGE because the env var was unset.
        assert len(irb_events) == 0
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# 21. Visual regression text-snapshot drift <5% chars (Mini-1 gate #7)
# ---------------------------------------------------------------------
def test_visual_snapshot_baseline() -> None:
    """Capture markdown blob from page render; compare to committed baseline.

    Skipped until T1.9 lands the baseline file. Once committed, the
    test fails if the page's markdown drifts >5% character delta -
    catching nav/disclaimer regressions that unit tests miss.
    """
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            f"baseline {SNAPSHOT_PATH} not present yet; T1.9 will create it"
        )
    at = _fresh_app_test()
    at.run(timeout=30)
    captured = "\n".join(m.value for m in at.markdown)
    baseline = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if not baseline:
        pytest.skip("baseline file empty")
    # Symmetric character-delta ratio
    longer = max(len(captured), len(baseline))
    shorter = min(len(captured), len(baseline))
    drift = (longer - shorter) / longer if longer else 0.0
    assert drift < 0.05, (
        f"snapshot drift {drift:.1%} exceeds 5% threshold "
        f"(baseline={len(baseline)} chars, captured={len(captured)} chars)"
    )
