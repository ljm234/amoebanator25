"""Predict page - Phase 4.5 Mini-1.

Form-based PAM risk prediction using the n=30 MLP at outputs/model/model.pt.
Wires the existing ml.infer.infer_one path (frozen - do not modify) through:

- 8 form widgets with NEUTRAL clinical defaults (Q11.A locked).
- 3 preset buttons (high_risk_pam / bacterial_meningitis_limitation /
  normal_csf) per Q12.A locked spec.
- D18 limitation banner adjacent to result when bacterial preset
  active (Q12.C locked: post-result, NOT pre-inference).
- Q15.A correlation-ID error path: uuid4 full server-side, 12-char
  display + INTEGRITY_VIOLATION audit emit.
- Q15.B graceful FileNotFoundError banner when Mahalanobis stats
  missing.
- Q15.D session-state debounce with 30s stale-lock recovery.
- 4 result badges: decision (Q15.5.A), T=0.27 calibration tooltip
  (Q3), SmallCalibrationWarning if n_cal<30, 3-state conformal regime
  badge green/yellow/red (Q4.C).
- IRB_BYPASS env-var branch (Mini-1 closure gate criterion #6):
  AMOEBANATOR_IRB_BYPASS=1 -> red banner + IRB_STATUS_CHANGE emit.
"""
from __future__ import annotations

import math
import os
import time
import uuid
from typing import Any

import streamlit as st

from app.disclaimer import render_disclaimer
from app.presets import PRESETS
from app.utils import KNOWN_SYMPTOMS, _fmt_metric, build_row, decision_badge
from ml.audit_hooks import _emit
from ml.data.audit_trail import AuditEventType
from ml.infer import infer_one


st.set_page_config(page_title="Predict - Amoebanator 25")
render_disclaimer()


# -- IRB_BYPASS branch (Mini-1 closure gate criterion #6) -------------
_irb_bypass_active = os.environ.get("AMOEBANATOR_IRB_BYPASS") == "1"
if _irb_bypass_active and not st.session_state.get("_irb_bypass_emitted"):
    _emit(
        AuditEventType.IRB_STATUS_CHANGE,
        actor="env_var",
        resource="AMOEBANATOR_IRB_BYPASS",
        action_detail="bypass active - synthetic-data research mode",
        metadata={"bypass": True},
    )
    st.session_state["_irb_bypass_emitted"] = True

if _irb_bypass_active:
    st.error(
        "IRB bypass active - research mode only. "
        "Synthetic n=30 data; no PHI. Phase 6 with MIMIC-IV will require "
        "AMOEBANATOR_IRB_BYPASS=0 + a real IRB record."
    )

st.title("PAM Risk Prediction")


# -- Preset buttons (Q12.A: 3 buttons + neutral default state) --------
_preset_cols = st.columns(3)
for _col, _key in zip(
    _preset_cols,
    ("high_risk_pam", "bacterial_meningitis_limitation", "normal_csf"),
):
    if _col.button(PRESETS[_key]["label"], key=f"preset_{_key}"):
        # Update session_state with preset inputs (form widgets read from
        # session_state so this populates the form on next render).
        for _field, _value in PRESETS[_key]["inputs"].items():
            st.session_state[f"input_{_field}"] = _value
        st.session_state["active_preset"] = _key
        _emit(
            AuditEventType.WEB_PRESET_LOADED,
            actor="streamlit_user",
            resource="pages/01_predict.py",
            action_detail=f"preset_loaded={_key}",
            metadata={"preset": _key},
        )


# -- Form (Q11.A locked NEUTRAL defaults) -----------------------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    age = col1.number_input(
        "Age (years)", min_value=0, max_value=120,
        value=int(st.session_state.get("input_age", 12)), key="age",
    )
    csf_glucose = col1.number_input(
        "CSF glucose (mg/dL)", min_value=0.0, max_value=500.0,
        value=float(st.session_state.get("input_csf_glucose", 65.0)),
        step=1.0, key="csf_glucose",
    )
    csf_protein = col1.number_input(
        "CSF protein (mg/dL)", min_value=0.0, max_value=1000.0,
        value=float(st.session_state.get("input_csf_protein", 30.0)),
        step=1.0, key="csf_protein",
    )
    csf_wbc = col1.number_input(
        "CSF WBC (cells/uL)", min_value=0, max_value=50000,
        value=int(st.session_state.get("input_csf_wbc", 3)),
        step=1, key="csf_wbc",
    )
    pcr = col2.checkbox(
        "PCR positive",
        value=bool(st.session_state.get("input_pcr", False)), key="pcr",
    )
    microscopy = col2.checkbox(
        "Microscopy positive",
        value=bool(st.session_state.get("input_microscopy", False)),
        key="microscopy",
    )
    exposure = col2.checkbox(
        "Recent freshwater exposure",
        value=bool(st.session_state.get("input_exposure", False)),
        key="exposure",
    )
    symptoms = col2.multiselect(
        "Symptoms", options=list(KNOWN_SYMPTOMS),
        default=list(st.session_state.get("input_symptoms", [])),
        key="symptoms",
    )
    submitted = st.form_submit_button("Run inference")


def _render_result(out: dict[str, Any]) -> None:
    """Render the 4 locked badges + key metrics."""
    badge = decision_badge(str(out.get("prediction", "")), out.get("reason"))
    st.markdown(f"### Result: {badge}")

    # Q3: T=0.27 amplification badge with hover tooltip.
    st.markdown(
        '<span title="Calibrated by temperature scaling (Guo 2017, '
        "L-BFGS, n=6 validation). T=0.27 means the calibrator amplifies "
        "the model's raw confidence - typical temperature scaling has "
        "T>1 (attenuation); T<1 here is unusual and reflects fitting "
        "on only 6 samples. ECE and coverage estimates are empirical-"
        'only, not asymptotic. See docs/model_card.md section 9.">'
        "<sub>T=0.27 (n=6)</sub></span>",
        unsafe_allow_html=True,
    )

    # Q3: SmallCalibrationWarning banner when n_cal < 30.
    n_cal = int(out.get("n_cal", 6))
    if n_cal < 30:
        st.warning(
            f"Calibration set is small (n={n_cal}). Probability estimates "
            "are indicative only. Do not use as a clinical confidence score."
        )

    # Q4.C: 3-state conformal regime badge from (n, alpha, k).
    alpha = float(out.get("alpha", 0.10))
    n = int(out.get("n_cal", 6))
    k = math.ceil((n + 1) * (1 - alpha))
    if n >= k and n >= 100:
        st.success(
            "ASYMPTOTIC: Guarantee holds; "
            "finite-sample bound 1-alpha + 2/(n+2) is tight."
        )
    elif n >= k:
        st.info(
            "FINITE-SAMPLE: bound holds but loose; "
            "treat reported coverage as empirical."
        )
    else:
        st.error(
            f"INVALID: Order-statistic clamped (k clipped from {k} to "
            f"n={n}); the formal guarantee 1-alpha is mathematically inapplicable. "
            "Reported coverage is the empirical hit-rate on the validation set "
            "only. Phase 6 MIMIC-IV (target n>=200) will fix this."
        )

    # Key numeric metrics - _fmt_metric tolerates missing/None/garbage.
    st.markdown(
        f"**p_high:** {_fmt_metric(out, 'p_high')} &nbsp;&nbsp; "
        f"**Mahalanobis d^2:** {_fmt_metric(out, 'mahalanobis_d2')} "
        f"(tau={_fmt_metric(out, 'd2_tau')}) &nbsp;&nbsp; "
        f"**Logit energy:** {_fmt_metric(out, 'energy')} "
        f"(tau={_fmt_metric(out, 'energy_tau')})"
    )


# -- Submit handler ---------------------------------------------------
if submitted:
    # Q15.D: session-state debounce with 30s stale-lock recovery.
    if st.session_state.get("predicting"):
        lock_age = time.time() - float(
            st.session_state.get("predicting_at", 0.0)
        )
        if lock_age < 30:
            st.warning(
                "Already processing - wait for the current prediction to "
                "complete before submitting again."
            )
            st.stop()
        # else: stale lock, fall through and re-acquire

    st.session_state["predicting"] = True
    st.session_state["predicting_at"] = time.time()
    try:
        row = build_row(
            age=int(age),
            csf_glucose=float(csf_glucose),
            csf_protein=float(csf_protein),
            csf_wbc=int(csf_wbc),
            pcr=bool(pcr),
            microscopy=bool(microscopy),
            exposure=bool(exposure),
            symptoms=list(symptoms),
        )
        _emit(
            AuditEventType.WEB_PREDICT_RECEIVED,
            actor="streamlit_user",
            resource="pages/01_predict.py",
            action_detail="form submitted",
            metadata={"row_keys": sorted(row.keys())},
        )
        out = infer_one(row)
        _render_result(out)
        # Q12.C: D18 banner ONLY when bacterial preset is active, AND
        # adjacent to result (post-inference, not pre-inference).
        if (
            st.session_state.get("active_preset")
            == "bacterial_meningitis_limitation"
        ):
            st.error(PRESETS["bacterial_meningitis_limitation"]["description"])
        _emit(
            AuditEventType.WEB_PREDICT_RETURNED,
            actor="streamlit_user",
            resource="pages/01_predict.py",
            action_detail=f"prediction={out.get('prediction', '?')}",
            metadata={
                "p_high": float(out.get("p_high", float("nan"))),
                "reason": out.get("reason"),
            },
        )
    except FileNotFoundError as e:
        # Q15.B: Mahalanobis stats / model.pt missing -> graceful banner,
        # NOT crash. Disable submit on next render via session_state flag.
        st.warning(
            "OOD gate is unconfigured (required artefact missing: "
            f"{e.filename or e}). All predictions return ABSTAIN/OOD until "
            "re-fit. See README section Quickstart for refit instructions."
        )
        st.session_state["_artefact_missing"] = True
    except Exception as e:  # noqa: BLE001 - Q15.A correlation-ID catch-all
        # Q15.A: uuid4 full server-side, 12-char display, audit emit.
        error_id_full = uuid.uuid4().hex
        error_id_user = error_id_full[:12]
        st.error(
            f"Prediction failed (error ID: {error_id_user}). "
            "Server-side log captured."
        )
        _emit(
            AuditEventType.INTEGRITY_VIOLATION,
            actor="streamlit_user",
            resource="pages/01_predict.py",
            action_detail=f"prediction error: {type(e).__name__}",
            metadata={
                "error_id": error_id_full,
                "exception_type": type(e).__name__,
                "exception_repr": repr(e),
            },
        )
    finally:
        st.session_state["predicting"] = False
        st.session_state["predicting_at"] = 0.0
