"""About page - Phase 4.5 Mini-2 T2.2.

Reviewer-grade landing page for the model card excerpt + feature
importance panel + interactive conformal regime explorer + authorship
disclosure. The panels follow a standard reporting order: architecture, training, calibration, explainability, uncertainty, authorship.

Q17.A/B/C locked: |w_i| panel renders ONLY here (NOT on the predict
page) because |w_i| is model-level, not per-prediction; rendering
adjacent to a result would falsely imply input-specificity. The
caption is fixed, model-level text.

Q4.A locked: Advanced expander hosts the α slider so PIs can move
α ∈ {0.05, 0.10, 0.20} live and watch q-hat + the regime badge
respond. Pedagogical, not load-bearing for the landing-page render.

The authorship section names the repository (github.com/ljm234/
amoebanator25) and notes the companion HuggingFace Space is by the
same author.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from app.disclaimer import render_disclaimer


_MODEL_PATH = Path("outputs/model/model.pt")
_FEATURE_NAMES: tuple[str, ...] = (
    "age",
    "csf_glucose",
    "csf_protein",
    "csf_wbc",
    "pcr",
    "microscopy",
    "exposure",
    "sym_fever",
    "sym_headache",
    "sym_nuchal_rigidity",
)


@st.cache_resource
def _compute_feature_importance() -> pd.DataFrame:
    """Load model.pt, extract first Linear(10,32) layer weight, compute |w_i|.

    Returns a DataFrame with feature names + normalized |w_i| values
    suitable for st.bar_chart consumption. Cached because the model
    is frozen - recomputing every rerun would be wasted CPU.
    """
    state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    w = state["net.0.weight"]  # shape (32, 10) - Linear(in=10, out=32)
    imp = w.abs().mean(dim=0).numpy()  # mean across 32 output dims → (10,)
    imp_norm = imp / imp.sum()
    return pd.DataFrame({"feature": list(_FEATURE_NAMES), "|w_i|": imp_norm})


st.set_page_config(page_title="About - Amoebanator 25")
render_disclaimer()

st.title("About Amoebanator 25")


# -- §1. Model architecture --------------------------------------------
st.subheader("Model architecture")
st.markdown(
    "Tabular MLP, 914 parameters, 6.4 KB serialized. "
    "`Linear(10, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 2)` - "
    "binary classifier (PAM risk: Low / High) with two output logits "
    "consumed by softmax + temperature scaling at inference time. "
    "Architecture verified against `outputs/model/model.pt` "
    "tensor shapes during the Phase 4.5 PRE-FLIGHT audit."
)


# -- §2. Training summary ----------------------------------------------
st.subheader("Training data")
st.markdown(
    "**n=30 synthetic patient vignettes** drawn from published case-series "
    "marginals (Yoder 2010, Cope 2016, CDC 2025). Train/val split: "
    "n_train=24, n_val=6, `random_state=42`, `test_size=0.2`, "
    "`stratify=y`. Zero real PHI; vignettes are reproducibility-friendly "
    "but not externally calibrated - Phase 6 (MIMIC-IV cohort, target "
    "n ≥ 200) will provide the first contact with real-world clinical data. "
    "See `docs/data_card.md` (Gebru et al. 2021 datasheet format) for "
    "the full lineage."
)


# -- §3. Calibration summary -------------------------------------------
st.subheader("Calibration")
st.markdown(
    "Temperature scaling (Guo et al. 2017) optimised via L-BFGS on the "
    "n=6 validation set. Current `T = 0.27`. **T < 1 means the calibrator "
    "amplifies the model's raw confidence** - the opposite of typical "
    "Guo 2017 behaviour (T > 1 attenuates overconfidence). On n=6 the "
    "L-BFGS landscape lacks curvature to constrain T meaningfully; "
    "different random subsets of n=6 would produce T values in the "
    "range 0.1-2.0. Treat the reported T as a sample-specific point "
    "estimate, not as evidence of structural under-/over-confidence. "
    "See `docs/model_card.md` §Caveats for the full discussion."
)


# -- §4. Feature importance via |w_i| (Q17.A/B/C) ----------------------
st.subheader("Feature importance (model-level)")
imp_df = _compute_feature_importance()
st.bar_chart(imp_df, x="feature", y="|w_i|", horizontal=True)

# Q17.C verbatim caption - locked text, do not reword.
st.caption(
    "Feature importance via |w_i| (model-level mean of first Linear "
    "layer weights, normalized). NOT per-prediction attribution - for "
    "that, see SHAP (deferred to Phase 6 with MIMIC-IV n ≥ 200). "
    f"Current range: {imp_df['|w_i|'].min():.1%} to "
    f"{imp_df['|w_i|'].max():.1%}, max/min ratio "
    f"{imp_df['|w_i|'].max() / imp_df['|w_i|'].min():.2f}×. "
    "Interpretation: the model treats all 10 features near-equally, "
    "consistent with the n=30 training set limitation. SHAP on n=30 "
    "background data is mathematically vacuous; this panel is the "
    "honest substitute at current scale. See `docs/model_card.md` "
    "§Caveats for full discussion."
)


# -- §5. Conformal advanced expander (Q4.A α slider) -------------------
with st.expander("Advanced: explore conformal coverage"):
    st.markdown(
        "Move the slider to see how `q-hat` and the regime badge "
        "respond to different significance levels. The 3-state regime "
        "badge (ASYMPTOTIC / FINITE-SAMPLE / INVALID) is "
        "computed from `(n_cal, α, k)` where "
        "`k = ⌈(n_cal + 1)(1 − α)⌉`."
    )
    alpha = st.slider(
        "α (significance level)",
        min_value=0.05, max_value=0.20, value=0.10, step=0.05,
        key="conformal_alpha_slider",
    )
    n_cal = 6  # current n; Phase 6 will lift this to ≥200
    k = math.ceil((n_cal + 1) * (1 - alpha))
    st.markdown(
        f"With `n_cal = {n_cal}`, `α = {alpha:.2f}` → "
        f"`k = ⌈(n+1)(1−α)⌉ = {k}`."
    )
    if n_cal >= k and n_cal >= 100:
        st.success(
            "ASYMPTOTIC: Guarantee holds; "
            "finite-sample bound 1−α + 2/(n+2) is tight."
        )
    elif n_cal >= k:
        st.info(
            "FINITE-SAMPLE: bound holds but loose; "
            "treat reported coverage as empirical."
        )
    else:
        st.error(
            f"INVALID: Order-statistic clamped (k clipped from {k} "
            f"to n={n_cal}); the formal guarantee 1−α is mathematically "
            "inapplicable. Reported coverage is the empirical hit-rate "
            "on the validation set only. Phase 6 MIMIC-IV (target "
            "n ≥ 200) will fix this."
        )


# -- §6. Authorship + handle disclosure (Q19.D) ------------------------
st.subheader("Authorship")
st.markdown(
    "Jordan Montenegro-Calla - ORCID 0009-0000-7851-7139 - "
    "jordanmontenegroc.99@gmail.com"
)
st.caption(
    "Repo: github.com/ljm234/amoebanator25. The companion "
    "HuggingFace Space is by the same author."
)
