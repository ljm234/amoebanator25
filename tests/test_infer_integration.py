"""
Wired-inference test suite.

This module is the regression boundary for the audit's #1 credibility-killer:
ml.infer used to return a constant (0.0, 0.3) from _toy_logits regardless of
input. After training, infer_one loads outputs/model/model.pt, applies the
fitted temperature, and runs a real forward pass.

Coverage:
  * end-to-end integration: distinct patient rows produce distinct p_high
  * directional sanity: severe presentation scores higher than benign
  * determinism: same input -> identical output (process-cached model)
  * unit tests for _build_feature_vector (symptom parsing, NaN handling,
    direct-column override, missing features, non-numeric strings)
  * unit tests for _real_logits (cache reuse, temperature scaling math)
  * unit tests for _softmax_high (numerical stability, equivalence to numpy)
  * load-time validation (missing artifacts, bad temperature, architecture
    mismatch) all raise actionable errors
  * a constant-output regression sentinel: scans 10 perturbed rows and fails
    if all p_high values collapse to the same number.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import ml.infer as infer_mod
from ml.infer import (
    _build_feature_vector,
    _load_model_artifacts,
    _real_logits,
    _softmax_high,
    infer_one,
)


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "age": 30,
        "csf_glucose": 50.0,
        "csf_protein": 1.0,
        "csf_wbc": 10,
        "pcr": 0,
        "microscopy": 0,
        "exposure": 0,
        "symptoms": "",
        "risk_score": 5,
    }
    base.update(overrides)
    return base


_BENIGN: dict[str, Any] = _row(
    age=45, csf_glucose=70.0, csf_protein=0.4, csf_wbc=3,
    pcr=0, microscopy=0, exposure=0, symptoms="",
)
_SEVERE: dict[str, Any] = _row(
    age=12, csf_glucose=18.0, csf_protein=420.0, csf_wbc=2100,
    pcr=1, microscopy=1, exposure=1,
    symptoms="fever;headache;nuchal_rigidity",
)


# -----------------------------------------------------------------------------
# End-to-end integration
# -----------------------------------------------------------------------------


def test_distinct_inputs_produce_distinct_p_high() -> None:
    out_b = infer_one(_BENIGN)
    out_s = infer_one(_SEVERE)
    assert "p_high" in out_b and "p_high" in out_s
    assert math.isfinite(out_b["p_high"])
    assert math.isfinite(out_s["p_high"])
    assert abs(out_b["p_high"] - out_s["p_high"]) > 1e-6, (
        f"distinct inputs collapsed to identical p_high: "
        f"benign={out_b['p_high']!r} severe={out_s['p_high']!r}. "
        f"Likely cause: _toy_logits regressed or model.pt failed to load."
    )


def test_p_high_is_a_probability() -> None:
    out = infer_one(_row())
    assert "p_high" in out
    assert 0.0 <= float(out["p_high"]) <= 1.0


def test_severe_case_scores_higher_than_benign() -> None:
    out_b = infer_one(_BENIGN)
    out_s = infer_one(_SEVERE)
    assert float(out_s["p_high"]) > float(out_b["p_high"]), (
        f"severe case did not score higher than benign: "
        f"severe={out_s['p_high']!r} benign={out_b['p_high']!r}"
    )


def test_pandas_series_input_matches_dict_input() -> None:
    out_dict = infer_one(_SEVERE)
    out_series = infer_one(pd.Series(_SEVERE))
    assert out_dict["p_high"] == out_series["p_high"]
    assert out_dict["prediction"] == out_series["prediction"]


def test_inference_is_deterministic() -> None:
    out_a = infer_one(_SEVERE)
    out_b = infer_one(_SEVERE)
    assert out_a["p_high"] == out_b["p_high"]
    assert out_a["prediction"] == out_b["prediction"]


def test_no_constant_output_across_perturbations() -> None:
    """
    Regression sentinel for the _toy_logits class of bug.

    Sweeps 10 rows that vary every clinically meaningful feature. If every
    p_high collapses to the same value, the inference path has regressed
    back to a constant predictor and this test fails loudly.
    """
    rows = [
        _row(age=a, csf_glucose=g, csf_protein=p, csf_wbc=w, pcr=pc, microscopy=mi)
        for a, g, p, w, pc, mi in [
            (5, 80, 0.2, 2, 0, 0),
            (15, 60, 1.0, 8, 0, 0),
            (22, 45, 2.5, 25, 1, 0),
            (35, 30, 80, 200, 1, 1),
            (50, 22, 250, 900, 1, 1),
            (8, 18, 420, 2100, 1, 1),
            (60, 65, 0.8, 5, 0, 0),
            (12, 40, 60, 600, 1, 0),
            (28, 55, 5, 40, 0, 1),
            (44, 25, 180, 1500, 1, 1),
        ]
    ]
    p_highs = [float(infer_one(r)["p_high"]) for r in rows]
    assert len(set(p_highs)) > 1, (
        f"All {len(rows)} perturbed inputs produced identical p_high={p_highs[0]!r}. "
        f"This is the _toy_logits failure signature; the model is not wired."
    )


# -----------------------------------------------------------------------------
# Unit tests - _build_feature_vector
# -----------------------------------------------------------------------------


_FEATS: tuple[str, ...] = (
    "age", "csf_glucose", "csf_protein", "csf_wbc",
    "pcr", "microscopy", "exposure",
    "sym_fever", "sym_headache", "sym_nuchal_rigidity",
)


def test_build_feature_vector_empty_row_is_all_zeros() -> None:
    x = _build_feature_vector(pd.Series({}, dtype=object), _FEATS)
    assert x.shape == (len(_FEATS),)
    assert x.dtype == np.float32
    assert (x == 0.0).all()


def test_build_feature_vector_numeric_features_are_preserved() -> None:
    row = pd.Series({"age": 12.0, "csf_glucose": 18.0, "csf_protein": 420.0})
    x = _build_feature_vector(row, _FEATS)
    assert x[0] == pytest.approx(12.0)
    assert x[1] == pytest.approx(18.0)
    assert x[2] == pytest.approx(420.0)
    assert x[3] == 0.0  # csf_wbc absent


def test_build_feature_vector_parses_symptoms_string() -> None:
    row = pd.Series({"symptoms": "fever;nuchal_rigidity"})
    x = _build_feature_vector(row, _FEATS)
    assert x[7] == 1.0  # sym_fever
    assert x[8] == 0.0  # sym_headache absent
    assert x[9] == 1.0  # sym_nuchal_rigidity


def test_build_feature_vector_explicit_sym_column_overrides_string() -> None:
    row = pd.Series({"symptoms": "headache", "sym_fever": 1.0, "sym_headache": 0.0})
    x = _build_feature_vector(row, _FEATS)
    assert x[7] == 1.0  # explicit sym_fever wins
    assert x[8] == 0.0  # explicit sym_headache wins (overrides parsed "headache")


def test_build_feature_vector_handles_nan_and_garbage() -> None:
    row = pd.Series({"age": float("nan"), "csf_glucose": "not-a-number", "csf_protein": 2.0})
    x = _build_feature_vector(row, _FEATS)
    assert x[0] == 0.0  # NaN -> 0
    assert x[1] == 0.0  # non-numeric string -> 0
    assert x[2] == pytest.approx(2.0)


def test_build_feature_vector_ignores_extra_columns() -> None:
    row = pd.Series({"age": 30.0, "totally_unknown_field": 999.0, "another": "x"})
    x = _build_feature_vector(row, _FEATS)
    assert x[0] == pytest.approx(30.0)
    # remaining slots stay at zero; no exceptions raised
    assert (x[1:] == 0.0).all()


# -----------------------------------------------------------------------------
# Unit tests - _real_logits and _softmax_high
# -----------------------------------------------------------------------------


def test_real_logits_is_deterministic() -> None:
    a = _real_logits(pd.Series(_SEVERE))
    b = _real_logits(pd.Series(_SEVERE))
    assert a == b


def test_real_logits_distinct_inputs_distinct_outputs() -> None:
    a = _real_logits(pd.Series(_SEVERE))
    b = _real_logits(pd.Series(_BENIGN))
    assert a != b
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-3


def test_real_logits_applies_temperature_scaling() -> None:
    """Verify scaled = raw / T to within float32 precision."""
    model, feats, T = _load_model_artifacts()
    row = pd.Series(_SEVERE)
    x = _build_feature_vector(row, feats)
    with torch.no_grad():
        raw = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    expected = (float(raw[0] / T), float(raw[1] / T))
    actual = _real_logits(row)
    assert actual[0] == pytest.approx(expected[0], rel=1e-5)
    assert actual[1] == pytest.approx(expected[1], rel=1e-5)


def test_load_model_artifacts_is_cached() -> None:
    m1, _, _ = _load_model_artifacts()
    m2, _, _ = _load_model_artifacts()
    assert m1 is m2


def test_softmax_high_matches_numpy() -> None:
    for lo, hi in [(0.0, 0.0), (1.0, -1.0), (-3.0, 5.0), (10.0, 10.0)]:
        expected = float(np.exp(hi) / (np.exp(lo) + np.exp(hi)))
        assert _softmax_high(lo, hi) == pytest.approx(expected, rel=1e-9)


def test_softmax_high_numerical_stability_large_logits() -> None:
    """Inputs that would overflow exp() must still produce a valid probability."""
    p = _softmax_high(1000.0, 1001.0)
    assert math.isfinite(p)
    assert 0.0 < p < 1.0
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_softmax_high_returns_zero_to_one() -> None:
    for lo, hi in [(-1e6, 1e6), (1e6, -1e6), (0.0, 0.0)]:
        p = _softmax_high(lo, hi)
        assert 0.0 <= p <= 1.0


# -----------------------------------------------------------------------------
# Load-time validation - error paths
# -----------------------------------------------------------------------------


@pytest.fixture()
def _isolated_model_dir(tmp_path: Path) -> Iterator[Path]:
    """Yield a tmp_path patched as MODEL_DIR/MODEL_PATH/etc with the cache cleared."""
    md = tmp_path / "model"
    md.mkdir()
    _load_model_artifacts.cache_clear()
    with (
        patch.object(infer_mod, "MODEL_DIR", md),
        patch.object(infer_mod, "MODEL_PATH", md / "model.pt"),
        patch.object(infer_mod, "FEATURES_JSON", md / "features.json"),
        patch.object(infer_mod, "TEMPERATURE_JSON", md / "temperature_scale.json"),
    ):
        yield md
    _load_model_artifacts.cache_clear()


def test_missing_model_pt_raises_clear_error(_isolated_model_dir: Path) -> None:
    (_isolated_model_dir / "features.json").write_text(json.dumps(["age"]))
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(FileNotFoundError, match="model weights"):
        _load_model_artifacts()


def test_missing_features_json_raises_clear_error(_isolated_model_dir: Path) -> None:
    (_isolated_model_dir / "model.pt").write_bytes(b"placeholder")
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(FileNotFoundError, match="feature schema"):
        _load_model_artifacts()


def test_missing_temperature_raises_clear_error(_isolated_model_dir: Path) -> None:
    (_isolated_model_dir / "model.pt").write_bytes(b"placeholder")
    (_isolated_model_dir / "features.json").write_text(json.dumps(["age"]))
    with pytest.raises(FileNotFoundError, match="temperature scale"):
        _load_model_artifacts()


def test_empty_features_json_raises_value_error(_isolated_model_dir: Path) -> None:
    (_isolated_model_dir / "model.pt").write_bytes(b"placeholder")
    (_isolated_model_dir / "features.json").write_text(json.dumps([]))
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(ValueError, match="non-empty JSON list"):
        _load_model_artifacts()


@pytest.mark.parametrize("bad_t", [0.0, -1.0, float("nan"), float("inf"), "abc", None])
def test_bad_temperature_raises_value_error(_isolated_model_dir: Path, bad_t: object) -> None:
    (_isolated_model_dir / "model.pt").write_bytes(b"placeholder")
    (_isolated_model_dir / "features.json").write_text(json.dumps(["age"]))
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": bad_t}))
    with pytest.raises(ValueError):
        _load_model_artifacts()


def test_state_dict_architecture_mismatch_raises_value_error(_isolated_model_dir: Path) -> None:
    """A state_dict whose keys don't match MLP must fail with a remediation hint."""
    bogus = {"some.unrelated.layer.weight": torch.zeros(2, 2)}
    torch.save(bogus, _isolated_model_dir / "model.pt")
    (_isolated_model_dir / "features.json").write_text(json.dumps(["age", "csf_glucose"]))
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(ValueError, match="state_dict does not match"):
        _load_model_artifacts()


def test_non_dict_pickle_raises_value_error(_isolated_model_dir: Path) -> None:
    torch.save([1, 2, 3], _isolated_model_dir / "model.pt")
    (_isolated_model_dir / "features.json").write_text(json.dumps(["age"]))
    (_isolated_model_dir / "temperature_scale.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(ValueError, match="state_dict"):
        _load_model_artifacts()
