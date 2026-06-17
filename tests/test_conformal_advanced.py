"""Phase 4.2 / 4.3 / 4.4 - tests for ml.conformal_advanced."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from ml.conformal_advanced import (
    SMALL_CAL_FLOOR,
    SmallCalibrationWarning,
    compute_qhat,
    coverage_sweep,
    empirical_coverage,
    label_conditional_qhats,
    nonconformity_from_p,
)


def test_compute_qhat_matches_vovk_formula() -> None:
    """qhat = ⌈(n+1)(1-α)⌉-th smallest score with finite-sample correction."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n = len(scores)
    alpha = 0.10
    expected_k = int(np.ceil((n + 1) * (1.0 - alpha)))
    expected_q = float(np.partition(scores, expected_k - 1)[expected_k - 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SmallCalibrationWarning)
        actual = compute_qhat(scores, alpha=alpha)
    assert actual == pytest.approx(expected_q)


def test_compute_qhat_without_correction_matches_quantile() -> None:
    scores = np.linspace(0.0, 1.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = compute_qhat(scores, alpha=0.10, finite_sample_correction=False)
    assert q == pytest.approx(float(np.quantile(scores, 0.90)), rel=1e-3)


def test_small_cal_warning_fires_below_floor() -> None:
    scores = np.linspace(0, 1, SMALL_CAL_FLOOR - 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=SmallCalibrationWarning)
        compute_qhat(scores, alpha=0.10)
    assert any(issubclass(w.category, SmallCalibrationWarning) for w in caught)


def test_small_cal_warning_does_not_fire_above_floor() -> None:
    scores = np.linspace(0, 1, SMALL_CAL_FLOOR + 50)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=SmallCalibrationWarning)
        compute_qhat(scores, alpha=0.10)
    assert not any(issubclass(w.category, SmallCalibrationWarning) for w in caught)


def test_compute_qhat_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        compute_qhat(np.array([0.1, 0.2, 0.3]), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        compute_qhat(np.array([0.1, 0.2, 0.3]), alpha=1.0)


def test_compute_qhat_empty_calibration_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_qhat(np.array([]), alpha=0.10)


def test_label_conditional_returns_per_class_qhats() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    scores = rng.random(n)
    qhats = label_conditional_qhats(scores, y, alpha=0.10)
    assert set(qhats.keys()) == {0, 1}
    for v in qhats.values():
        assert 0.0 <= v <= 1.0


def test_label_conditional_handles_three_classes() -> None:
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 3, size=n)
    scores = rng.random(n)
    qhats = label_conditional_qhats(scores, y, alpha=0.10)
    assert set(qhats.keys()) == {0, 1, 2}


def test_empirical_coverage_perfect_predictor_has_full_coverage() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.05, 0.05, 0.95, 0.95, 0.95])
    out = empirical_coverage(p, y, qhat=0.10)
    assert out["coverage"] == 1.0


def test_empirical_coverage_returns_nan_for_empty() -> None:
    out = empirical_coverage(np.array([]), np.array([]), qhat=0.10)
    assert math.isnan(out["coverage"])
    assert out["n"] == 0


def test_nonconformity_from_p_correct_for_positive_class() -> None:
    p = np.array([0.9, 0.2])
    y = np.array([1, 0])
    s = nonconformity_from_p(p, y)
    assert s[0] == pytest.approx(1.0 - 0.9)
    assert s[1] == pytest.approx(1.0 - (1.0 - 0.2))


def test_coverage_sweep_returns_one_row_per_alpha() -> None:
    rng = np.random.default_rng(2)
    n_cal, n_test = 200, 200
    p_cal = rng.random(n_cal)
    y_cal = (p_cal > 0.5).astype(int)
    cal_scores = nonconformity_from_p(p_cal, y_cal)
    p_test = rng.random(n_test)
    y_test = (p_test > 0.5).astype(int)
    rows = coverage_sweep(cal_scores, p_test, y_test, alphas=(0.05, 0.10, 0.20))
    assert len(rows) == 3
    for r, a in zip(rows, (0.05, 0.10, 0.20), strict=True):
        assert r["alpha"] == pytest.approx(a)


def test_coverage_sweep_empirical_close_to_target_at_large_n() -> None:
    """With n_cal=2000, empirical coverage should track the target alpha."""
    rng = np.random.default_rng(3)
    n = 2000
    p = rng.beta(2.0, 2.0, size=n)
    y = (rng.random(n) < p).astype(int)
    cal = slice(0, n // 2)
    test = slice(n // 2, n)
    cal_scores = nonconformity_from_p(p[cal], y[cal])
    rows = coverage_sweep(cal_scores, p[test], y[test], alphas=(0.10,))
    target = 0.90
    assert abs(rows[0]["coverage"] - target) < 0.05  # within 5pp at n=1000
