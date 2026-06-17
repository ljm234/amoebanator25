"""Phase 3.5 - tests for ml.metrics.bootstrap (bootstrap_ci, bootstrap_ci_paired)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml.metrics.bootstrap import bootstrap_ci, bootstrap_ci_paired


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def test_basic_shape_and_keys() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    out = bootstrap_ci(_auc, y, p, n_resamples=300, alpha=0.05, seed=0)
    assert {"point", "mean", "std", "lo", "hi", "alpha", "n_resamples", "n_skipped"} <= set(out.keys())
    assert math.isfinite(out["lo"])
    assert math.isfinite(out["hi"])
    assert out["lo"] <= out["hi"]


def test_ci_brackets_point_estimate_for_random_score() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    y = rng.integers(0, 2, size=n)
    p = rng.random(n)
    out = bootstrap_ci(_auc, y, p, n_resamples=500, alpha=0.05, seed=1)
    # CI should bracket the chance level 0.5 for a random scorer
    assert out["lo"] < 0.5 + 0.1
    assert out["hi"] > 0.5 - 0.1
    # CI half-width should be small with n=1000
    assert (out["hi"] - out["lo"]) < 0.15


def test_perfect_classifier_has_tight_ci_at_one() -> None:
    n = 200
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p = y.astype(float) + np.random.default_rng(2).normal(0, 0.001, size=n)
    out = bootstrap_ci(_auc, y, p, n_resamples=300, alpha=0.05, seed=2)
    assert out["point"] > 0.99
    assert out["hi"] >= 0.99
    assert (out["hi"] - out["lo"]) < 0.05


def test_seeded_results_are_reproducible() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    a = bootstrap_ci(_auc, y, p, n_resamples=200, seed=99)
    b = bootstrap_ci(_auc, y, p, n_resamples=200, seed=99)
    assert a["lo"] == b["lo"]
    assert a["hi"] == b["hi"]


def test_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(_auc, np.array([0, 1]), np.array([0.1, 0.9]), n_resamples=200, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(_auc, np.array([0, 1]), np.array([0.1, 0.9]), n_resamples=200, alpha=0.6)


def test_n_resamples_minimum() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(_auc, np.array([0, 1]), np.array([0.1, 0.9]), n_resamples=10)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_ci(_auc, np.array([0, 1, 0]), np.array([0.1, 0.9]), n_resamples=200)


def test_paired_ci_zero_for_identical_scorers() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    out = bootstrap_ci_paired(_auc, y, p, p, n_resamples=200, seed=3)
    assert out["point"] == pytest.approx(0.0)
    # CI should bracket 0
    assert out["lo"] <= 0.0 <= out["hi"]


def test_paired_ci_detects_better_scorer() -> None:
    n = 300
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=n)
    p_random = rng.random(n)
    p_perfect = y.astype(float) + rng.normal(0, 0.05, size=n)
    out = bootstrap_ci_paired(_auc, y, p_random, p_perfect, n_resamples=300, seed=4)
    # B should beat A → positive difference
    assert out["point"] > 0.3
    assert out["lo"] > 0


def test_failure_threshold_raises() -> None:
    """Force every resample to fail by giving constant labels."""
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(5).random(100)
    with pytest.raises(RuntimeError, match="More than half"):
        bootstrap_ci(_auc, y, p, n_resamples=200, seed=5)


def test_n_skipped_recorded() -> None:
    """A metric that sometimes errors should report n_skipped > 0."""
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=300)
    p = rng.random(300)
    call_count = {"n": 0}

    def flaky(y_b: np.ndarray, p_b: np.ndarray) -> float:
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            raise ValueError("fail")
        return _auc(y_b, p_b)

    out = bootstrap_ci(flaky, y, p, n_resamples=300, seed=6)
    assert out["n_skipped"] > 0
