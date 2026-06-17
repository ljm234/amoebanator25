"""Phase 3.1/3.2/3.3 - tests for ml.baselines (LR+Platt, RF, GBM)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ml.baselines import GBMIsotonic, LogisticPlatt, RFCalibrated, build_all_baselines, lightgbm_available


@pytest.fixture
def synthetic_classification() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        weights=[0.7, 0.3], flip_y=0.05, random_state=0,
    )
    return X, y


def _check_proba_basic(p: np.ndarray, n: int) -> None:
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.any(p > 0.0) and np.any(p < 1.0)


def test_logistic_platt_fits_and_predicts(synthetic_classification: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_classification
    clf = LogisticPlatt().fit(X[:200], y[:200])
    p = clf.predict_proba_high(X[200:])
    _check_proba_basic(p, n=100)


def test_rf_calibrated_fits_and_predicts(synthetic_classification: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_classification
    clf = RFCalibrated(n_estimators=50).fit(X[:200], y[:200])
    p = clf.predict_proba_high(X[200:])
    _check_proba_basic(p, n=100)


def test_gbm_isotonic_fits_and_predicts(synthetic_classification: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_classification
    clf = GBMIsotonic(n_estimators=50).fit(X[:200], y[:200])
    p = clf.predict_proba_high(X[200:])
    _check_proba_basic(p, n=100)
    assert clf.backend_ in {"lightgbm", "sklearn_gbm"}


def test_calling_predict_before_fit_raises() -> None:
    for cls in (LogisticPlatt, RFCalibrated, GBMIsotonic):
        clf = cls()
        with pytest.raises(RuntimeError, match="call fit"):
            clf.predict_proba_high(np.zeros((1, 5)))


def test_baseline_handles_tiny_dataset() -> None:
    """Each baseline must auto-fall-back its CV / calibration when n_per_class < 5."""
    X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    y = np.array([0, 0, 1, 1])
    for cls in (LogisticPlatt, RFCalibrated, GBMIsotonic):
        p = cls().fit(X, y).predict_proba_high(X)
        assert p.shape == (4,)
        assert np.all(np.isfinite(p))


def test_build_all_baselines_returns_three_pairs() -> None:
    items = build_all_baselines()
    assert len(items) == 3
    names = {n for n, _ in items}
    assert names == {"logistic_platt", "rf_calibrated", "gbm_isotonic"}


def test_lightgbm_available_is_bool() -> None:
    assert isinstance(lightgbm_available(), bool)


def test_logistic_outperforms_random_on_separable(synthetic_classification: tuple[np.ndarray, np.ndarray]) -> None:
    """Sanity: with linearly-separable-ish synthetic data, LR + Platt should beat 0.6 AUC."""
    from sklearn.metrics import roc_auc_score

    X, y = synthetic_classification
    clf = LogisticPlatt().fit(X[:200], y[:200])
    p = clf.predict_proba_high(X[200:])
    auc = float(roc_auc_score(y[200:], p))
    assert auc > 0.6
