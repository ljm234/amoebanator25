"""
Advanced split-conformal predictor: marginal + label-conditional + coverage tools.

Phase 4.2 / 4.3 / 4.4. Builds on ml/conformal.py (which only exposes the
threshold-band decision rule) by adding:

  * compute_qhat            - finite-sample-corrected split conformal threshold
  * label_conditional_qhats - per-class threshold dictionary
  * empirical_coverage      - joint and per-class coverage on a held-out set
  * coverage_sweep          - coverage / abstain-rate across multiple alphas
  * SmallCalibrationWarning - issued when n_cal is below the recommended floor

References:
  - Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World.
    Springer, 2005. (split conformal coverage bound 1 - α - 1/(n+1) ≤ E[cov] ≤ 1 - α + 1/(n+1))
  - Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. "Distribution-Free
    Predictive Inference for Regression." JASA 2018.
  - Vovk V. "Conditional Validity of Inductive Conformal Predictors."
    Mach Learn 2013;92:349-376. (label-conditional / Mondrian conformal)
"""
from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np


SMALL_CAL_FLOOR: int = 100  # below this, the finite-sample correction matters in practice


class SmallCalibrationWarning(UserWarning):
    """Emitted when a conformal calibration set is too small for the asymptotic guarantee."""


class CoverageResult(TypedDict):
    alpha: float
    qhat: float
    coverage: float
    abstain_rate: float
    n: int


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")


def _check_calibration_size(n: int) -> None:
    if n < 1:
        raise ValueError(f"calibration set must be non-empty; got n={n}.")
    if n < SMALL_CAL_FLOOR:
        warnings.warn(
            f"Conformal calibration set size n={n} is below the recommended "
            f"floor of {SMALL_CAL_FLOOR}. The finite-sample bound from Vovk et "
            f"al. (1 - α - 1/(n+1) ≤ coverage ≤ 1 - α + 1/(n+1)) implies a "
            f"slack of ±{1.0 / (n + 1):.3f} around the target. Treat coverage "
            f"reports on this set as empirical, not population-level guarantees.",
            SmallCalibrationWarning,
            stacklevel=3,
        )


def compute_qhat(
    nonconformity_scores: np.ndarray,
    alpha: float,
    finite_sample_correction: bool = True,
) -> float:
    """
    Split-conformal qhat from nonconformity scores at miscoverage level alpha.

    With the finite-sample correction (default), the threshold is the
    ⌈(n+1)(1-α)⌉-th smallest score, which yields the Vovk bound. Without the
    correction it falls back to the (1-α) sample quantile.

    Issues SmallCalibrationWarning when n < SMALL_CAL_FLOOR.
    """
    _check_alpha(alpha)
    scores = np.asarray(nonconformity_scores, dtype=float).ravel()
    n = len(scores)
    _check_calibration_size(n)
    if finite_sample_correction:
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(np.partition(scores, k - 1)[k - 1])
    return float(np.quantile(scores, 1.0 - alpha))


def label_conditional_qhats(
    nonconformity_scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    finite_sample_correction: bool = True,
) -> dict[int, float]:
    """
    Vovk 2013 label-conditional (Mondrian) conformal: a separate qhat per class.

    Returns {class_label: qhat}. Per-class coverage holds at 1-α conditional on
    each class label, which is what you want for low-prevalence triage where
    the marginal positive rate would otherwise dominate the threshold.
    """
    _check_alpha(alpha)
    scores = np.asarray(nonconformity_scores, dtype=float).ravel()
    labs = np.asarray(labels).ravel()
    if scores.shape != labs.shape:
        raise ValueError(
            f"scores shape {scores.shape} != labels shape {labs.shape}"
        )
    out: dict[int, float] = {}
    for cls in np.unique(labs):
        mask = labs == cls
        cls_scores = scores[mask]
        if len(cls_scores) == 0:
            continue
        out[int(cls)] = compute_qhat(cls_scores, alpha, finite_sample_correction)
    return out


def empirical_coverage(
    p_high: np.ndarray,
    y_true: np.ndarray,
    qhat: float,
) -> CoverageResult:
    """
    Empirical (joint) coverage and abstain rate of the conformal band on
    held-out data. Returns alpha-style summary so callers can compare to target.
    """
    p = np.asarray(p_high, dtype=float).ravel()
    y = np.asarray(y_true).ravel()
    if p.shape != y.shape:
        raise ValueError(f"p_high shape {p.shape} != y_true shape {y.shape}")
    n = len(p)
    if n == 0:
        return {
            "alpha": float("nan"), "qhat": float(qhat),
            "coverage": float("nan"), "abstain_rate": float("nan"), "n": 0,
        }
    include_high = p >= (1.0 - qhat)
    include_low = p <= qhat
    abstain = include_high & include_low
    is_high = (y == 1)
    contained = (is_high & include_high) | (~is_high & include_low)
    coverage = float(contained.mean())
    abstain_rate = float(abstain.mean())
    return {
        "alpha": float(1.0 - coverage),
        "qhat": float(qhat),
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "n": int(n),
    }


def coverage_sweep(
    cal_scores: np.ndarray,
    test_p_high: np.ndarray,
    test_y: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20),
    finite_sample_correction: bool = True,
) -> list[CoverageResult]:
    """
    Fit qhat on `cal_scores` for each alpha, then evaluate empirical coverage
    + abstain rate on (`test_p_high`, `test_y`). Returns one CoverageResult
    per alpha, in the same order as `alphas`.
    """
    out: list[CoverageResult] = []
    for a in alphas:
        qhat = compute_qhat(cal_scores, a, finite_sample_correction)
        result = empirical_coverage(test_p_high, test_y, qhat)
        result["alpha"] = float(a)
        out.append(result)
    return out


def nonconformity_from_p(p_high: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Standard nonconformity score for binary classification with calibrated
    probabilities: 1 - probability assigned to the true class.
    """
    p = np.asarray(p_high, dtype=float).ravel()
    y = np.asarray(y_true).ravel()
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    p_true = np.where(y == 1, p, 1.0 - p)
    return 1.0 - p_true
