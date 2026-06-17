"""
Bootstrap confidence intervals for any metric over (y_true, y_score) pairs.

Phase 3.5. Used by every baseline + ablation report so headline numbers always
ship with a 95% CI rather than a point estimate.

Implementation notes:
  * Default n_resamples = 2000, alpha = 0.05 → percentile 95% CI.
  * Stratified resampling preserves the marginal class balance, which matters
    a lot at low prevalence (the unstratified bootstrap would give degenerate
    samples with no positives).
  * Failed metric calls (e.g. AUC undefined when one class is missing in a
    resample) are skipped, not silently zeroed; if more than half of the
    resamples fail we raise rather than report a bogus interval.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np


class BootstrapResult(TypedDict):
    point: float
    mean: float
    std: float
    lo: float
    hi: float
    alpha: float
    n_resamples: int
    n_skipped: int


def _stratified_resample_indices(
    y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample indices with replacement, preserving per-class counts."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.integers(0, len(y), size=len(y))
    pos_b = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    neg_b = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([pos_b, neg_b])


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    stratified: bool = True,
    seed: int = 0,
) -> BootstrapResult:
    """
    Percentile bootstrap CI for `metric_fn(y_true, y_score)`.

    Returns lo/hi at the (alpha/2, 1-alpha/2) quantiles, plus the point
    estimate computed on the full sample. Stratified by default - flip
    to False only if you want the unstratified Efron bootstrap.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must lie in (0, 0.5); got {alpha!r}.")
    if n_resamples < 100:
        raise ValueError(f"n_resamples should be >= 100; got {n_resamples!r}.")
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))
    boot: list[float] = []
    skipped = 0
    for _ in range(n_resamples):
        if stratified:
            idx = _stratified_resample_indices(y_true, rng)
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            val = float(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            skipped += 1
            continue
        if not np.isfinite(val):
            skipped += 1
            continue
        boot.append(val)

    if len(boot) < n_resamples // 2:
        raise RuntimeError(
            f"More than half of bootstrap resamples failed "
            f"({skipped}/{n_resamples}). Metric is likely undefined "
            f"under the resampling scheme."
        )
    arr = np.asarray(boot, dtype=float)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return {
        "point": point,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_skipped": skipped,
    }


def bootstrap_ci_paired(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    stratified: bool = True,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap CI for the *difference* metric_fn(b) - metric_fn(a)."""
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must lie in (0, 0.5); got {alpha!r}.")
    y_true = np.asarray(y_true)
    a = np.asarray(y_score_a)
    b = np.asarray(y_score_b)
    if y_true.shape != a.shape or y_true.shape != b.shape:
        raise ValueError("y_true, y_score_a, y_score_b must share shape")
    rng = np.random.default_rng(seed)
    point_diff = float(metric_fn(y_true, b)) - float(metric_fn(y_true, a))
    boot: list[float] = []
    skipped = 0
    for _ in range(n_resamples):
        if stratified:
            idx = _stratified_resample_indices(y_true, rng)
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            d = float(metric_fn(y_true[idx], b[idx])) - float(metric_fn(y_true[idx], a[idx]))
        except Exception:
            skipped += 1
            continue
        if not np.isfinite(d):
            skipped += 1
            continue
        boot.append(d)
    if len(boot) < n_resamples // 2:
        raise RuntimeError(
            f"More than half of paired bootstrap resamples failed ({skipped}/{n_resamples})."
        )
    arr = np.asarray(boot, dtype=float)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return {
        "point": point_diff,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_skipped": skipped,
    }
