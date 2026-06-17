"""
Stratified train/val/test split with optional group-disjoint constraint.

Phase 2.5. Used by every downstream pipeline (baselines, conformal calibration,
OOD fits) so that no row in the test set ever appears in the training or
calibration sets - including across grouping variables like site or year when
those are available.

The function returns a dict of index arrays so that callers can apply the same
split to multiple feature matrices (raw features, scaled features, embeddings)
without re-randomising.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0):
        raise ValueError(
            f"All fractions must lie in (0, 1); got "
            f"train={train_frac}, val={val_frac}, test={test_frac}."
        )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0 within 1e-6; got sum={total!r}."
        )


def stratified_split(
    y: np.ndarray | Sequence[int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    *,
    groups: np.ndarray | Sequence[Any] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Return a dict {"train", "val", "test"} of integer index arrays.

    When `groups` is None the split is class-stratified using
    StratifiedShuffleSplit. When `groups` is provided the split is
    group-disjoint (GroupShuffleSplit): no group appears in more than one
    partition, so site- or year-leakage is impossible. Group-disjoint splits
    cannot strictly enforce class stratification - we warn if the realised
    class balance drifts more than 5 percentage points from the marginal.

    Reproducibility: `seed` is passed to scikit-learn's RNG.
    """
    _validate_fractions(train_frac, val_frac, test_frac)

    y_arr = np.asarray(y)
    n = len(y_arr)
    if n < 4:
        raise ValueError(f"Need at least 4 samples to make a 3-way split; got {n}.")

    idx_all = np.arange(n)
    if groups is None:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
        train_idx, holdout_idx = next(sss1.split(idx_all, y_arr))
        val_size = val_frac / (val_frac + test_frac)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_size, random_state=seed + 1)
        rel_val_idx, rel_test_idx = next(sss2.split(holdout_idx, y_arr[holdout_idx]))
        val_idx = holdout_idx[rel_val_idx]
        test_idx = holdout_idx[rel_test_idx]
    else:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError(f"groups length {len(groups_arr)} != y length {n}.")
        gss1 = GroupShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
        train_idx, holdout_idx = next(gss1.split(idx_all, y_arr, groups=groups_arr))
        val_size_in_holdout = val_frac / (val_frac + test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_size_in_holdout, random_state=seed + 1)
        rel_val_idx, rel_test_idx = next(
            gss2.split(holdout_idx, y_arr[holdout_idx], groups=groups_arr[holdout_idx])
        )
        val_idx = holdout_idx[rel_val_idx]
        test_idx = holdout_idx[rel_test_idx]

        marginal = float(y_arr.mean())
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            partition_rate = float(y_arr[idx].mean()) if len(idx) else float("nan")
            if not np.isnan(partition_rate) and abs(partition_rate - marginal) > 0.05:
                warnings.warn(
                    f"Group-disjoint split: {name} positive rate {partition_rate:.3f} "
                    f"drifts from marginal {marginal:.3f} by >5pp. "
                    f"Consider increasing the dataset or relaxing the group constraint.",
                    UserWarning,
                    stacklevel=2,
                )

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Internal error: split partitions overlap.")
    if len(train_set) + len(val_set) + len(test_set) != n:
        raise RuntimeError("Internal error: split partitions do not cover all rows.")

    return {
        "train": np.array(sorted(train_idx), dtype=int),
        "val": np.array(sorted(val_idx), dtype=int),
        "test": np.array(sorted(test_idx), dtype=int),
    }


def split_summary(
    y: np.ndarray | Sequence[int],
    splits: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """One-line per-partition summary: n, n_pos, positive rate."""
    y_arr = np.asarray(y)
    out: dict[str, dict[str, float | int]] = {}
    for name, idx in splits.items():
        ys = y_arr[idx]
        out[name] = {
            "n": int(len(ys)),
            "n_pos": int(ys.sum()),
            "pos_rate": float(ys.mean()) if len(ys) else float("nan"),
        }
    return out
