"""Phase 2.5 - tests for ml.splits.stratified_split / split_summary."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml.splits import split_summary, stratified_split


def test_three_partitions_cover_all_indices() -> None:
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1] * 20)
    sp = stratified_split(y, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)
    union = set(sp["train"].tolist()) | set(sp["val"].tolist()) | set(sp["test"].tolist())
    assert union == set(range(len(y)))
    assert not (set(sp["train"].tolist()) & set(sp["val"].tolist()))
    assert not (set(sp["train"].tolist()) & set(sp["test"].tolist()))
    assert not (set(sp["val"].tolist()) & set(sp["test"].tolist()))


def test_partition_sizes_close_to_requested_fractions() -> None:
    y = np.array([0, 1] * 100)
    sp = stratified_split(y, 0.6, 0.2, 0.2, seed=0)
    n = len(y)
    assert abs(len(sp["train"]) / n - 0.6) < 0.05
    assert abs(len(sp["val"]) / n - 0.2) < 0.05
    assert abs(len(sp["test"]) / n - 0.2) < 0.05


def test_class_balance_preserved_when_no_groups() -> None:
    y = np.array([0] * 80 + [1] * 20)
    sp = stratified_split(y, 0.6, 0.2, 0.2, seed=42)
    target = float(y.mean())
    for name in ("train", "val", "test"):
        assert abs(float(y[sp[name]].mean()) - target) <= 0.05


def test_groups_disjoint_when_groups_provided() -> None:
    y = np.array([0, 1] * 50)
    groups = np.repeat(np.arange(20), 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = stratified_split(y, 0.6, 0.2, 0.2, groups=groups, seed=42)
    g_train = set(groups[sp["train"]].tolist())
    g_val = set(groups[sp["val"]].tolist())
    g_test = set(groups[sp["test"]].tolist())
    assert not (g_train & g_val)
    assert not (g_train & g_test)
    assert not (g_val & g_test)


def test_seed_is_deterministic() -> None:
    y = np.array([0, 1] * 50)
    a = stratified_split(y, 0.6, 0.2, 0.2, seed=7)
    b = stratified_split(y, 0.6, 0.2, 0.2, seed=7)
    np.testing.assert_array_equal(a["train"], b["train"])
    np.testing.assert_array_equal(a["val"], b["val"])
    np.testing.assert_array_equal(a["test"], b["test"])


def test_different_seeds_produce_different_splits() -> None:
    y = np.array([0, 1] * 50)
    a = stratified_split(y, 0.6, 0.2, 0.2, seed=1)
    b = stratified_split(y, 0.6, 0.2, 0.2, seed=2)
    assert not np.array_equal(a["train"], b["train"])


def test_fractions_must_sum_to_one() -> None:
    y = np.array([0, 1] * 50)
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(y, 0.5, 0.2, 0.2, seed=0)


def test_fractions_must_be_in_zero_one() -> None:
    y = np.array([0, 1] * 50)
    with pytest.raises(ValueError):
        stratified_split(y, 0.0, 0.5, 0.5, seed=0)


def test_too_few_samples_raises() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        stratified_split(np.array([0, 1]), 0.6, 0.2, 0.2, seed=0)


def test_groups_length_must_match() -> None:
    y = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="groups length"):
        stratified_split(y, 0.6, 0.2, 0.2, groups=np.arange(5), seed=0)


def test_split_summary_reports_n_pos_pos_rate() -> None:
    y = np.array([0] * 30 + [1] * 10)
    sp = stratified_split(y, 0.5, 0.25, 0.25, seed=42)
    summ = split_summary(y, sp)
    assert set(summ.keys()) == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        s = summ[name]
        assert s["n"] == len(sp[name])
        assert s["n_pos"] == int(y[sp[name]].sum())
        assert 0.0 <= s["pos_rate"] <= 1.0


def test_group_split_warns_when_class_drift_large() -> None:
    """When groups force class imbalance across partitions, the warning must fire."""
    n_per_group = 8
    n_groups = 12
    # Create strongly correlated group→label structure so the disjoint split forces drift.
    y_chunks = []
    g_chunks = []
    for gi in range(n_groups):
        group_label = 1 if gi < 4 else 0  # 4 of 12 groups are positive
        y_chunks.append(np.full(n_per_group, group_label, dtype=int))
        g_chunks.append(np.full(n_per_group, gi, dtype=int))
    y = np.concatenate(y_chunks)
    groups = np.concatenate(g_chunks)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=UserWarning)
        stratified_split(y, 0.5, 0.25, 0.25, groups=groups, seed=0)
    # Either the warning fires, or the random draw happened to land balanced - both are valid;
    # the assertion only checks the warning is *possible*, not always emitted.
    has_drift = any("drifts from marginal" in str(w.message) for w in caught)
    # Soft assertion: with extreme group→label correlation, drift is overwhelmingly likely.
    if not has_drift:
        # Try a few seeds; at least one should warn.
        for s in (1, 2, 3, 4, 5):
            with warnings.catch_warnings(record=True) as caught_s:
                warnings.simplefilter("always", category=UserWarning)
                stratified_split(y, 0.5, 0.25, 0.25, groups=groups, seed=s)
            if any("drifts from marginal" in str(w.message) for w in caught_s):
                has_drift = True
                break
    assert has_drift, "Expected drift warning under group-disjoint constraint with correlated labels"
