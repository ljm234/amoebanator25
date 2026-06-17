"""Phase 9.5 - tests for ml.seeds."""
from __future__ import annotations

import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ml.seeds import DEFAULT_SEED, SEED_ENV, SeedReport, set_global_seeds


def test_default_seed_is_42() -> None:
    assert DEFAULT_SEED == 42


def test_set_global_seeds_returns_report() -> None:
    report = set_global_seeds(7)
    assert isinstance(report, SeedReport)
    assert report.seed == 7
    assert report.python_random is True
    assert report.numpy is True
    assert report.torch_cpu is True


def test_two_seeds_produce_same_random_sequence() -> None:
    set_global_seeds(42)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5)
    a_torch = torch.rand(5).tolist()
    set_global_seeds(42)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5)
    b_torch = torch.rand(5).tolist()
    assert a_py == b_py
    np.testing.assert_array_equal(a_np, b_np)
    assert a_torch == b_torch


def test_different_seeds_produce_different_sequences() -> None:
    set_global_seeds(1)
    a = torch.rand(20)
    set_global_seeds(2)
    b = torch.rand(20)
    assert not torch.equal(a, b)


def test_env_var_overrides_default() -> None:
    with patch.dict(os.environ, {SEED_ENV: "99"}):
        report = set_global_seeds()
    assert report.seed == 99


def test_explicit_seed_beats_env_var() -> None:
    with patch.dict(os.environ, {SEED_ENV: "99"}):
        report = set_global_seeds(7)
    assert report.seed == 7


def test_invalid_env_var_raises() -> None:
    with patch.dict(os.environ, {SEED_ENV: "not-an-int"}), pytest.raises(ValueError, match="integer"):
        set_global_seeds()


def test_training_is_deterministic_under_pinned_seed() -> None:
    """Two training runs with set_global_seeds() must produce the same val AUC."""
    from ml.training import train_and_save

    set_global_seeds(42)
    out_a = train_and_save()
    set_global_seeds(42)
    out_b = train_and_save()
    assert out_a["auc"] == out_b["auc"]
    assert out_a["T"] == pytest.approx(out_b["T"], rel=1e-5)
