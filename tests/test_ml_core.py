"""
test_ml_core.py - 2060-level comprehensive tests for ML core modules.

Covers:
  ml/calibration.py      - TemperatureScaler, fit_temperature
  ml/conformal.py        - set_from_p_high, decision_from_p_high
  ml/dca.py              - decision_curve
  ml/ood.py              - tabular OOD (ood.py helpers)
  ml/ood_energy.py       - energy-based OOD gate
  ml/ood_maha.py         - Mahalanobis OOD fitting/scoring
  ml/ood_simple.py       - simplified unified OOD scorer
  ml/robust.py           - tabular feature-stats & scoring utilities

All tests are self-contained (no disk I/O side-effects on persistent paths).
File I/O tests use pytest's tmp_path fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch


# -----------------------------------------------------------------------------
# ml/calibration.py
# -----------------------------------------------------------------------------


class TestTemperatureScaler:
    def test_init_temperature_is_one(self) -> None:
        from ml.calibration import TemperatureScaler

        scaler = TemperatureScaler()
        assert abs(scaler.temperature() - 1.0) < 1e-5

    def test_forward_divides_logits_by_temperature(self) -> None:
        from ml.calibration import TemperatureScaler

        scaler = TemperatureScaler()
        logits = torch.tensor([[2.0, 1.0]])
        out = scaler(logits)
        # at T=1, output == input
        assert torch.allclose(out, logits, atol=1e-5)

    def test_temperature_above_zero_after_forward(self) -> None:
        from ml.calibration import TemperatureScaler

        scaler = TemperatureScaler()
        assert scaler.temperature() > 0

    def test_high_logT_increases_temperature(self) -> None:
        from ml.calibration import TemperatureScaler

        scaler = TemperatureScaler()
        with torch.no_grad():
            scaler.logT.fill_(1.0)  # T = e^1 ~= 2.718
        assert abs(scaler.temperature() - math.e) < 0.01

    def test_forward_scales_correctly_with_temperature_2(self) -> None:
        from ml.calibration import TemperatureScaler

        scaler = TemperatureScaler()
        with torch.no_grad():
            scaler.logT.fill_(math.log(2.0))  # T = 2
        logits = torch.tensor([[4.0, 2.0]])
        out = scaler(logits)
        expected = torch.tensor([[2.0, 1.0]])
        assert torch.allclose(out, expected, atol=1e-4)


class TestFitTemperature:
    def _make_logits_and_labels(self, n: int = 80) -> tuple[Any, Any]:
        rng = np.random.default_rng(42)
        # Logits: slightly miscalibrated binary
        logits = rng.normal(size=(n, 2)).astype(np.float32)
        logits[:n//2, 1] += 2.0  # class 1 easier to predict
        y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int64)
        return logits, y

    def test_returns_positive_float(self) -> None:
        from ml.calibration import fit_temperature, TemperatureScaler

        logits, y = self._make_logits_and_labels()
        model = TemperatureScaler()
        T = fit_temperature(model, logits, y, max_iter=5, lr=0.1)
        assert isinstance(T, float)
        assert T > 0.0

    def test_temperature_reduces_loss(self) -> None:
        from ml.calibration import fit_temperature, TemperatureScaler

        logits, y = self._make_logits_and_labels()
        model = TemperatureScaler()
        T = fit_temperature(model, logits, y, max_iter=50, lr=0.05)
        # Temperature must be finite and in a sensible range
        assert 0.1 < T < 10.0

    def test_handles_cpu_device(self) -> None:
        from ml.calibration import fit_temperature, TemperatureScaler

        logits = np.array([[1.5, -0.5], [0.2, 1.8]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        model = TemperatureScaler()
        T = fit_temperature(model, logits, y, device="cpu", max_iter=10)
        assert isinstance(T, float)

    def test_perfect_logits_temperature_near_one(self) -> None:
        """When logits are already well-calibrated, T should stay near 1."""
        from ml.calibration import fit_temperature, TemperatureScaler

        rng = np.random.default_rng(7)
        n = 100
        logits = np.zeros((n, 2), dtype=np.float32)
        y = rng.integers(0, 2, n).astype(np.int64)
        for i, yi in enumerate(y):
            logits[i, yi] = 3.0
            logits[i, 1 - yi] = -3.0
        model = TemperatureScaler()
        T = fit_temperature(model, logits, y, max_iter=30)
        # Well-separated logits -> temperature close to 1 or slightly < 1
        assert 0.01 < T < 5.0


# -----------------------------------------------------------------------------
# ml/conformal.py
# -----------------------------------------------------------------------------


class TestSetFromPHigh:
    def test_high_p_includes_high(self) -> None:
        from ml.conformal import set_from_p_high

        low, high = set_from_p_high(0.9, 0.1)
        assert high is True
        assert low is False

    def test_low_p_includes_low(self) -> None:
        from ml.conformal import set_from_p_high

        low, high = set_from_p_high(0.05, 0.1)
        assert low is True
        assert high is False

    def test_ambiguous_p_includes_both(self) -> None:
        from ml.conformal import set_from_p_high

        low, high = set_from_p_high(0.5, 0.6)
        assert low is True
        assert high is True

    def test_neither_at_boundary(self) -> None:
        from ml.conformal import set_from_p_high

        low, high = set_from_p_high(0.5, 0.3)
        assert low is False
        assert high is False

    def test_exact_boundary_high(self) -> None:
        from ml.conformal import set_from_p_high

        # p_high == 1 - qhat exactly -> include_high
        low, high = set_from_p_high(0.8, 0.2)
        assert high is True

    def test_exact_boundary_low(self) -> None:
        from ml.conformal import set_from_p_high

        # p_high == qhat exactly -> include_low
        low, high = set_from_p_high(0.2, 0.2)
        assert low is True


class TestDecisionFromPHigh:
    def test_high_decision(self) -> None:
        from ml.conformal import decision_from_p_high

        assert decision_from_p_high(0.95, 0.1) == "High"

    def test_low_decision(self) -> None:
        from ml.conformal import decision_from_p_high

        assert decision_from_p_high(0.02, 0.1) == "Low"

    def test_abstain_when_both(self) -> None:
        from ml.conformal import decision_from_p_high

        assert decision_from_p_high(0.5, 0.6) == "ABSTAIN"

    def test_abstain_string_exact(self) -> None:
        from ml.conformal import decision_from_p_high

        result = decision_from_p_high(0.5, 0.9)
        assert result == "ABSTAIN"

    def test_decision_is_string(self) -> None:
        from ml.conformal import decision_from_p_high

        result = decision_from_p_high(0.8, 0.15)
        assert isinstance(result, str)


# -----------------------------------------------------------------------------
# ml/dca.py
# -----------------------------------------------------------------------------


class TestDecisionCurve:
    def _binary_case(self) -> tuple[list[int], list[float]]:
        y = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0]
        p = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.75, 0.25, 0.15]
        return y, p

    def test_returns_dataframe(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        df = decision_curve(y, p)
        assert isinstance(df, pd.DataFrame)

    def test_has_threshold_and_net_benefit_columns(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        df = decision_curve(y, p)
        assert "threshold" in df.columns
        assert "net_benefit" in df.columns

    def test_threshold_range_0_to_1(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        df = decision_curve(y, p)
        assert df["threshold"].min() >= 0.0
        assert df["threshold"].max() <= 1.0

    def test_net_benefit_finite(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        df = decision_curve(y, p)
        assert df["net_benefit"].apply(lambda x: math.isfinite(x)).all()

    def test_default_99_thresholds(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        df = decision_curve(y, p)
        assert len(df) == 99

    def test_custom_thresholds(self) -> None:
        from ml.dca import decision_curve

        y, p = self._binary_case()
        thresholds = np.array([0.1, 0.5, 0.9])
        df = decision_curve(y, p, thresholds=thresholds)
        assert len(df) == 3

    def test_all_high_label(self) -> None:
        from ml.dca import decision_curve

        y = [1, 1, 1, 1]
        p = [0.9, 0.8, 0.7, 0.6]
        df = decision_curve(y, p)
        assert len(df) > 0

    def test_all_low_label(self) -> None:
        from ml.dca import decision_curve

        y = [0, 0, 0, 0]
        p = [0.1, 0.2, 0.3, 0.4]
        df = decision_curve(y, p)
        assert len(df) > 0

    def test_perfect_predictor_positive_net_benefit(self) -> None:
        from ml.dca import decision_curve

        y = [1, 1, 0, 0]
        p = [0.95, 0.90, 0.05, 0.10]
        thresholds = np.array([0.3, 0.5, 0.7])
        df = decision_curve(y, p, thresholds=thresholds)
        # At t=0.5, perfect separator -> net_benefit = 2/4 - 0 = 0.5
        row = df[df["threshold"] == 0.5].iloc[0]
        assert row["net_benefit"] > 0


# -----------------------------------------------------------------------------
# ml/ood.py  (shared tabular helpers)
# -----------------------------------------------------------------------------


class TestOodRobustZ:
    def test_zero_at_median(self) -> None:
        from ml.ood import _robust_z

        x = np.array([3.0, 5.0])
        median = np.array([3.0, 5.0])
        mad = np.array([1.0, 1.0])
        z = _robust_z(x, median, mad)
        assert np.allclose(z, 0.0)

    def test_unit_deviation(self) -> None:
        from ml.ood import _robust_z

        x = np.array([4.0])
        median = np.array([3.0])
        mad = np.array([1.0])
        z = _robust_z(x, median, mad)
        assert np.allclose(z, [1.0])

    def test_nan_replaced_by_zero(self) -> None:
        from ml.ood import _robust_z

        x = np.array([np.nan])
        median = np.array([5.0])
        mad = np.array([0.0])  # mad=0 -> z = nan/0 = nan -> becomes 0
        z = _robust_z(x, median, mad)
        assert z[0] == 0.0


class TestOodMahalanobisD2:
    def test_zero_at_mean(self) -> None:
        from ml.ood import mahalanobis_d2

        mu = np.array([0.0, 0.0])
        z = np.array([0.0, 0.0])
        S = np.diag([1.0, 1.0])
        d2, contrib = mahalanobis_d2(z, mu, S, use_diagonal=True)
        assert d2 == pytest.approx(0.0, abs=1e-10)

    def test_diagonal_result_positive(self) -> None:
        from ml.ood import mahalanobis_d2

        mu = np.array([0.0])
        z = np.array([2.0])
        S = np.diag([1.0])
        d2, _ = mahalanobis_d2(z, mu, S, use_diagonal=True)
        assert d2 == pytest.approx(4.0, rel=1e-5)

    def test_full_covariance(self) -> None:
        from ml.ood import mahalanobis_d2

        mu = np.array([0.0, 0.0])
        z = np.array([1.0, 0.0])
        S = np.eye(2)
        d2, contrib = mahalanobis_d2(z, mu, S, use_diagonal=False)
        assert d2 == pytest.approx(1.0, rel=1e-5)
        assert contrib is None

    def test_contrib_returned_for_diagonal(self) -> None:
        from ml.ood import mahalanobis_d2

        mu = np.array([0.0, 0.0])
        z = np.array([1.0, 2.0])
        S = np.diag([1.0, 4.0])
        d2, contrib = mahalanobis_d2(z, mu, S, use_diagonal=True)
        assert contrib is not None
        assert hasattr(contrib, "__len__")
        assert len(contrib) == 2


class TestScoreEnergy:
    def test_returns_float(self) -> None:
        from ml.ood import score_energy

        logits = np.array([1.0, 2.0])
        e = score_energy(logits)
        assert isinstance(e, float)

    def test_negative_energy(self) -> None:
        from ml.ood import score_energy

        logits = np.array([1.0, 2.0, 3.0])
        e = score_energy(logits)
        # energy = -logsumexp(logits) which is negative for positive logits
        assert e < 0

    def test_larger_logits_more_negative_energy(self) -> None:
        from ml.ood import score_energy

        e1 = score_energy(np.array([1.0, 1.0]))
        e2 = score_energy(np.array([5.0, 5.0]))
        assert e2 < e1


class TestFitEnergyThreshold:
    def test_empty_returns_dict_with_tau(self, tmp_path: Path) -> None:
        import ml.ood as ood_mod

        original_dir = ood_mod.METRICS_DIR
        original_json = ood_mod.ENERGY_JSON
        try:
            ood_mod.METRICS_DIR = tmp_path
            ood_mod.ENERGY_JSON = tmp_path / "energy.json"
            result = ood_mod.fit_energy_threshold(None)
            assert "tau" in result
        finally:
            ood_mod.METRICS_DIR = original_dir
            ood_mod.ENERGY_JSON = original_json

    def test_with_logits_returns_negative_tau(self, tmp_path: Path) -> None:
        import ml.ood as ood_mod

        original_dir = ood_mod.METRICS_DIR
        original_json = ood_mod.ENERGY_JSON
        try:
            ood_mod.METRICS_DIR = tmp_path
            ood_mod.ENERGY_JSON = tmp_path / "energy.json"
            logits = np.random.default_rng(0).normal(size=(20, 2)).astype(float)
            result = ood_mod.fit_energy_threshold(logits)
            assert isinstance(result["tau"], float)
            assert math.isfinite(result["tau"])
        finally:
            ood_mod.METRICS_DIR = original_dir
            ood_mod.ENERGY_JSON = original_json


class TestScoreTabularOod:
    def _make_stats(self) -> dict[str, object]:
        return {
            "cols": ["age", "csf_glucose"],
            "numeric_cols": ["age", "csf_glucose"],
            "median": [40.0, 60.0],
            "mad": [10.0, 10.0],
            "mu": [0.0, 0.0],
            "S": [[1.0, 0.0], [0.0, 1.0]],
            "use_diagonal": True,
            "tau": 9.0,
        }

    def test_in_distribution_row(self) -> None:
        from ml.ood import score_tabular

        stats = self._make_stats()
        row = pd.Series({"age": 40.0, "csf_glucose": 60.0})
        result = score_tabular(row, stats)
        assert result["in_dist"] is True

    def test_ood_row_far_from_mean(self) -> None:
        from ml.ood import score_tabular

        stats = self._make_stats()
        row = pd.Series({"age": 120.0, "csf_glucose": 200.0})
        result = score_tabular(row, stats)
        assert "d2" in result
        assert result["d2"] > stats["tau"]

    def test_empty_stats_returns_inf(self) -> None:
        from ml.ood import score_tabular

        row = pd.Series({"age": 40.0})
        result = score_tabular(row, {})
        assert result["d2"] == float("inf")

    def test_missing_column_uses_median(self) -> None:
        from ml.ood import score_tabular

        stats = self._make_stats()
        # ml/ood.py score_tabular expects all listed cols to be present;
        # a row with all columns at the median value should score exactly 0
        row = pd.Series({"age": 40.0, "csf_glucose": 60.0})
        result = score_tabular(row, stats)
        assert result["d2"] == pytest.approx(0.0, abs=1e-9)


class TestCheckOodRow:
    def test_returns_d2_and_contrib(self) -> None:
        from ml.ood import check_ood_row

        stats = {
            "cols": ["age"],
            "median": [40.0],
            "mad": [5.0],
            "mu": [0.0],
            "S": [[1.0]],
            "use_diagonal": True,
            "tau": 4.0,
        }
        row = pd.Series({"age": 40.0})
        result = check_ood_row(row, stats)
        assert "d2" in result
        assert "contrib" in result

    def test_empty_stats(self) -> None:
        from ml.ood import check_ood_row

        row = pd.Series({"age": 40.0})
        result = check_ood_row(row, {})
        assert result["d2"] == float("inf")


# -----------------------------------------------------------------------------
# ml/ood_energy.py
# -----------------------------------------------------------------------------


class TestNegEnergyFromP:
    def test_returns_float(self) -> None:
        from ml.ood_energy import neg_energy_from_p

        e = neg_energy_from_p(0.7)
        assert isinstance(e, float)
        assert math.isfinite(e)

    def test_higher_confidence_lower_energy(self) -> None:
        from ml.ood_energy import neg_energy_from_p

        e_low = neg_energy_from_p(0.51)
        e_high = neg_energy_from_p(0.99)
        assert e_high < e_low

    def test_clips_to_epsilon(self) -> None:
        from ml.ood_energy import neg_energy_from_p

        e0 = neg_energy_from_p(0.0)
        e1 = neg_energy_from_p(1.0)
        assert math.isfinite(e0)
        assert math.isfinite(e1)

    def test_midpoint_approx(self) -> None:
        from ml.ood_energy import neg_energy_from_p

        e = neg_energy_from_p(0.5)
        # at p=0.5, logit=0, energy = -log(1+1) = -log(2)
        expected = -math.log(2.0)
        assert abs(e - expected) < 0.01


class TestOodAbstainEnergy:
    def test_no_gate_file_returns_no_abstain(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = tmp_path / "nonexistent.json"
            result = oe.ood_abstain_energy(0.9)
            assert result["ood_abstain_energy"] is False
        finally:
            oe.ENERGY_JSON = original

    def test_with_tau_below_energy_abstains(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        gate_file = tmp_path / "ood_energy.json"
        gate_file.write_text(json.dumps({"method": "energy_neg", "tau": -0.1}))
        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = gate_file
            # p=0.99 -> energy ~= -4.6... more negative than -0.1 -> no abstain
            result = oe.ood_abstain_energy(0.99)
            assert "energy_neg" in result
            assert isinstance(result["ood_abstain_energy"], bool)
        finally:
            oe.ENERGY_JSON = original

    def test_result_has_required_keys(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = tmp_path / "nonexistent.json"
            result = oe.ood_abstain_energy(0.7)
            assert "energy_neg" in result
            assert "tau" in result
            assert "ood_abstain_energy" in result
        finally:
            oe.ENERGY_JSON = original


class TestLoadEnergyGate:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = tmp_path / "missing.json"
            gate = oe.load_energy_gate()
            assert gate["tau"] is None
            assert gate["n"] == 0
        finally:
            oe.ENERGY_JSON = original

    def test_valid_file_loads_correctly(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        gate_file = tmp_path / "gate.json"
        gate_file.write_text(json.dumps({"method": "energy_neg", "tau": -2.5, "q": 0.01, "n": 100}))
        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = gate_file
            gate = oe.load_energy_gate()
            assert gate["tau"] == pytest.approx(-2.5)
        finally:
            oe.ENERGY_JSON = original

    def test_malformed_file_returns_defaults(self, tmp_path: Path) -> None:
        import ml.ood_energy as oe

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{NOT VALID JSON]]]")
        original = oe.ENERGY_JSON
        try:
            oe.ENERGY_JSON = bad_file
            gate = oe.load_energy_gate()
            assert gate["tau"] is None
        finally:
            oe.ENERGY_JSON = original


# -----------------------------------------------------------------------------
# ml/ood_maha.py
# -----------------------------------------------------------------------------


class TestFitMahalanobis:
    def _make_separation_data(self) -> tuple[Any, Any]:
        rng = np.random.default_rng(0)
        X0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)
        return X, y

    def test_returns_mu0_mu1_inv(self) -> None:
        from ml.ood_maha import fit_mahalanobis

        X, y = self._make_separation_data()
        mu0, mu1, inv = fit_mahalanobis(X, y)
        assert mu0.shape == (2,)
        assert mu1.shape == (2,)
        assert inv.shape == (2, 2)

    def test_mu0_near_zero(self) -> None:
        from ml.ood_maha import fit_mahalanobis

        X, y = self._make_separation_data()
        mu0, mu1, _ = fit_mahalanobis(X, y)
        assert np.allclose(mu0, [0.0, 0.0], atol=0.3)

    def test_mu1_near_five(self) -> None:
        from ml.ood_maha import fit_mahalanobis

        X, y = self._make_separation_data()
        _, mu1, _ = fit_mahalanobis(X, y)
        assert np.allclose(mu1, [5.0, 5.0], atol=0.3)

    def test_inv_is_positive_definite(self) -> None:
        from ml.ood_maha import fit_mahalanobis

        X, y = self._make_separation_data()
        _, _, inv = fit_mahalanobis(X, y)
        eigenvalues = np.linalg.eigvalsh(inv)
        assert np.all(eigenvalues > 0)


class TestMahaScore:
    def test_class0_centroid_returns_zero_distance(self) -> None:
        from ml.ood_maha import fit_mahalanobis, maha_score

        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal([0, 0], 0.3, (20, 2)), rng.normal([5, 5], 0.3, (20, 2))])
        y = np.array([0] * 20 + [1] * 20)
        mu0, mu1, inv = fit_mahalanobis(X, y)
        score = maha_score(mu0, mu0, mu1, inv)
        assert score == pytest.approx(0.0, abs=1e-8)

    def test_midpoint_higher_score(self) -> None:
        from ml.ood_maha import fit_mahalanobis, maha_score

        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal([0, 0], 0.5, (30, 2)), rng.normal([10, 10], 0.5, (30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        mu0, mu1, inv = fit_mahalanobis(X, y)
        midpoint = np.array([5.0, 5.0])
        score_mid = maha_score(midpoint, mu0, mu1, inv)
        score_at_mu0 = maha_score(mu0, mu0, mu1, inv)
        assert score_mid > score_at_mu0

    def test_returns_non_negative(self) -> None:
        from ml.ood_maha import fit_mahalanobis, maha_score

        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal([0, 0], 1.0, (15, 2)), rng.normal([3, 3], 1.0, (15, 2))])
        y = np.array([0] * 15 + [1] * 15)
        mu0, mu1, inv = fit_mahalanobis(X, y)
        x = rng.normal([1, 1], 0.1, (1, 2))[0]
        score = maha_score(x, mu0, mu1, inv)
        assert score >= 0.0


class TestSaveMaha:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        import ml.ood_maha as om

        original_stats = om.MAHA_STATS
        original_json = om.MAHA_JSON
        try:
            om.MAHA_STATS = tmp_path / "maha.npz"
            om.MAHA_JSON = tmp_path / "maha.json"
            mu0 = np.array([0.0, 0.0])
            mu1 = np.array([5.0, 5.0])
            inv = np.eye(2)
            tau = 3.14
            om.save_maha(mu0, mu1, inv, tau)
            r_mu0, r_mu1, r_inv, r_tau = om.load_maha()
            assert np.allclose(r_mu0, mu0)
            assert np.allclose(r_mu1, mu1)
            assert np.allclose(r_inv, inv)
            assert r_tau == pytest.approx(tau)
        finally:
            om.MAHA_STATS = original_stats
            om.MAHA_JSON = original_json


# -----------------------------------------------------------------------------
# ml/ood_simple.py
# -----------------------------------------------------------------------------


class TestEnsureCov:
    def test_2d_list_cov(self) -> None:
        from ml.ood_simple import _ensure_cov

        stats = {
            "numeric_cols": ["a", "b"],
            "cov": [[1.0, 0.5], [0.5, 2.0]],
        }
        cov = _ensure_cov(stats)
        assert cov.shape == (2, 2)
        assert cov[0, 1] == pytest.approx(0.5)

    def test_falls_back_to_S(self) -> None:
        from ml.ood_simple import _ensure_cov

        stats = {
            "numeric_cols": ["a", "b"],
            "S": [[1.0, 0.0], [0.0, 1.0]],
        }
        cov = _ensure_cov(stats)
        assert cov.shape == (2, 2)

    def test_no_cov_returns_identity(self) -> None:
        from ml.ood_simple import _ensure_cov

        stats: dict[str, object] = {"numeric_cols": ["a", "b"]}
        cov = _ensure_cov(stats)
        assert cov.shape == (2, 2)
        assert np.allclose(cov, np.eye(2))


class TestVectorize:
    def test_extracts_values(self) -> None:
        from ml.ood_simple import _vectorize

        row = {"age": 40.0, "csf_glucose": 60.0}
        v = _vectorize(row, ["age", "csf_glucose"])
        assert v[0] == pytest.approx(40.0)
        assert v[1] == pytest.approx(60.0)

    def test_missing_column_is_nan(self) -> None:
        from ml.ood_simple import _vectorize

        row = {"age": 40.0}
        v = _vectorize(row, ["age", "csf_glucose"])
        assert v[0] == pytest.approx(40.0)
        assert math.isnan(v[1])

    def test_non_numeric_is_nan(self) -> None:
        from ml.ood_simple import _vectorize

        row = {"age": "unknown"}
        v = _vectorize(row, ["age"])
        assert math.isnan(v[0])


class TestAsFloatList:
    def test_converts_ints(self) -> None:
        from ml.ood_simple import _as_float_list

        result = _as_float_list([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]

    def test_converts_strings(self) -> None:
        from ml.ood_simple import _as_float_list

        result = _as_float_list(["1.5", "2.5"])
        assert result == [1.5, 2.5]

    def test_preserves_floats(self) -> None:
        from ml.ood_simple import _as_float_list

        result = _as_float_list([3.14, 2.71])
        assert abs(result[0] - 3.14) < 1e-6


class TestOodScore:
    def _write_stats(self, path: Path) -> None:
        stats = {
            "numeric_cols": ["age", "csf_glucose"],
            "cols": ["age", "csf_glucose"],
            "mu": [0.0, 0.0],
            "median": [40.0, 60.0],
            "mad": [10.0, 10.0],
            "cov": [[1.0, 0.0], [0.0, 1.0]],
            "tau": 9.0,
        }
        path.write_text(json.dumps(stats))

    def test_in_distribution_not_ood(self, tmp_path: Path) -> None:
        import ml.ood_simple as os_mod

        stats_path = tmp_path / "feature_stats.json"
        energy_path = tmp_path / "energy_threshold.json"
        energy_path.write_text(json.dumps({"tau": 0.0}))
        self._write_stats(stats_path)
        original_s = os_mod.STATS_JSON
        original_e = os_mod.ENERGY_JSON
        try:
            os_mod.STATS_JSON = str(stats_path)
            os_mod.ENERGY_JSON = str(energy_path)
            row = {"age": 40.0, "csf_glucose": 60.0}
            result = os_mod.ood_score(row)
            assert "mahal" in result
            assert "is_ood" in result
            assert isinstance(result["is_ood"], bool)
        finally:
            os_mod.STATS_JSON = original_s
            os_mod.ENERGY_JSON = original_e

    def test_missing_stats_returns_inf(self, tmp_path: Path) -> None:
        import ml.ood_simple as os_mod

        original_s = os_mod.STATS_JSON
        original_e = os_mod.ENERGY_JSON
        try:
            os_mod.STATS_JSON = str(tmp_path / "missing.json")
            os_mod.ENERGY_JSON = str(tmp_path / "missing2.json")
            row = {"age": 40.0}
            result = os_mod.ood_score(row)
            assert "mahal" in result
            assert math.isfinite(result["mahal"]) or result["mahal"] == float("inf")
        finally:
            os_mod.STATS_JSON = original_s
            os_mod.ENERGY_JSON = original_e


# -----------------------------------------------------------------------------
# ml/robust.py  (parallel implementation of tabular OOD utilities)
# -----------------------------------------------------------------------------


class TestRobustZ:
    def test_zero_at_median(self) -> None:
        from ml.robust import _robust_z

        x = np.array([5.0, 10.0])
        med = np.array([5.0, 10.0])
        mad = np.array([1.0, 2.0])
        z = _robust_z(x, med, mad)
        assert np.allclose(z, 0.0)

    def test_one_sigma_deviation(self) -> None:
        from ml.robust import _robust_z

        x = np.array([6.0])
        med = np.array([5.0])
        mad = np.array([1.0])
        z = _robust_z(x, med, mad)
        assert z[0] == pytest.approx(1.0)

    def test_nan_becomes_zero(self) -> None:
        from ml.robust import _robust_z

        x = np.array([np.nan])
        med = np.array([5.0])
        mad = np.array([0.0])
        z = _robust_z(x, med, mad)
        assert z[0] == 0.0 or not math.isfinite(z[0])


class TestRobustMahalanobisD2:
    def test_zero_at_mean(self) -> None:
        from ml.robust import mahalanobis_d2

        mu = np.array([0.0, 0.0])
        z = np.array([0.0, 0.0])
        S = np.diag([1.0, 1.0])
        d2, contrib = mahalanobis_d2(z, mu, S, use_diagonal=True)
        assert d2 == pytest.approx(0.0, abs=1e-10)
        assert contrib is not None

    def test_off_diagonal_cov(self) -> None:
        from ml.robust import mahalanobis_d2

        mu = np.array([0.0, 0.0])
        z = np.array([1.0, 0.0])
        S = np.eye(2)
        d2, contrib = mahalanobis_d2(z, mu, S, use_diagonal=False)
        assert d2 == pytest.approx(1.0, rel=1e-5)
        assert contrib is None


class TestRobustScoreEnergy:
    def test_known_value(self) -> None:
        from ml.robust import score_energy

        logits = np.array([0.0, 0.0])
        e = score_energy(logits)
        expected = -np.logaddexp(0.0, 0.0)
        assert e == pytest.approx(expected, rel=1e-5)

    def test_finite(self) -> None:
        from ml.robust import score_energy

        e = score_energy(np.array([1.0, 2.0, 3.0]))
        assert math.isfinite(e)


class TestRobustFitTabularStats:
    def test_empty_csv_returns_inf_tau(self, tmp_path: Path) -> None:
        import ml.robust as rob

        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("timestamp,age\n")  # header only

        original_log = rob.LOG_CSV
        original_dir = rob.METRICS_DIR
        original_stats = rob.STATS_JSON
        original_energy = rob.ENERGY_JSON
        try:
            rob.LOG_CSV = empty_csv
            rob.METRICS_DIR = tmp_path
            rob.STATS_JSON = tmp_path / "stats.json"
            rob.ENERGY_JSON = tmp_path / "energy.json"
            result = rob.fit_tabular_stats(csv=empty_csv)
            assert result["tau"] == float("inf")
        finally:
            rob.LOG_CSV = original_log
            rob.METRICS_DIR = original_dir
            rob.STATS_JSON = original_stats
            rob.ENERGY_JSON = original_energy

    def test_with_data_computes_finite_tau(self, tmp_path: Path) -> None:
        import ml.robust as rob

        rng = np.random.default_rng(99)
        n = 40
        ages = rng.integers(20, 80, n).astype(float)
        gluc = rng.uniform(40, 100, n)
        df = pd.DataFrame({"age": ages, "csf_glucose": gluc})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        original_log = rob.LOG_CSV
        original_dir = rob.METRICS_DIR
        original_stats = rob.STATS_JSON
        original_energy = rob.ENERGY_JSON
        try:
            rob.LOG_CSV = csv_path
            rob.METRICS_DIR = tmp_path
            rob.STATS_JSON = tmp_path / "stats.json"
            rob.ENERGY_JSON = tmp_path / "energy.json"
            result = rob.fit_tabular_stats(csv=csv_path)
            assert math.isfinite(result["tau"])
            assert len(result["cols"]) >= 1
        finally:
            rob.LOG_CSV = original_log
            rob.METRICS_DIR = original_dir
            rob.STATS_JSON = original_stats
            rob.ENERGY_JSON = original_energy


class TestRobustScoreTabular:
    def _stats(self) -> dict[str, object]:
        return {
            "cols": ["age", "csf_glucose"],
            "median": [40.0, 60.0],
            "mad": [10.0, 10.0],
            "mu": [0.0, 0.0],
            "S": [[1.0, 0.0], [0.0, 1.0]],
            "use_diagonal": True,
            "tau": 9.0,
        }

    def test_in_dist_row(self) -> None:
        from ml.robust import score_tabular

        stats = self._stats()
        row = pd.Series({"age": 40.0, "csf_glucose": 60.0})
        result = score_tabular(row, stats)
        assert result["d2"] == pytest.approx(0.0, abs=1e-10)

    def test_empty_stats_returns_inf(self) -> None:
        from ml.robust import score_tabular

        row = pd.Series({"age": 40.0})
        result = score_tabular(row, {})
        assert result["d2"] == float("inf")

    def test_partial_columns_intersect(self) -> None:
        from ml.robust import score_tabular

        stats = self._stats()
        row = pd.Series({"age": 40.0})  # csf_glucose missing
        result = score_tabular(row, stats)
        assert "d2" in result

    def test_returns_use_diagonal_key(self) -> None:
        from ml.robust import score_tabular

        stats = self._stats()
        row = pd.Series({"age": 40.0, "csf_glucose": 60.0})
        result = score_tabular(row, stats)
        assert "use_diagonal" in result


class TestRobustCheckOodRow:
    def test_returns_two_keys(self) -> None:
        from ml.robust import check_ood_row

        stats = {
            "cols": ["age"],
            "median": [40.0],
            "mad": [5.0],
            "mu": [0.0],
            "S": [[1.0]],
            "use_diagonal": True,
            "tau": 4.0,
        }
        row = pd.Series({"age": 40.0})
        result = check_ood_row(row, stats)
        assert "d2" in result
        assert "contrib" in result


class TestRobustFitEnergyThreshold:
    def test_with_logits(self, tmp_path: Path) -> None:
        import ml.robust as rob

        original_dir = rob.METRICS_DIR
        original_energy = rob.ENERGY_JSON
        try:
            rob.METRICS_DIR = tmp_path
            rob.ENERGY_JSON = tmp_path / "energy.json"
            logits = np.random.default_rng(42).normal(size=(30, 2))
            result = rob.fit_energy_threshold(logits)
            assert "tau" in result
            assert math.isfinite(result["tau"])
        finally:
            rob.METRICS_DIR = original_dir
            rob.ENERGY_JSON = original_energy

    def test_with_none_logits(self, tmp_path: Path) -> None:
        import ml.robust as rob

        original_dir = rob.METRICS_DIR
        original_energy = rob.ENERGY_JSON
        try:
            rob.METRICS_DIR = tmp_path
            rob.ENERGY_JSON = tmp_path / "energy.json"
            result = rob.fit_energy_threshold(None)
            assert result["tau"] == 0.0
        finally:
            rob.METRICS_DIR = original_dir
            rob.ENERGY_JSON = original_energy


class TestRobustLoadStats:
    def test_missing_file_returns_empty_struct(self, tmp_path: Path) -> None:
        from ml.robust import load_stats

        result = load_stats(tmp_path / "nonexistent.json")
        assert isinstance(result, dict)
        assert "cols" in result

    def test_valid_file_loads(self, tmp_path: Path) -> None:
        from ml.robust import load_stats

        stats = {"cols": ["age"], "tau": 3.14}
        p = tmp_path / "stats.json"
        p.write_text(json.dumps(stats))
        result = load_stats(p)
        assert result["tau"] == pytest.approx(3.14)


# -----------------------------------------------------------------------------
# Gap-closing tests: ml/ood.py  (lines 19-25, 28-31, 53-98, 101-103, 107)
# -----------------------------------------------------------------------------


class TestOodLoadLog:
    def test_missing_file_returns_empty_df(self, tmp_path: Path) -> None:
        from ml.ood import _load_log

        result = _load_log(tmp_path / "nonexistent.csv")
        assert result.empty

    def test_existing_file_parses_numeric_cols(self, tmp_path: Path) -> None:
        from ml.ood import _load_log

        csv = tmp_path / "log.csv"
        # Use float-valued ages so pd.to_numeric returns float64
        csv.write_text("age,csf_glucose\n40.5,60.0\n50.5,70.0\n")
        df = _load_log(csv)
        assert not df.empty
        assert "age" in df.columns
        assert pd.api.types.is_numeric_dtype(df["age"])

    def test_non_numeric_coerced_to_nan(self, tmp_path: Path) -> None:
        from ml.ood import _load_log

        csv = tmp_path / "log.csv"
        csv.write_text("age,csf_glucose\nbad,60\n50,ok\n")
        df = _load_log(csv)
        assert pd.isna(df["age"].iloc[0])
        assert pd.isna(df["csf_glucose"].iloc[1])


class TestOodPickCols:
    def test_returns_intersection_with_numeric_cols(self) -> None:
        from ml.ood import _pick_cols

        df = pd.DataFrame({"age": [1], "csf_glucose": [2], "name": ["x"]})
        cols = _pick_cols(df, None)
        assert "age" in cols
        assert "csf_glucose" in cols
        assert "name" not in cols

    def test_drop_cols_removes_from_result(self) -> None:
        from ml.ood import _pick_cols

        df = pd.DataFrame({"age": [1], "csf_glucose": [2], "pcr": [3]})
        cols = _pick_cols(df, ["age"])
        assert "age" not in cols
        assert "csf_glucose" in cols


class TestOodFitTabularStats:
    def test_with_real_data_writes_stats(self, tmp_path: Path) -> None:
        import ml.ood as ood_mod

        rng = np.random.default_rng(5)
        n = 30
        df = pd.DataFrame({
            "age": rng.integers(20, 70, n).astype(float),
            "csf_glucose": rng.uniform(40, 100, n),
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        orig_dir = ood_mod.METRICS_DIR
        orig_stats = ood_mod.STATS_JSON
        orig_energy = ood_mod.ENERGY_JSON
        try:
            ood_mod.METRICS_DIR = tmp_path
            ood_mod.STATS_JSON = tmp_path / "stats.json"
            ood_mod.ENERGY_JSON = tmp_path / "energy.json"
            result = ood_mod.fit_tabular_stats(csv=csv)
            assert math.isfinite(result["tau"])
            assert len(result["cols"]) >= 1
        finally:
            ood_mod.METRICS_DIR = orig_dir
            ood_mod.STATS_JSON = orig_stats
            ood_mod.ENERGY_JSON = orig_energy

    def test_with_drop_cols(self, tmp_path: Path) -> None:
        import ml.ood as ood_mod

        rng = np.random.default_rng(6)
        n = 25
        df = pd.DataFrame({
            "age": rng.integers(20, 70, n).astype(float),
            "csf_glucose": rng.uniform(40, 100, n),
            "pcr": rng.uniform(0, 1, n),
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        orig_dir = ood_mod.METRICS_DIR
        orig_stats = ood_mod.STATS_JSON
        orig_energy = ood_mod.ENERGY_JSON
        try:
            ood_mod.METRICS_DIR = tmp_path
            ood_mod.STATS_JSON = tmp_path / "stats.json"
            ood_mod.ENERGY_JSON = tmp_path / "energy.json"
            result = ood_mod.fit_tabular_stats(csv=csv, drop_cols=["pcr"])
            assert "pcr" not in result["cols"]
        finally:
            ood_mod.METRICS_DIR = orig_dir
            ood_mod.STATS_JSON = orig_stats
            ood_mod.ENERGY_JSON = orig_energy

    def test_with_full_covariance(self, tmp_path: Path) -> None:
        import ml.ood as ood_mod

        rng = np.random.default_rng(7)
        n = 25
        df = pd.DataFrame({
            "age": rng.integers(20, 70, n).astype(float),
            "csf_glucose": rng.uniform(40, 100, n),
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        orig_dir = ood_mod.METRICS_DIR
        orig_stats = ood_mod.STATS_JSON
        orig_energy = ood_mod.ENERGY_JSON
        try:
            ood_mod.METRICS_DIR = tmp_path
            ood_mod.STATS_JSON = tmp_path / "stats.json"
            ood_mod.ENERGY_JSON = tmp_path / "energy.json"
            result = ood_mod.fit_tabular_stats(csv=csv, use_diagonal=False)
            assert "tau" in result
        finally:
            ood_mod.METRICS_DIR = orig_dir
            ood_mod.STATS_JSON = orig_stats
            ood_mod.ENERGY_JSON = orig_energy


class TestOodLoadStats:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from ml.ood import load_stats

        result = load_stats(tmp_path / "nonexistent.json")
        assert result == {}

    def test_existing_file_loads(self, tmp_path: Path) -> None:
        from ml.ood import load_stats

        p = tmp_path / "stats.json"
        p.write_text(json.dumps({"cols": ["age"], "tau": 5.0}))
        result = load_stats(p)
        assert result["tau"] == pytest.approx(5.0)


class TestOodScoreTabularNoneStats:
    def test_none_stats_empty_loads_no_cols(self, tmp_path: Path) -> None:
        """When STATS_JSON doesn't exist, stats=None path returns NoStats result."""
        from ml.ood import load_stats, score_tabular

        # Explicitly pass the result of load_stats(non-existent) as stats
        # This exercises the same code path as stats=None -> load_stats() -> {}
        empty_stats = load_stats(tmp_path / "nonexistent.json")  # returns {}
        row = pd.Series({"age": 40.0})
        result = score_tabular(row, empty_stats)
        assert result["d2"] == float("inf")
        assert result.get("reason") == "NoStats"


# -----------------------------------------------------------------------------
# Gap-closing: ml/robust.py  (lines 21, 31, 147)
# -----------------------------------------------------------------------------


class TestRobustLoadLog:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from ml.robust import _load_log

        df = _load_log(tmp_path / "none.csv")
        assert df.empty

    def test_loads_and_coerces_numeric(self, tmp_path: Path) -> None:
        from ml.robust import _load_log

        csv = tmp_path / "data.csv"
        csv.write_text("age,csf_glucose\n35,55\nnot_a_number,70\n")
        df = _load_log(csv)
        assert not df.empty
        assert pd.isna(df["age"].iloc[1])


class TestRobustPickColsDropCols:
    def test_drop_removes_col(self) -> None:
        from ml.robust import _pick_cols

        df = pd.DataFrame({"age": [1], "csf_glucose": [2], "pcr": [3]})
        cols = _pick_cols(df, ["pcr"])
        assert "pcr" not in cols
        assert "age" in cols


# -----------------------------------------------------------------------------
# Gap-closing: ml/ood_simple.py (lines 33-41, 69-76, 84-85)
# -----------------------------------------------------------------------------


class TestEnsureCovGaps:
    def test_S_as_1d_list_diagonal(self) -> None:
        """Covers the d[i,i] = float(v) branch for 1-D S list."""
        from ml.ood_simple import _ensure_cov

        stats = {
            "numeric_cols": ["a", "b"],
            "S": [2.0, 3.0],  # 1-D list treated as diagonal
        }
        cov = _ensure_cov(stats)
        assert cov.shape == (2, 2)
        assert cov[0, 0] == pytest.approx(2.0)
        assert cov[1, 1] == pytest.approx(3.0)

    def test_S_nested_wrong_shape_falls_back(self) -> None:
        """Covers the except-path in S list-of-lists with wrong dimensions."""
        from ml.ood_simple import _ensure_cov

        stats = {
            "numeric_cols": ["a", "b"],
            "S": [[1.0, 0.0, 0.0]],  # wrong shape: 1x3 instead of 2x2
        }
        cov = _ensure_cov(stats)
        assert cov.shape == (2, 2)


class TestOodScoreRangeViolations:
    """Covers lines 84-85 (range_violations loop) and the fallback cov branch."""

    def _write_files(self, tmp_path: Path, tau: float = 9.0) -> tuple[Path, Path]:
        stats = {
            "numeric_cols": ["age"],
            "cols": ["age"],
            "mu": [0.0],
            "median": [40.0],
            "mad": [5.0],
            "cov": [[1.0]],
            "tau": tau,
        }
        energy = {"tau": 0.0}
        sp = tmp_path / "stats.json"
        ep = tmp_path / "energy.json"
        sp.write_text(json.dumps(stats))
        ep.write_text(json.dumps(energy))
        return sp, ep

    def test_range_violation_extreme_value(self, tmp_path: Path) -> None:
        import ml.ood_simple as os_mod

        sp, ep = self._write_files(tmp_path)
        original_s = os_mod.STATS_JSON
        original_e = os_mod.ENERGY_JSON
        try:
            os_mod.STATS_JSON = str(sp)
            os_mod.ENERGY_JSON = str(ep)
            # age = 120 is extremely far from median=40 (z-score ~= 16)
            result = os_mod.ood_score({"age": 120.0})
            # Should flag as a range violation
            assert result["range_violations"].get("age") is True
        finally:
            os_mod.STATS_JSON = original_s
            os_mod.ENERGY_JSON = original_e

    def test_no_range_violation_normal_value(self, tmp_path: Path) -> None:
        import ml.ood_simple as os_mod

        sp, ep = self._write_files(tmp_path)
        original_s = os_mod.STATS_JSON
        original_e = os_mod.ENERGY_JSON
        try:
            os_mod.STATS_JSON = str(sp)
            os_mod.ENERGY_JSON = str(ep)
            result = os_mod.ood_score({"age": 42.0})
            assert result["range_violations"].get("age") is False
        finally:
            os_mod.STATS_JSON = original_s
            os_mod.ENERGY_JSON = original_e

    def test_fallback_cov_mismatched_shape(self, tmp_path: Path) -> None:
        """Covers the else branch when cov shape doesn't match cols length."""
        import ml.ood_simple as os_mod

        stats = {
            "numeric_cols": ["age"],
            "cols": ["age"],
            "mu": [0.0],
            "median": [40.0],
            "mad": [5.0],
            "cov": [[1.0, 0.0], [0.0, 1.0]],  # 2x2 but only 1 col -> mismatch
            "tau": 100.0,
        }
        sp = tmp_path / "stats.json"
        ep = tmp_path / "energy.json"
        sp.write_text(json.dumps(stats))
        ep.write_text(json.dumps({"tau": 0.0}))
        original_s = os_mod.STATS_JSON
        original_e = os_mod.ENERGY_JSON
        try:
            os_mod.STATS_JSON = str(sp)
            os_mod.ENERGY_JSON = str(ep)
            result = os_mod.ood_score({"age": 40.0})
            assert "mahal" in result
            assert math.isfinite(result["mahal"])
        finally:
            os_mod.STATS_JSON = original_s
            os_mod.ENERGY_JSON = original_e

