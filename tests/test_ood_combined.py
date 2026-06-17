"""Phase 5.3 - tests for ml.ood_combined."""
from __future__ import annotations

import pytest

from ml.ood_combined import (
    combine_signals,
    combined_decision_from_infer,
    signals_from_infer_output,
)


def _signals(*flags: bool) -> list[dict]:
    return [
        {"name": f"gate_{i}", "score": float(i + 1), "threshold": 0.0, "flag": bool(f)}
        for i, f in enumerate(flags)
    ]


def test_or_rule_fires_if_any_flag() -> None:
    out = combine_signals(_signals(False, True, False), rule="OR")
    assert out["abstain"] is True
    out = combine_signals(_signals(False, False, False), rule="OR")
    assert out["abstain"] is False


def test_and_rule_fires_only_if_all_flags() -> None:
    assert combine_signals(_signals(True, True, True), rule="AND")["abstain"] is True
    assert combine_signals(_signals(True, True, False), rule="AND")["abstain"] is False


def test_and_rule_with_no_signals_returns_false() -> None:
    out = combine_signals([], rule="AND")
    assert out["abstain"] is False


def test_weighted_rule_sums_normalized_scores() -> None:
    sigs = [
        {"name": "a", "score": 5.0, "threshold": 1.0, "flag": True},
        {"name": "b", "score": 2.0, "threshold": 4.0, "flag": False},
    ]
    out = combine_signals(sigs, rule="WEIGHTED", weighted_threshold=0.0)
    # contribution: (5-1) + (2-4) = 2 → above 0 → abstain
    assert out["abstain"] is True
    assert out["weighted_sum"] == pytest.approx(2.0)


def test_weighted_rule_respects_weights_dict() -> None:
    sigs = [
        {"name": "a", "score": 5.0, "threshold": 1.0, "flag": True},
        {"name": "b", "score": 2.0, "threshold": 4.0, "flag": False},
    ]
    out = combine_signals(sigs, rule="WEIGHTED", weights={"a": 0.5, "b": 1.0}, weighted_threshold=0.0)
    # contribution: 0.5 * 4 + 1.0 * (-2) = 0 → not strictly above 0
    assert out["weighted_sum"] == pytest.approx(0.0)
    assert out["abstain"] is True  # >= 0


def test_invalid_rule_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        combine_signals([], rule="XOR")  # type: ignore[arg-type]


def test_signals_from_infer_output_extracts_three_gates() -> None:
    out = {
        "mahalanobis_d2": 19.0, "d2_tau": 26.7,
        "energy": -9.84, "energy_tau": 86.24,
        "energy_neg": -3.0, "energy_neg_tau": -1e-8,
        "ood_abstain_energy_neg": False,
    }
    sigs = signals_from_infer_output(out)
    names = {s["name"] for s in sigs}
    assert names == {"mahalanobis", "logit_energy", "neg_energy"}


def test_signals_skip_missing_fields() -> None:
    out = {"prediction": "ABSTAIN", "reason": "OOD", "p_high": 0.0}
    sigs = signals_from_infer_output(out)
    assert sigs == []


def test_signals_handle_garbage_values_gracefully() -> None:
    out = {"mahalanobis_d2": "not-a-number", "d2_tau": "also-bad"}
    sigs = signals_from_infer_output(out)
    assert len(sigs) == 0


def test_combined_decision_end_to_end() -> None:
    out = {
        "mahalanobis_d2": 19.0, "d2_tau": 26.7,        # not OOD (d2 < tau)
        "energy": -50.0, "energy_tau": -0.99,           # NOT LogitEnergyAboveOODShift (energy << tau, well below the OOD ceiling)
        "energy_neg": -18.4, "energy_neg_tau": -1e-8,  # confident (very negative)
        "ood_abstain_energy_neg": False,
    }
    decision = combined_decision_from_infer(out, rule="OR")
    assert decision["abstain"] is False
    assert len(decision["signals"]) == 3


def test_logit_energy_flag_inverted_correctly() -> None:
    """ml.infer treats `energy > tau` as LogitEnergyAboveOODShift abstain - the adapter should preserve that."""
    out = {"energy": 5.0, "energy_tau": -0.99}
    sigs = signals_from_infer_output(out)
    le = next(s for s in sigs if s["name"] == "logit_energy")
    assert le["flag"] is True  # energy=5.0 > tau=-0.99 → flag should fire (above OOD shift)
