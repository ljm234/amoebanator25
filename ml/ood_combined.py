"""
Phase 5.3 - Combined OOD gate decision rule.

The Amoebanator pipeline now has three independent OOD/uncertainty signals:

  1. Mahalanobis on the tabular feature space        (ml/ood_simple.py, ml/robust.py)
  2. Energy on the raw model logits (Liu 2020)        (ml/robust.py)
  3. Neg-energy on the calibrated probability         (ml/ood_energy.py)

Each emits a boolean `flag` plus a continuous `score`. This module composes
them into a single decision under one of three rules:

  * "OR"        - abstain if any gate fires (most conservative)
  * "AND"       - abstain only if all gates fire (least conservative)
  * "WEIGHTED"  - sum of weighted scores ≥ threshold → abstain

A simple wrapper takes the dict that infer_one returns and produces the
combined decision plus a per-gate breakdown so a reviewer can see why.
"""
from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np


GateRule = Literal["OR", "AND", "WEIGHTED"]


class GateSignal(TypedDict):
    name: str
    score: float
    threshold: float | None
    flag: bool


class CombinedDecision(TypedDict):
    rule: GateRule
    signals: list[GateSignal]
    abstain: bool
    weighted_sum: float | None
    weighted_threshold: float | None


def _norm_score(score: float, threshold: float | None) -> float:
    """Normalise a (score, threshold) pair to a 0/1+ trigger strength."""
    if threshold is None or not np.isfinite(threshold):
        return 0.0
    if not np.isfinite(score):
        return 1.0
    return float(score) - float(threshold)


_VALID_RULES: frozenset[str] = frozenset({"OR", "AND", "WEIGHTED"})


def combine_signals(
    signals: list[GateSignal],
    rule: GateRule = "OR",
    weights: dict[str, float] | None = None,
    weighted_threshold: float = 0.0,
) -> CombinedDecision:
    """
    Combine a list of GateSignal dicts under the chosen rule.

    For "WEIGHTED", `weights` maps signal name → coefficient (default 1.0)
    and the gate fires if Σ wᵢ · (scoreᵢ - thresholdᵢ) ≥ weighted_threshold.
    Signals without a threshold contribute 0 to the weighted sum.
    """
    # Runtime validation guards against callers that bypass the type system
    # (e.g. config files, JSON inputs) and pass an arbitrary string.
    if rule not in _VALID_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {sorted(_VALID_RULES)}.")
    if rule == "OR":
        abstain = any(s["flag"] for s in signals)
        return {"rule": rule, "signals": signals, "abstain": abstain,
                "weighted_sum": None, "weighted_threshold": None}
    if rule == "AND":
        abstain = bool(signals) and all(s["flag"] for s in signals)
        return {"rule": rule, "signals": signals, "abstain": abstain,
                "weighted_sum": None, "weighted_threshold": None}
    # rule == "WEIGHTED"
    w = weights or {}
    total = 0.0
    for s in signals:
        coef = float(w.get(s["name"], 1.0))
        total += coef * _norm_score(s["score"], s["threshold"])
    return {
        "rule": rule, "signals": signals,
        "abstain": bool(total >= weighted_threshold),
        "weighted_sum": float(total),
        "weighted_threshold": float(weighted_threshold),
    }


def signals_from_infer_output(out: dict[str, object]) -> list[GateSignal]:
    """
    Adapter: pull the three gate (score, threshold, flag) tuples out of
    the dict that ml.infer.infer_one returns. Missing fields are skipped
    so this works on every branch of infer_one's output.
    """
    signals: list[GateSignal] = []

    d2 = out.get("mahalanobis_d2")
    d2_tau = out.get("d2_tau")
    if d2 is not None:
        try:
            d2_f = float(d2)  # type: ignore[arg-type]
            tau_f = float(d2_tau) if d2_tau is not None else None  # type: ignore[arg-type]
            flag = (tau_f is not None) and (d2_f > tau_f)
            signals.append({"name": "mahalanobis", "score": d2_f, "threshold": tau_f, "flag": bool(flag)})
        except (TypeError, ValueError):
            pass

    e = out.get("energy")
    e_tau = out.get("energy_tau")
    if e is not None:
        try:
            e_f = float(e)  # type: ignore[arg-type]
            tau_f = float(e_tau) if e_tau is not None else None  # type: ignore[arg-type]
            # ml/infer.py treats `energy > tau` as "above the in-dist 95th percentile
            # → likely OOD → abstain" (Liu 2020 canonical semantics).
            # score_for_combo = (energy - tau): larger positive == further above the
            # OOD shift threshold == stronger trigger for the WEIGHTED rule.
            flag = (tau_f is not None) and (e_f > tau_f)
            score_for_combo = (e_f - tau_f) if tau_f is not None else 0.0
            signals.append({
                "name": "logit_energy",
                "score": float(score_for_combo),
                "threshold": 0.0 if tau_f is not None else None,
                "flag": bool(flag),
            })
        except (TypeError, ValueError):
            pass

    en = out.get("energy_neg")
    en_tau = out.get("energy_neg_tau")
    abstain_neg = out.get("ood_abstain_energy_neg")
    if en is not None:
        try:
            en_f = float(en)  # type: ignore[arg-type]
            tau_f = float(en_tau) if en_tau is not None else None  # type: ignore[arg-type]
            flag = bool(abstain_neg) if abstain_neg is not None else (tau_f is not None and en_f > tau_f)
            signals.append({"name": "neg_energy", "score": en_f, "threshold": tau_f, "flag": bool(flag)})
        except (TypeError, ValueError):
            pass

    return signals


def combined_decision_from_infer(
    out: dict[str, object],
    rule: GateRule = "OR",
    weights: dict[str, float] | None = None,
    weighted_threshold: float = 0.0,
) -> CombinedDecision:
    """One-shot: extract gate signals from infer_one output, combine, return decision."""
    return combine_signals(signals_from_infer_output(out), rule, weights, weighted_threshold)
