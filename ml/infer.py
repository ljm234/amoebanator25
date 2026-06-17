from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.robust import score_tabular, load_stats, score_energy, ENERGY_JSON
from ml.model import MLP

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

METRICS_DIR: Path = Path("outputs/metrics")
CONF_JSON: Path = METRICS_DIR / "conformal.json"
CONF_G_JSON: Path = METRICS_DIR / "conformal_grouped.json"
THRESH_PICK_JSON: Path = METRICS_DIR / "threshold_pick.json"
NEG_ENERGY_JSON: Path = METRICS_DIR / "ood_energy.json"
DEFAULT_THRESHOLD: float = 0.15

MODEL_DIR: Path = _REPO_ROOT / "outputs" / "model"
MODEL_PATH: Path = MODEL_DIR / "model.pt"
FEATURES_JSON: Path = MODEL_DIR / "features.json"
TEMPERATURE_JSON: Path = MODEL_DIR / "temperature_scale.json"


def _read_json(p: Path) -> dict:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        return {}
    return {}


def _missing_artifact(path: Path, kind: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"Required {kind} not found at {path}. "
        f"Train the model first: `python -m ml.training`."
    )


@lru_cache(maxsize=1)
def _load_model_artifacts() -> tuple[MLP, tuple[str, ...], float]:
    if not MODEL_PATH.exists():
        raise _missing_artifact(MODEL_PATH, "model weights")
    if not FEATURES_JSON.exists():
        raise _missing_artifact(FEATURES_JSON, "feature schema")
    if not TEMPERATURE_JSON.exists():
        raise _missing_artifact(TEMPERATURE_JSON, "temperature scale")

    feats_raw = json.loads(FEATURES_JSON.read_text())
    if not isinstance(feats_raw, list) or not feats_raw:
        raise ValueError(
            f"{FEATURES_JSON} must contain a non-empty JSON list of feature names; "
            f"got {type(feats_raw).__name__}."
        )
    feats: tuple[str, ...] = tuple(str(c) for c in feats_raw)

    T_raw = json.loads(TEMPERATURE_JSON.read_text()).get("T")
    try:
        T = float(T_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{TEMPERATURE_JSON} field 'T' must be numeric; got {T_raw!r}."
        ) from e
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(
            f"Temperature must be a positive finite number; got T={T!r}. "
            f"Re-fit with ml.calibration.fit_temperature."
        )

    model = MLP(len(feats))
    expected_keys = set(model.state_dict().keys())
    state = torch.load(MODEL_PATH, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(
            f"{MODEL_PATH} did not deserialize to a state_dict; "
            f"got {type(state).__name__}."
        )
    saved_keys = set(state.keys())
    if saved_keys != expected_keys:
        missing = sorted(expected_keys - saved_keys)
        extra = sorted(saved_keys - expected_keys)
        raise ValueError(
            f"{MODEL_PATH} state_dict does not match ml.training.MLP architecture. "
            f"Missing keys: {missing} | Unexpected keys: {extra}. "
            f"Re-train so the saved weights match the live class definition."
        )
    model.load_state_dict(state)
    model.eval()
    return model, feats, T


def _build_feature_vector(row: pd.Series, feats: tuple[str, ...]) -> np.ndarray:
    """
    Project a patient row onto the trained model's feature schema.

    Numeric features are read directly from row (NaN or non-numeric → 0).
    Symptom indicators (sym_<token>) prefer an explicit column when present;
    otherwise they fall back to parsing row["symptoms"] as a semicolon-
    separated token string. Missing features default to 0.0, matching the
    fillna(0) policy used during training (ml/training.py).
    """
    sym_tokens: set[str] = set()
    if "symptoms" in row.index:
        s = row["symptoms"]
        if isinstance(s, str):
            sym_tokens = {t for t in s.split(";") if t}
    x = np.zeros(len(feats), dtype=np.float32)
    for i, f in enumerate(feats):
        if f.startswith("sym_") and f not in row.index:
            x[i] = 1.0 if f[4:] in sym_tokens else 0.0
            continue
        if f in row.index:
            try:
                v = float(row[f])
                if math.isfinite(v):
                    x[i] = v
            except (TypeError, ValueError):
                pass
    return x


def _real_logits(row: pd.Series) -> tuple[float, float]:
    """
    Return temperature-scaled (lo, hi) logits from the trained model.

    Loads outputs/model/model.pt once per process via lru_cache and applies
    the temperature T fit by L-BFGS during training. Callers compute p_high
    via softmax over the returned (lo, hi) pair.
    """
    model, feats, T = _load_model_artifacts()
    x = _build_feature_vector(row, feats)
    with torch.no_grad():
        raw = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    scaled = raw / T
    return float(scaled[0]), float(scaled[1])


def _softmax_high(lo: float, hi: float) -> float:
    """Numerically stable softmax over (lo, hi); returns the High-class probability."""
    m = max(lo, hi)
    e_lo = math.exp(lo - m)
    e_hi = math.exp(hi - m)
    return e_hi / (e_lo + e_hi)


def _choose_qhat(age_val: float | None) -> tuple[float, float, str]:
    g = None
    if CONF_G_JSON.exists() and age_val is not None:
        data = _read_json(CONF_G_JSON)
        alpha = float(data.get("alpha", 0.10))
        groups = data.get("groups", {})
        g = "child" if age_val < 18 else "adult"
        if g in groups:
            return float(groups[g].get("qhat", 0.10)), alpha, g
    data = _read_json(CONF_JSON)
    return float(data.get("qhat", 0.10)), float(data.get("alpha", 0.10)), g or "global"


def _energy_tau() -> float:
    d = _read_json(ENERGY_JSON)
    return float(d.get("tau", -2.0))


def _neg_energy_from_p(p: float) -> float:
    """Energy on a calibrated binary probability: -log(1 + exp(logit(p)))."""
    p_c = float(min(max(p, 1e-8), 1.0 - 1e-8))
    z = math.log(p_c / (1.0 - p_c))
    return -math.log(1.0 + math.exp(z))


def _neg_energy_signal(p_high: float) -> tuple[float, float | None, bool]:
    """
    Secondary OOD/uncertainty signal on the calibrated probability.

    Returns (neg_energy, tau, abstain_flag). When `ood_energy.json` has not
    been fit yet, tau is None and the abstain flag is False (no-op).
    Independent of the primary logit-energy gate; reported for transparency
    so the dashboard and CLI surface both signals.
    """
    e = _neg_energy_from_p(p_high)
    gate = _read_json(NEG_ENERGY_JSON)
    tau_raw = gate.get("tau") if gate else None
    if tau_raw is None:
        return e, None, False
    try:
        tau = float(tau_raw)
    except (TypeError, ValueError):
        return e, None, False
    return e, tau, e > tau


def infer_one(row_in: dict | pd.Series) -> dict:
    """
    Run the full inference pipeline on one patient row.

    Pipeline order: Mahalanobis OOD gate → trained MLP → temperature scaling
    → energy gate → split-conformal band assignment → label.

    Returns a dict with `prediction` (one of "Low", "High", "Moderate",
    "ABSTAIN"), calibrated `p_high` in [0, 1], conformal band membership,
    Mahalanobis distance, and energy-gate readout. ABSTAIN always carries
    a `reason` field: "OOD", "LogitEnergyAboveOODShift", or
    "ConformalAmbiguity". The energy-gate reason name is precise on three
    dimensions: (1) signal = logit energy, (2) direction = above OOD shift,
    (3) reference = above the in-distribution validation 95th percentile -
    Liu 2020 canonical semantics (high energy → OOD). See git log for
    refactor history.
    """
    row = pd.Series(row_in) if not isinstance(row_in, pd.Series) else row_in
    stats = load_stats()
    ood = score_tabular(row=row, stats=stats)
    d2 = ood["d2"]
    tau_d2 = ood["tau"]
    if not ood["in_dist"]:
        return {
            "prediction": "ABSTAIN",
            "reason": "OOD",
            "p_high": 0.0,
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": None,
            "energy_tau": None,
            "include_low": False,
            "include_high": False,
            "contrib": ood.get("contrib")
        }
    lo, hi = _real_logits(row)
    energy = score_energy(np.array([lo, hi], dtype=float))
    tau_e = _energy_tau()
    p_high = _softmax_high(lo, hi)
    neg_e, neg_e_tau, ood_neg = _neg_energy_signal(p_high)
    if energy > tau_e:
        return {
            "prediction": "ABSTAIN",
            "reason": "LogitEnergyAboveOODShift",
            "p_high": p_high,
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": float(energy),
            "energy_tau": float(tau_e),
            "energy_neg": float(neg_e),
            "energy_neg_tau": neg_e_tau,
            "ood_abstain_energy_neg": ood_neg,
            "include_low": False,
            "include_high": False,
            "contrib": ood.get("contrib")
        }
    age_val = None
    if "age" in row.index:
        try:
            age_val = float(row["age"])
        except Exception:
            age_val = None
    qhat, alpha, group = _choose_qhat(age_val)
    include_high = bool(p_high >= (1.0 - qhat))
    include_low = bool(p_high <= qhat)
    if include_high and include_low:
        return {
            "prediction": "ABSTAIN",
            "reason": "ConformalAmbiguity",
            "p_high": p_high,
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": float(energy),
            "energy_tau": float(tau_e),
            "energy_neg": float(neg_e),
            "energy_neg_tau": neg_e_tau,
            "ood_abstain_energy_neg": ood_neg,
            "include_low": True,
            "include_high": True,
            "qhat": float(qhat),
            "alpha": float(alpha),
            "group": group,
            "contrib": ood.get("contrib")
        }
    thresh_pick = _read_json(THRESH_PICK_JSON)
    threshold = float(thresh_pick.get("threshold", DEFAULT_THRESHOLD))
    if include_high and not include_low:
        label = "High"
    elif include_low and not include_high:
        label = "Low"
    else:
        label = "Moderate"
    out = {
        "prediction": label,
        "p_high": p_high,
        "threshold": threshold,
        "qhat": float(qhat),
        "alpha": float(alpha),
        "include_low": include_low,
        "include_high": include_high,
        "mahalanobis_d2": float(d2),
        "d2_tau": float(tau_d2),
        "energy": float(energy),
        "energy_tau": float(tau_e),
        "energy_neg": float(neg_e),
        "energy_neg_tau": neg_e_tau,
        "ood_abstain_energy_neg": ood_neg,
        "contrib": ood.get("contrib")
    }
    if group is not None:
        out["group"] = group
    return out
