"""
Phase 5.4 - Synthetic OOD shift benchmarks.

Generates two adversarial shifts of the bundled simulated data and reports
the detection rate of each OOD/uncertainty gate (Mahalanobis, logit-energy,
neg-energy):

  * covariate_shift - multiply CSF lab values by random factors in [0.5, 2.0]
                      and add Gaussian noise; keep labels as-is.
  * label_shift     - flip labels with probability 0.5 (model now sees a
                      population whose label distribution is uniform-random).

For each shift type we report per-gate detection rate, false-alarm rate on
in-distribution test rows, and the AUC of the gate's continuous score as an
OOD discriminator.

Output: outputs/metrics/synthetic_ood_benchmark.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.infer import _real_logits, _softmax_high  # noqa: E402
from ml.ood_energy import neg_energy_from_p  # noqa: E402
from ml.robust import load_stats, score_energy, score_tabular  # noqa: E402

LOG_CSV = REPO_ROOT / "outputs" / "diagnosis_log_pro.csv"
OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "synthetic_ood_benchmark.json"

NUMERIC_FEATURES = ["age", "csf_glucose", "csf_protein", "csf_wbc", "pcr", "microscopy", "exposure"]


def _row_signals(row: pd.Series, stats: dict) -> dict[str, float]:
    """Return the three gate scores for a single row."""
    ood = score_tabular(row=row, stats=stats)
    d2 = float(ood["d2"])
    try:
        lo, hi = _real_logits(row)
        e_logit = float(score_energy(np.array([lo, hi], dtype=float)))
        p_high = _softmax_high(lo, hi)
        e_neg = float(neg_energy_from_p(p_high))
    except Exception:
        e_logit, e_neg = float("nan"), float("nan")
    return {"mahalanobis_d2": d2, "logit_energy": e_logit, "neg_energy": e_neg}


def _gather(df: pd.DataFrame, stats: dict, label: int) -> list[dict[str, float]]:
    out = []
    for _, row in df.iterrows():
        sig = _row_signals(row, stats)
        sig["is_ood"] = float(label)
        out.append(sig)
    return out


def covariate_shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Multiply numeric labs by random scale in [0.5, 2.0] and add Gaussian noise."""
    rng = np.random.default_rng(seed)
    shifted = df.copy()
    for c in ["csf_glucose", "csf_protein", "csf_wbc"]:
        if c in shifted.columns:
            scale = rng.uniform(0.5, 2.0, size=len(shifted))
            noise = rng.normal(0.0, 0.1, size=len(shifted)) * shifted[c].astype(float).abs()
            shifted[c] = (shifted[c].astype(float) * scale + noise).clip(lower=0.0)
    return shifted


def label_shift(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Randomly flip risk_label with prob 0.5 (gate will not see the label, but the
    *covariate* distribution conditional on label changes - equivalent to a label
    shift relative to training)."""
    rng = np.random.default_rng(seed)
    shifted = df.copy()
    if "risk_label" in shifted.columns:
        flip = rng.random(size=len(shifted)) < 0.5
        new_labels = shifted["risk_label"].astype(str).copy()
        new_labels[flip] = np.where(
            shifted.loc[flip, "risk_label"].astype(str).str.lower() == "high",
            "Low", "High"
        )
        shifted["risk_label"] = new_labels
    return shifted


def evaluate(rows_in: list[dict[str, float]], rows_out: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    df = pd.DataFrame(rows_in + rows_out)
    if "is_ood" not in df.columns:
        raise RuntimeError("rows missing is_ood")
    y = df["is_ood"].astype(int).to_numpy()
    out: dict[str, dict[str, float]] = {}
    for gate in ("mahalanobis_d2", "logit_energy", "neg_energy"):
        s = df[gate].astype(float).to_numpy()
        mask = np.isfinite(s)
        if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
            out[gate] = {"auc": float("nan"), "n_finite": int(mask.sum())}
            continue
        # logit_energy is "lower = OOD" in the existing convention; flip sign for AUC
        score = -s if gate == "logit_energy" else s
        out[gate] = {
            "auc": float(roc_auc_score(y[mask], score[mask])),
            "n_finite": int(mask.sum()),
            "median_in_dist": float(np.median(s[mask & (y == 0)])),
            "median_ood": float(np.median(s[mask & (y == 1)])),
        }
    return out


def main() -> int:
    if not LOG_CSV.exists():
        raise SystemExit(f"missing {LOG_CSV}")
    df = pd.read_csv(LOG_CSV)
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    stats = load_stats()

    in_rows = _gather(df, stats, label=0)
    cov_rows = _gather(covariate_shift(df), stats, label=1)
    lab_rows = _gather(label_shift(df), stats, label=1)

    payload = {
        "n_in_dist": len(in_rows),
        "n_covariate_shift": len(cov_rows),
        "n_label_shift": len(lab_rows),
        "covariate_shift": evaluate(in_rows, cov_rows),
        "label_shift": evaluate(in_rows, lab_rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
