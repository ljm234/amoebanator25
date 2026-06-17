"""
Fit both energy-based OOD/uncertainty gates from validation predictions.

Writes:
  outputs/metrics/energy_threshold.json - Liu et al. 2020 energy on raw logits
  outputs/metrics/ood_energy.json       - neg-energy-from-probability gate

If outputs/metrics/val_preds.csv lacks logit_low/logit_high columns (older
trainings did not emit them), this script recomputes the validation logits by
re-deriving the train/val split deterministically (random_state=42,
test_size=0.2, stratify=y - matching ml/training_calib_dca.py) and running
the saved model.pt over the val rows.

Usage:
  PYTHONPATH=. python scripts/ood/fit_gates.py
  PYTHONPATH=. python scripts/ood/fit_gates.py --quantile 0.99
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
MODEL_DIR = REPO_ROOT / "outputs" / "model"
LOG_CSV = REPO_ROOT / "outputs" / "diagnosis_log_pro.csv"
VAL_PREDS = METRICS_DIR / "val_preds.csv"


def _neg_energy_from_p(p: float) -> float:
    p = float(min(max(p, 1e-8), 1.0 - 1e-8))
    z = float(np.log(p / (1.0 - p)))
    return -float(np.log1p(np.exp(z)))


def _recompute_val_logits() -> np.ndarray:
    """Re-derive the val split deterministically and run model.pt over it."""
    from ml.training_calib_dca import load_tabular  # noqa: E402

    if not LOG_CSV.exists():
        raise SystemExit(f"Missing {LOG_CSV}. Run training first.")
    if not (MODEL_DIR / "model.pt").exists():
        raise SystemExit(f"Missing {MODEL_DIR / 'model.pt'}. Run training first.")

    X, y, _ = load_tabular(str(LOG_CSV))
    _, Xva, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    from ml.model import MLP  # noqa: E402
    model = MLP(X.shape[1])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva, dtype=torch.float32)).cpu().numpy()
    return logits.astype(float)


def fit_logit_energy(logits: np.ndarray, q: float) -> dict[str, float | int]:
    """Liu et al. 2020 energy on raw logits: E = -logsumexp(logits)."""
    e = np.asarray([-float(np.logaddexp.reduce(row)) for row in logits], dtype=float)
    tau = float(np.quantile(e, q))
    return {"tau": tau, "q": float(q), "n": int(len(e))}


def fit_neg_energy_from_p(p_high: np.ndarray, q: float) -> dict[str, float | int | str]:
    """Energy-from-probability gate. Higher entropy → larger neg-energy → flag."""
    e = np.asarray([_neg_energy_from_p(float(p)) for p in p_high], dtype=float)
    tau = float(np.quantile(e, q))
    return {"method": "energy_neg", "tau": tau, "q": float(q), "n": int(len(e))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile of in-distribution energies to use as the OOD threshold (default 0.95).",
    )
    args = parser.parse_args(argv)
    if not (0.0 < args.quantile < 1.0):
        parser.error("--quantile must be in (0, 1)")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not VAL_PREDS.exists():
        raise SystemExit(f"Missing {VAL_PREDS}. Run `python -m ml.training_calib_dca` first.")
    df = pd.read_csv(VAL_PREDS)

    if {"logit_low", "logit_high"}.issubset(df.columns):
        logits = df[["logit_low", "logit_high"]].to_numpy(dtype=float)
        provenance = "val_preds.csv"
    else:
        logits = _recompute_val_logits()
        provenance = "recomputed from model.pt + deterministic val split"

    if "p_high_cal" not in df.columns:
        raise SystemExit(f"{VAL_PREDS} must contain p_high_cal")
    p_high = df["p_high_cal"].astype(float).to_numpy()

    gate_logit = fit_logit_energy(logits, args.quantile)
    (METRICS_DIR / "energy_threshold.json").write_text(json.dumps(gate_logit, indent=2))

    gate_prob = fit_neg_energy_from_p(p_high, args.quantile)
    (METRICS_DIR / "ood_energy.json").write_text(json.dumps(gate_prob, indent=2))

    print(json.dumps({
        "logits_provenance": provenance,
        "energy_threshold": gate_logit,
        "ood_energy": gate_prob,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
