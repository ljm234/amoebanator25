"""
Fit a split-conformal qhat from calibrated probabilities in val_preds.csv.

All conformal math now routes through
`ml.conformal_advanced.compute_qhat` + `nonconformity_from_p`. The previous
inline implementation here duplicated the math (3rd duplication caught in
the discovery audit, alongside the MLP one) and silently bypassed the
`SmallCalibrationWarning` issued by the canonical helper. Going forward,
running this script on a small calibration set emits the warning to stderr,
matching the in-module behaviour.

Usage:
  PYTHONPATH=. python scripts/conformal/conformal_fit_from_probs.py
  PYTHONPATH=. python scripts/conformal/conformal_fit_from_probs.py --alpha 0.20
  PYTHONPATH=. python scripts/conformal/conformal_fit_from_probs.py --out /tmp/c.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml.conformal_advanced import compute_qhat, nonconformity_from_p

MET = Path("outputs/metrics")
DEFAULT_OUT = MET / "conformal.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--val_preds", type=str, default=str(MET / "val_preds.csv"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.val_preds)
    if not {"y_true", "p_high_cal"}.issubset(df.columns):
        raise ValueError("val_preds.csv must have columns: y_true, p_high_cal")

    y = df["y_true"].astype(int).to_numpy()
    p_high = df["p_high_cal"].astype(float).to_numpy()

    scores = nonconformity_from_p(p_high, y)
    qhat = compute_qhat(scores, alpha=args.alpha)

    out = {
        "alpha": float(args.alpha),
        "qhat": float(qhat),
        "n": int(len(scores)),
        "source": Path(args.val_preds).name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
