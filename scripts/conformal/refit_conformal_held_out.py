"""
Phase 4.1 - proper held-out split-conformal calibration framework.

Replaces the audit-flagged practice of fitting conformal qhat on the same n=6
validation set used for calibration. The framework here:

  1. Loads val_preds.csv (or any (y, p_high_cal) CSV) plus an optional
     test_preds.csv with the same columns.
  2. Refuses to silently proceed when the calibration set is below
     SMALL_CAL_FLOOR (=100) - the SmallCalibrationWarning fires, and a
     `--force-small` flag is required to write the artifact anyway. This
     blocks accidental shipment of a coverage claim against tiny n.
  3. Writes outputs/metrics/conformal_heldout.json with the computed qhat,
     alpha, calibration set size, and a `provenance` field naming the file
     the calibration came from.

When Phase 2 supplies a real held-out set with n >= 200, this script runs
unchanged against the new artifact and produces the conformal numbers the
preprint will quote.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.conformal_advanced import (  # noqa: E402
    SMALL_CAL_FLOOR,
    SmallCalibrationWarning,
    compute_qhat,
    label_conditional_qhats,
    nonconformity_from_p,
)

DEFAULT_CAL = REPO_ROOT / "outputs" / "metrics" / "val_preds.csv"
OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "conformal_heldout.json"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    for col in ("y_true", "p_high_cal"):
        if col not in df.columns:
            raise SystemExit(f"{path} must contain '{col}'; got {list(df.columns)}")
    return df["y_true"].astype(int).to_numpy(), df["p_high_cal"].astype(float).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", type=Path, default=DEFAULT_CAL,
                        help="Calibration CSV with columns y_true,p_high_cal.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--label-conditional", action="store_true",
                        help="Also fit per-class qhats (Vovk Mondrian conformal).")
    parser.add_argument("--force-small", action="store_true",
                        help="Allow writing a qhat fit on n < SMALL_CAL_FLOOR.")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    if not args.cal.exists():
        raise SystemExit(f"missing calibration file: {args.cal}")
    y, p = _load(args.cal)
    n = len(y)
    scores = nonconformity_from_p(p, y)

    payload: dict[str, object] = {
        "alpha": float(args.alpha),
        "n": int(n),
        "provenance": str(args.cal.relative_to(REPO_ROOT)) if args.cal.is_relative_to(REPO_ROOT) else str(args.cal),
        "small_cal_floor": SMALL_CAL_FLOOR,
        "force_small": bool(args.force_small),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=SmallCalibrationWarning)
        qhat = compute_qhat(scores, alpha=args.alpha)
        small_warning_fired = any(issubclass(w.category, SmallCalibrationWarning) for w in caught)

    payload["qhat"] = qhat
    payload["small_calibration_warning"] = small_warning_fired

    if args.label_conditional:
        per_class = label_conditional_qhats(scores, y, alpha=args.alpha)
        payload["qhats_per_class"] = {str(k): float(v) for k, v in per_class.items()}

    if small_warning_fired and not args.force_small:
        print(json.dumps({
            "status": "blocked",
            "reason": (
                f"Calibration set n={n} is below SMALL_CAL_FLOOR={SMALL_CAL_FLOOR}. "
                f"Re-run with --force-small to write anyway, or supply a larger held-out set "
                f"via --cal."
            ),
            "would_have_written": payload,
        }, indent=2, default=float))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({"status": "ok", "wrote": str(args.out), "payload": payload}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
