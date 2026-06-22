"""
Empirical conformal coverage across alpha in {0.05, 0.10, 0.20}.

Uses ml/conformal_advanced.coverage_sweep. Splits val_preds.csv into a
calibration half and a held-out half (proper conformal protocol; current
val_preds is small but the framework is correct), then for each alpha
reports the qhat, empirical coverage, and abstain rate.

Output: outputs/metrics/coverage_sweep.json (table) + coverage_sweep.png (figure).

Note: with the bundled n=6 val_preds the per-alpha numbers are noisy by
construction; the SmallCalibrationWarning is intentional and documented.
The script is the framework a future run will populate with n>=200.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.conformal_advanced import (  # noqa: E402
    SmallCalibrationWarning,
    coverage_sweep,
    nonconformity_from_p,
)

VAL_PREDS = REPO_ROOT / "outputs" / "metrics" / "val_preds.csv"
OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "coverage_sweep.json"
OUT_PNG = REPO_ROOT / "outputs" / "metrics" / "coverage_sweep.png"
ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)


def main() -> int:
    if not VAL_PREDS.exists():
        raise SystemExit(f"missing {VAL_PREDS}")
    df = pd.read_csv(VAL_PREDS)
    if "p_high_cal" not in df.columns or "y_true" not in df.columns:
        raise SystemExit(f"{VAL_PREDS} must contain p_high_cal + y_true columns")
    y = df["y_true"].astype(int).to_numpy()
    p = df["p_high_cal"].astype(float).to_numpy()
    n = len(y)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    half = n // 2
    cal_idx, test_idx = perm[:half], perm[half:]
    if len(cal_idx) == 0 or len(test_idx) == 0:
        raise SystemExit(f"need n >= 2; got n={n}")

    cal_scores = nonconformity_from_p(p[cal_idx], y[cal_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("default", category=SmallCalibrationWarning)
        results = coverage_sweep(
            cal_scores=cal_scores,
            test_p_high=p[test_idx],
            test_y=y[test_idx],
            alphas=ALPHAS,
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)), "rows": results}, indent=2, default=float))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    target = [1.0 - r["alpha"] for r in results]
    empirical = [r["coverage"] for r in results]
    abstain = [r["abstain_rate"] for r in results]
    ax.plot(ALPHAS, target, "k--", label="target coverage = 1 - alpha")
    ax.plot(ALPHAS, empirical, "o-", label="empirical coverage")
    ax.plot(ALPHAS, abstain, "s-", label="abstain rate")
    ax.set_xlabel("alpha (target miscoverage)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Conformal coverage sweep (n_cal={len(cal_idx)}, n_test={len(test_idx)})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print(json.dumps({"wrote": [str(OUT_JSON), str(OUT_PNG)], "rows": results}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
