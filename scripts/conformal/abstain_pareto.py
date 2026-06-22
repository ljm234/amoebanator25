"""
ABSTAIN-rate vs accuracy Pareto frontier.

Sweeps qhat across the unit interval, computes the conformal abstain rate
at each setting, and plots the resulting trade-off between abstain rate
(x) and accuracy on the kept rows (y). Higher and to the left is better.

Output:
  outputs/metrics/abstain_pareto.json - table of (qhat, abstain, accuracy)
  outputs/metrics/abstain_pareto.png  - figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

VAL_PREDS = REPO_ROOT / "outputs" / "metrics" / "val_preds.csv"
OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "abstain_pareto.json"
OUT_PNG = REPO_ROOT / "outputs" / "metrics" / "abstain_pareto.png"


def main() -> int:
    if not VAL_PREDS.exists():
        raise SystemExit(f"missing {VAL_PREDS}")
    df = pd.read_csv(VAL_PREDS)
    if "p_high_cal" not in df.columns or "y_true" not in df.columns:
        raise SystemExit(f"{VAL_PREDS} must contain p_high_cal + y_true columns")
    y = df["y_true"].astype(int).to_numpy()
    p = df["p_high_cal"].astype(float).to_numpy()
    n = len(y)

    qhats = np.linspace(0.0, 0.5, 51)
    rows = []
    for q in qhats:
        include_high = p >= (1.0 - q)
        include_low = p <= q
        abstain = include_high & include_low
        keep = ~abstain
        if keep.sum() == 0:
            rows.append({"qhat": float(q), "abstain_rate": 1.0, "accuracy": float("nan"), "n_kept": 0})
            continue
        pred = (p[keep] >= 0.5).astype(int)
        acc = float((pred == y[keep]).mean())
        rows.append({
            "qhat": float(q),
            "abstain_rate": float(abstain.mean()),
            "accuracy": acc,
            "n_kept": int(keep.sum()),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"n": int(n), "rows": rows}, indent=2, default=float))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs = [r["abstain_rate"] for r in rows]
    ys = [r["accuracy"] for r in rows]
    ax.plot(xs, ys, "o-", color="C0")
    ax.set_xlabel("Abstain rate")
    ax.set_ylabel("Accuracy on retained predictions")
    ax.set_title(f"ABSTAIN <-> accuracy Pareto (n={n})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(json.dumps({"wrote": [str(OUT_JSON), str(OUT_PNG)], "n_points": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
