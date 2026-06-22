"""
Refit Mahalanobis OOD on the TRAIN split only.

Audit context: ml.robust.fit_tabular_stats currently fits per-feature stats
over the entire diagnosis_log_pro.csv, including rows that may end up in
val/test. The leakage is small (the gate is non-parametric over robust
moments) but a reviewer will catch it. This script rederives the train
indices the same way ml/training_calib_dca.py does (random_state=42,
test_size=0.2, stratify=y) and refits the stats only on those rows.

Output: outputs/metrics/feature_stats_train.json - same schema as
feature_stats.json but provably train-only.

Usage:
  PYTHONPATH=. python scripts/ood/refit_mahalanobis_train.py
  PYTHONPATH=. python scripts/ood/refit_mahalanobis_train.py --quantile 0.99 --replace
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.robust import NUMERIC_COLS, mahalanobis_d2  # noqa: E402

LOG_CSV = REPO_ROOT / "outputs" / "diagnosis_log_pro.csv"
OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "feature_stats_train.json"
PROD_JSON = REPO_ROOT / "outputs" / "metrics" / "feature_stats.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantile", type=float, default=0.999)
    parser.add_argument("--use-diagonal", action="store_true", default=True)
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite feature_stats.json with the train-only fit.")
    args = parser.parse_args(argv)

    if not LOG_CSV.exists():
        raise SystemExit(f"missing {LOG_CSV}")
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        raise SystemExit(f"{LOG_CSV} is empty")
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "risk_label" not in df.columns:
        raise SystemExit("diagnosis_log_pro.csv missing risk_label column")
    y = (df["risk_label"].astype(str).str.lower() == "high").astype(int).to_numpy()
    idx = np.arange(len(df))
    train_idx, _ = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    df_train = df.iloc[train_idx]
    cols = [c for c in NUMERIC_COLS if c in df_train.columns]
    X = df_train[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)

    col_median = np.nanmedian(X, axis=0)
    col_mad = np.nanmedian(np.abs(X - col_median), axis=0)
    col_mad = np.where(col_mad > 0, col_mad, 1.0)
    X_filled = np.where(np.isnan(X), col_median, X)
    z = (X_filled - col_median) / col_mad
    z = np.where(np.isfinite(z), z, 0.0)
    mu = z.mean(axis=0)
    S = np.cov(z, rowvar=False)
    if args.use_diagonal:
        S = np.diag(np.clip(np.diag(S), 1e-6, None))

    d2 = []
    for i in range(z.shape[0]):
        val, _ = mahalanobis_d2(z[i], mu, S, args.use_diagonal)
        d2.append(val)
    tau = float(np.quantile(d2, args.quantile))

    out = {
        "cols": cols,
        "median": col_median.tolist(),
        "mad": col_mad.tolist(),
        "mu": mu.tolist(),
        "S": S.tolist(),
        "use_diagonal": bool(args.use_diagonal),
        "tau": tau,
        "quantile": float(args.quantile),
        "n_train": int(len(train_idx)),
        "n_total": int(len(df)),
        "provenance": "fit on train split only (random_state=42, test_size=0.2, stratify=risk_label==High)",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({"wrote": str(OUT_JSON), "tau": tau, "n_train": len(train_idx), "cols": cols}, indent=2))

    if args.replace:
        shutil.copyfile(OUT_JSON, PROD_JSON)
        print(json.dumps({"replaced": str(PROD_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
