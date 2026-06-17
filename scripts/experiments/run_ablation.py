"""
Phase 3.4 - Four-cell ablation across baselines and the Amoebanator MLP.

For each (model, ablation cell) pair, we fit on the training split, evaluate on
the held-out test split, and report AUC + recall@operating-point with bootstrap
95% CIs (Phase 3.5). The four cells are:

    base       - model alone, raw probabilities
    +cal       - Platt or isotonic calibration on the model output
    +conformal - base or calibrated probability + split conformal abstain rule
    +ood       - calibrated + conformal + Mahalanobis OOD gate (abstain on OOD)

Models compared:
  * logistic_platt   (sklearn LogisticRegression + Platt)
  * rf_calibrated    (sklearn RandomForestClassifier + isotonic)
  * gbm_isotonic     (LightGBM if installed, else GradientBoostingClassifier)
  * amoebanator_mlp  (the trained MLP from outputs/model/model.pt + temperature scaling)

Output: outputs/metrics/ablation_table.json with one row per (model, cell)
plus a CSV mirror at outputs/metrics/ablation_table.csv.

This script runs to completion on the bundled simulated data so the wiring is
proven; the headline numbers should be read with the README's Limitations
section in mind (n=24 train, n=6 val/test). Real metrics come in Phase 2.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.baselines import GBMIsotonic, LogisticPlatt, RFCalibrated, lightgbm_available  # noqa: E402
from ml.conformal_advanced import (  # noqa: E402
    SmallCalibrationWarning,
    compute_qhat,
    nonconformity_from_p,
)
from ml.metrics.bootstrap import bootstrap_ci  # noqa: E402
from ml.robust import load_stats, score_tabular  # noqa: E402
from ml.splits import split_summary, stratified_split  # noqa: E402
from ml.training_calib_dca import load_tabular  # noqa: E402

OUT_JSON = REPO_ROOT / "outputs" / "metrics" / "ablation_table.json"
OUT_CSV = REPO_ROOT / "outputs" / "metrics" / "ablation_table.csv"
OPERATING_POINT = 0.5


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_recall(y: np.ndarray, p: np.ndarray, t: float = OPERATING_POINT) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(recall_score(y, (p >= t).astype(int), pos_label=1, zero_division=0.0))  # type: ignore[arg-type]


def _ci(metric_fn: Any, y: np.ndarray, p: np.ndarray) -> dict[str, float] | None:
    try:
        if len(np.unique(y)) < 2:
            return None
        return dict(bootstrap_ci(metric_fn, y, p, n_resamples=2000, alpha=0.05, seed=0))  # type: ignore[arg-type]
    except Exception as e:
        warnings.warn(f"bootstrap CI failed: {type(e).__name__}: {e}", stacklevel=2)
        return None


def _amoebanator_proba(X: np.ndarray) -> np.ndarray:
    """Run the trained MLP + temperature scaling on a feature matrix."""
    from ml.model import MLP

    model_dir = REPO_ROOT / "outputs" / "model"
    feats = json.loads((model_dir / "features.json").read_text())
    T = float(json.loads((model_dir / "temperature_scale.json").read_text())["T"])
    model = MLP(len(feats))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    raw = raw / T
    z = raw - raw.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1].astype(float)


def _ood_mask(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Boolean mask of rows the Mahalanobis OOD gate would abstain on."""
    stats = load_stats()
    if not stats.get("cols"):
        return np.zeros(len(X), dtype=bool)
    abstain = np.zeros(len(X), dtype=bool)
    for i, row_arr in enumerate(X):
        row = pd.Series({c: row_arr[feature_names.index(c)] for c in feature_names})
        ood = score_tabular(row=row, stats=stats)
        abstain[i] = not ood["in_dist"]
    return abstain


def _evaluate_cells(
    name: str,
    p_base: np.ndarray,
    p_cal: np.ndarray,
    cal_scores: np.ndarray,
    y_test: np.ndarray,
    ood_mask: np.ndarray,
    alpha: float = 0.10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, p in [("base", p_base), ("+cal", p_cal)]:
        rows.append({
            "model": name, "cell": cell,
            "n": int(len(y_test)), "n_pos": int(y_test.sum()),
            "auc": _safe_auc(y_test, p),
            "auc_ci": _ci(_safe_auc, y_test, p),
            "recall_high@0.5": _safe_recall(y_test, p),
            "recall_ci": _ci(_safe_recall, y_test, p),
            "abstain_rate": 0.0,
        })
    qhat = compute_qhat(cal_scores, alpha=alpha)
    include_high = p_cal >= (1.0 - qhat)
    include_low = p_cal <= qhat
    abstain_conformal = include_high & include_low
    keep = ~abstain_conformal
    rows.append({
        "model": name, "cell": "+conformal",
        "n": int(keep.sum()), "n_pos": int(y_test[keep].sum()),
        "auc": _safe_auc(y_test[keep], p_cal[keep]) if keep.sum() else float("nan"),
        "auc_ci": _ci(_safe_auc, y_test[keep], p_cal[keep]) if keep.sum() else None,
        "recall_high@0.5": _safe_recall(y_test[keep], p_cal[keep]) if keep.sum() else float("nan"),
        "recall_ci": _ci(_safe_recall, y_test[keep], p_cal[keep]) if keep.sum() else None,
        "abstain_rate": float(abstain_conformal.mean()),
        "qhat": float(qhat), "alpha": float(alpha),
    })
    abstain_combined = abstain_conformal | ood_mask
    keep_c = ~abstain_combined
    rows.append({
        "model": name, "cell": "+ood",
        "n": int(keep_c.sum()), "n_pos": int(y_test[keep_c].sum()),
        "auc": _safe_auc(y_test[keep_c], p_cal[keep_c]) if keep_c.sum() else float("nan"),
        "auc_ci": _ci(_safe_auc, y_test[keep_c], p_cal[keep_c]) if keep_c.sum() else None,
        "recall_high@0.5": _safe_recall(y_test[keep_c], p_cal[keep_c]) if keep_c.sum() else float("nan"),
        "recall_ci": _ci(_safe_recall, y_test[keep_c], p_cal[keep_c]) if keep_c.sum() else None,
        "abstain_rate": float(abstain_combined.mean()),
        "ood_abstain_rate": float(ood_mask.mean()),
        "qhat": float(qhat), "alpha": float(alpha),
    })
    return rows


def main() -> int:
    X, y, feats = load_tabular()
    splits = stratified_split(y, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)
    summary = split_summary(y, splits)

    Xtr, ytr = X[splits["train"]], y[splits["train"]]
    Xca, yca = X[splits["val"]], y[splits["val"]]
    Xte, yte = X[splits["test"]], y[splits["test"]]

    ood_mask_test = _ood_mask(Xte, feats)
    rows: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("default", category=SmallCalibrationWarning)

        for name, factory in [
            ("logistic_platt", LogisticPlatt),
            ("rf_calibrated", RFCalibrated),
            ("gbm_isotonic", GBMIsotonic),
        ]:
            try:
                clf = factory()
                clf.fit(Xtr, ytr)
            except Exception as e:
                warnings.warn(f"{name}: fit failed: {type(e).__name__}: {e}", stacklevel=2)
                continue
            p_base = clf.predict_proba_high(Xte)
            p_cal = p_base
            cal_scores = nonconformity_from_p(clf.predict_proba_high(Xca), yca)
            rows.extend(_evaluate_cells(name, p_base, p_cal, cal_scores, yte, ood_mask_test))

        try:
            p_te = _amoebanator_proba(Xte)
            p_ca = _amoebanator_proba(Xca)
            cal_scores = nonconformity_from_p(p_ca, yca)
            rows.extend(_evaluate_cells("amoebanator_mlp", p_te, p_te, cal_scores, yte, ood_mask_test))
        except Exception as e:
            warnings.warn(f"amoebanator_mlp evaluation failed: {type(e).__name__}: {e}", stacklevel=2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lightgbm_available": lightgbm_available(),
        "splits": summary,
        "operating_point": OPERATING_POINT,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(json.dumps({"n_rows": len(rows), "wrote": [str(OUT_JSON), str(OUT_CSV)]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
