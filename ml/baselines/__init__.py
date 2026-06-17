"""
Calibrated baselines for ablation comparisons against the Amoebanator MLP.

Every baseline exposes the same interface so the ablation runner treats them
uniformly:

    fit(X_train, y_train, X_cal, y_cal) -> object
    predict_proba(model, X) -> np.ndarray of shape (n,)   # P(High)

Baselines:
  * logistic.LogisticPlatt    - sklearn LogisticRegression + Platt scaling
  * random_forest.RFCalibrated - sklearn RandomForestClassifier + sigmoid/isotonic
  * gbm.GBMIsotonic           - LightGBM if available, else GradientBoostingClassifier,
                                 with isotonic calibration

`build_all_baselines()` returns a list of named (factory, hyperparams) tuples
so the ablation script can sweep without naming each one explicitly.
"""
from ml.baselines.gbm import GBMIsotonic, lightgbm_available
from ml.baselines.logistic import LogisticPlatt
from ml.baselines.random_forest import RFCalibrated

__all__ = [
    "LogisticPlatt",
    "RFCalibrated",
    "GBMIsotonic",
    "lightgbm_available",
    "build_all_baselines",
]


def build_all_baselines() -> list[tuple[str, type]]:
    return [
        ("logistic_platt", LogisticPlatt),
        ("rf_calibrated", RFCalibrated),
        ("gbm_isotonic", GBMIsotonic),
    ]
