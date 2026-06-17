"""
Calibrated random forest baseline (Phase 3.2).

Random forests systematically push probabilities toward 0.5 (Niculescu-Mizil
& Caruana 2005). We wrap sklearn's RandomForestClassifier in
CalibratedClassifierCV with isotonic regression by default - isotonic is the
right choice for RF because the calibration curve is non-monotone-S-shaped,
not just temperature-shifted.

References:
  Niculescu-Mizil A, Caruana R. "Predicting Good Probabilities With
  Supervised Learning." ICML 2005.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


class RFCalibrated:
    name: str = "rf_calibrated"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        method: str = "isotonic",
        random_state: int = 42,
        class_weight: str | dict | None = "balanced",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.method = method
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_: CalibratedClassifierCV | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RFCalibrated":
        base = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )
        n_per_class_min = int(min(np.bincount(y_train)))
        cv = max(2, min(5, n_per_class_min))
        # Isotonic needs ~50+ samples per class to avoid overfit; fall back to
        # sigmoid for very small datasets.
        method = self.method
        if method == "isotonic" and n_per_class_min < 5:
            method = "sigmoid"
        self.model_ = CalibratedClassifierCV(base, method=method, cv=cv)
        self.model_.fit(X_train, y_train)
        return self

    def predict_proba_high(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict_proba_high().")
        return self.model_.predict_proba(X)[:, 1]
