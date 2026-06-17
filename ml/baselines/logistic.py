"""
Logistic regression + Platt scaling baseline (Phase 3.1).

Platt's sigmoid calibration was originally proposed for SVMs but scikit-learn's
CalibratedClassifierCV(method="sigmoid") implements the same fit on top of any
classifier. For LR this serves as a sanity baseline: the underlying model
already outputs reasonably calibrated probabilities, so the temperature scaling
applied by Platt should make only a small adjustment.

References:
  Platt JC. "Probabilistic Outputs for Support Vector Machines and Comparisons
  to Regularized Likelihood Methods." Advances in Large Margin Classifiers, 1999.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticPlatt:
    name: str = "logistic_platt"

    def __init__(self, C: float = 1.0, class_weight: str | dict | None = "balanced") -> None:
        self.C = C
        self.class_weight = class_weight
        self.scaler_: StandardScaler | None = None
        self.model_: CalibratedClassifierCV | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticPlatt":
        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X_train)
        base = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            solver="lbfgs",
            max_iter=2000,
        )
        # cv="prefit" requires a separate calibration set; we use 3-fold internal
        # calibration when the dataset is too small for a held-out cal split.
        n_per_class_min = int(min(np.bincount(y_train)))
        cv = max(2, min(5, n_per_class_min))
        self.model_ = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        self.model_.fit(X_s, y_train)
        return self

    def predict_proba_high(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict_proba_high().")
        return self.model_.predict_proba(self.scaler_.transform(X))[:, 1]
