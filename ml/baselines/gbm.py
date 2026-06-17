"""
Gradient boosted trees + isotonic calibration baseline (Phase 3.3).

Prefers LightGBM (faster, better small-leaf handling) but falls back to
sklearn's GradientBoostingClassifier if LightGBM is not installed. The
fallback path is functionally equivalent for ablation purposes - both produce
miscalibrated trees that benefit from isotonic post-hoc calibration.

References:
  Ke G et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
  NeurIPS 2017.
  Friedman JH. "Greedy Function Approximation: A Gradient Boosting Machine."
  Annals of Statistics 2001.
"""
from __future__ import annotations

import importlib.util

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


class GBMIsotonic:
    name: str = "gbm_isotonic"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: CalibratedClassifierCV | None = None
        self.backend_: str = ""

    def _make_base(self) -> object:
        if lightgbm_available():
            import lightgbm as lgb  # type: ignore[import-not-found]
            self.backend_ = "lightgbm"
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                num_leaves=max(2, 2 ** self.max_depth - 1),
            )
        self.backend_ = "sklearn_gbm"
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GBMIsotonic":
        base = self._make_base()
        n_per_class_min = int(min(np.bincount(y_train)))
        cv = max(2, min(5, n_per_class_min))
        method = "isotonic" if n_per_class_min >= 5 else "sigmoid"
        self.model_ = CalibratedClassifierCV(base, method=method, cv=cv)
        self.model_.fit(X_train, y_train)
        return self

    def predict_proba_high(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict_proba_high().")
        return self.model_.predict_proba(X)[:, 1]
