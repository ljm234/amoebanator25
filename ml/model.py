"""
Canonical model architecture for Amoebanator V1.0.

Single source of truth for the trained MLP. Moved here in pre-Phase-4.5
refactor (Q2.2) to:
  * de-duplicate the class definition that previously lived in both
    ml/training.py and ml/training_calib_dca.py
  * decouple inference from the training module - ml/infer.py now imports
    from ml.model, not from ml.training, so loading model.pt does not
    pull in sklearn / training-only dependencies.

Architecture (matches the saved state_dict in outputs/model/model.pt):

    Linear(input_dim, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 2)

For the bundled 10-feature schema this is 914 parameters total.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn


class MLP(nn.Module):  # type: ignore[misc]
    """Tabular feed-forward classifier with two output logits."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)
