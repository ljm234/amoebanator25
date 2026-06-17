import torch
import torch.nn as nn
from typing import Any, cast

class TemperatureScaler(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, logits: Any) -> Any:
        T = torch.exp(self.logT)
        return logits / T
    def temperature(self) -> float:
        return cast(float, torch.exp(self.logT).item())

def fit_temperature(model: Any, logits_val: Any, y_val: Any, max_iter: int = 200, lr: float = 0.01, device: str = "cpu") -> float:
    model.eval()
    logits_val = torch.tensor(logits_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = criterion(scaler(logits_val), y_val)
        loss.backward()
        return loss
    opt.step(closure)  # type: ignore[no-untyped-call]
    return scaler.temperature()
