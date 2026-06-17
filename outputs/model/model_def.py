import torch
import torch.nn as nn
from typing import Any

class M(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self, x: Any) -> Any:
        return self.net(x)

def load_model(input_dim: int, path: str, device: str = 'cpu') -> M:
    m = M(input_dim)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.to(device)
    return m
