import math
import json
from pathlib import Path
from typing import cast

METRICS_DIR = Path("outputs/metrics")
ENERGY_JSON = METRICS_DIR / "ood_energy.json"

def neg_energy_from_p(p: float) -> float:
    p = float(min(max(p, 1e-8), 1 - 1e-8))
    z = math.log(p / (1.0 - p))
    return -math.log(1.0 + math.exp(z))

def load_energy_gate() -> dict[str, object]:
    if ENERGY_JSON.exists():
        try:
            return cast(dict[str, object], json.loads(ENERGY_JSON.read_text()))
        except Exception:
            pass
    return {"method": "energy_neg", "tau": None, "q": None, "n": 0}

def ood_abstain_energy(p_high: float) -> dict[str, object]:
    gate = load_energy_gate()
    tau = gate.get("tau", None)
    e = neg_energy_from_p(p_high)
    abstain = (tau is not None) and (e > float(tau))  # type: ignore[arg-type]
    return {"energy_neg": float(e), "tau": (float(tau) if tau is not None else None), "ood_abstain_energy": bool(abstain)}  # type: ignore[arg-type]
