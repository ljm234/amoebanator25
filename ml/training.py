import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from ml.calibration import fit_temperature
from ml.audit_hooks import (
    record_calibration_fit,
    record_data_loaded,
    record_model_saved,
    record_train_completed,
    record_train_started,
)
from ml.model import MLP
from typing import Any

torch.set_default_dtype(torch.float32)  # type: ignore[no-untyped-call]

def load_tabular(csv_path: str = "outputs/diagnosis_log_pro.csv") -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    all_symptoms = set()
    for s in df["symptoms"].astype(str):
        for token in [t for t in s.split(";") if t]:
            all_symptoms.add(token)
    for sym in sorted(all_symptoms):
        df[f"sym_{sym}"] = df["symptoms"].astype(str).apply(lambda x: 1 if sym in x.split(";") else 0)
    feats = ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure"] + [c for c in df.columns if c.startswith("sym_")]
    X = df[feats].fillna(0).astype(float).values
    y = (df["risk_label"].astype(str).str.lower()=="high").astype(int).values
    return X, y, feats  # type: ignore[return-value]

def train_and_save(model_dir: str = "outputs/model") -> dict[str, Any]:
    from ml.seeds import set_global_seeds
    set_global_seeds()
    os.makedirs(model_dir, exist_ok=True)
    X, y, feats = load_tabular()
    record_data_loaded(resource="outputs/diagnosis_log_pro.csv", n_rows=int(X.shape[0]), n_features=int(X.shape[1]))
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    record_train_started(resource=model_dir, n_train=int(len(ytr)), n_val=int(len(yva)))

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos = int((ytr == 1).sum())
    neg = int((ytr == 0).sum())
    w_pos = float(neg / max(pos, 1))  # weight for the High class
    weights = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights)

    for ep in range(60):
        model.train()
        xb = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yb = torch.tensor(ytr, dtype=torch.long, device=device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
    probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:,1]
    auc = roc_auc_score(yva, probs)
    print("Validation AUC:", round(float(auc), 4))
    print(classification_report(yva, (probs>=0.5).astype(int), target_names=["not-High","High"]))

    T = fit_temperature(model, logits, yva, device="cpu")
    record_calibration_fit(resource=model_dir, temperature=float(T), n_val=int(len(yva)))
    print("Fitted temperature:", round(float(T), 4))

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    record_model_saved(resource=model_dir, save_path=os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump(feats, f, indent=2)
    # Save simple model loader def used in app
    with open(os.path.join(model_dir, "model_def.py"), "w") as f:
        f.write(
            "import torch\n"
            "import torch.nn as nn\n"
            "from typing import Any\n\n"
            "class M(nn.Module):\n"
            "    def __init__(self, d: int) -> None:\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))\n"
            "    def forward(self, x: Any) -> Any:\n"
            "        return self.net(x)\n\n"
            "def load_model(input_dim: int, path: str, device: str = 'cpu') -> M:\n"
            "    m = M(input_dim)\n"
            "    sd = torch.load(path, map_location=device)\n"
            "    m.load_state_dict(sd)\n"
            "    m.to(device)\n"
            "    return m\n"
        )
    with open(os.path.join(model_dir, "temperature_scale.json"), "w") as f:
        json.dump({"T": float(T)}, f, indent=2)
    summary = {"auc": float(auc), "T": float(T), "n_train": int(len(ytr)), "n_val": int(len(yva))}
    record_train_completed(resource=model_dir, metrics=summary)
    return summary

if __name__ == "__main__":
    out = train_and_save()
    print(out)
