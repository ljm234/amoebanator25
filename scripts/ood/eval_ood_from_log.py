# scripts/ood/eval_ood_from_log.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.ood_simple import ood_score

LOG = Path("outputs/diagnosis_log_pro.csv")


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}")
    df = pd.read_csv(LOG)
    keep = ["age","csf_glucose","csf_protein","csf_wbc","symptoms","risk_label"]
    keep = [c for c in keep if c in df.columns]
    rows = []
    for _, r in df.iterrows():
        row: dict[str, Any] = {str(k): v for k, v in r.to_dict().items()}
        score = ood_score(row)
        row.update({"mahal": score["mahal"], "ood": score["is_ood"]})
        rows.append({k: row.get(k) for k in keep + ["mahal","ood"]})
    out = pd.DataFrame(rows)
    out_path = Path("outputs/metrics/ood_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (n={len(out)})")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
