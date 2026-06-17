"""
Q4.B regression test - `scripts.conformal.conformal_fit_from_probs` must route through
`ml.conformal_advanced.compute_qhat` and therefore emit `SmallCalibrationWarning`
on small calibration sets. Without this, the script can ship a qhat fit on
n < 100 with no warning to stderr - the silent-bypass bug surfaced during
the Phase 4.5 discovery audit.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from ml.conformal_advanced import SmallCalibrationWarning
from scripts.conformal.conformal_fit_from_probs import main as conformal_main


def _write_small_val_preds(p: Path) -> None:
    """Six-row val_preds.csv mimicking the production small-sample case."""
    df = pd.DataFrame({
        "y_true": [1, 0, 1, 0, 1, 0],
        "p_high_cal": [0.92, 0.05, 0.88, 0.10, 0.71, 0.35],
    })
    df.to_csv(p, index=False)


def test_script_emits_smallcalibrationwarning_at_n6(tmp_path: Path) -> None:
    val = tmp_path / "val_preds.csv"
    out = tmp_path / "conformal.json"
    _write_small_val_preds(val)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=SmallCalibrationWarning)
        rc = conformal_main(["--val_preds", str(val), "--out", str(out)])

    assert rc == 0
    assert out.exists(), "script must write the conformal.json artefact"
    written = json.loads(out.read_text())
    assert written["n"] == 6
    assert 0.0 <= written["qhat"] <= 1.0
    assert written["alpha"] == pytest.approx(0.10)

    fired = [w for w in caught if issubclass(w.category, SmallCalibrationWarning)]
    assert fired, (
        "scripts.conformal.conformal_fit_from_probs must emit SmallCalibrationWarning "
        "when n < SMALL_CAL_FLOOR; refactoring it through "
        "ml.conformal_advanced.compute_qhat is the fix logged in PHASE_4_5_PLAN.md (Q4.B)."
    )


def test_script_alpha_flag_propagates(tmp_path: Path) -> None:
    val = tmp_path / "val_preds.csv"
    out = tmp_path / "conformal.json"
    _write_small_val_preds(val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SmallCalibrationWarning)
        conformal_main(["--val_preds", str(val), "--out", str(out), "--alpha", "0.20"])
    assert json.loads(out.read_text())["alpha"] == pytest.approx(0.20)
