"""
Phase 7.2 - integration tests for de-identification wiring.

Verifies that:
  * Ages > 89 are capped at 89 per Safe Harbor
  * Dates are generalised to the year
  * Physician (actor) field is blanked
  * AMOEBANATOR_SKIP_DEIDENT=1 bypass flag works (with summary.bypassed=True)
  * load_tabular_safe_harbor returns the same (X, y) shape as ml.training.load_tabular
  * The audit log records a DATA_VERIFIED entry for each load
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ml.audit_hooks import AUDIT_PATH_ENV, reset_audit_log, verify_persisted_chain
from ml.data.audit_trail import IntegrityStatus
from ml.data_loader import (
    SKIP_DEIDENT_ENV,
    deidentify_dataframe,
    load_tabular_safe_harbor,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": ["a1", "b2", "c3"],
        "source": ["simulated"] * 3,
        "physician": ["Dr. Real Name", "Dr. Another", ""],
        "timestamp_tz": ["2025-11-01 12:00:00 -0600", "2024-01-15 09:00:00 -0600", None],
        "age": [12, 95, 45],
        "sex": ["M", "F", "M"],
        "csf_glucose": [18.0, 70.0, 50.0],
        "csf_protein": [420.0, 0.5, 1.0],
        "csf_wbc": [2100, 3, 10],
        "symptoms": ["fever;headache;nuchal_rigidity", "", "fever"],
        "pcr": [1, 0, 0],
        "microscopy": [1, 0, 0],
        "exposure": [1, 0, 0],
        "risk_score": [16, 2, 5],
        "risk_label": ["High", "Low", "Low"],
        "comments": ["Long comment that exceeds twenty characters and should be scrubbed.", "", ""],
    })


@pytest.fixture()
def isolated_audit_path(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    reset_audit_log()
    old_env = os.environ.get(AUDIT_PATH_ENV)
    os.environ[AUDIT_PATH_ENV] = str(p)
    try:
        yield p
    finally:
        if old_env is None:
            os.environ.pop(AUDIT_PATH_ENV, None)
        else:
            os.environ[AUDIT_PATH_ENV] = old_env
        reset_audit_log()


def test_ages_above_89_are_capped() -> None:
    df = _sample_df()
    out, summary = deidentify_dataframe(df)
    assert int(out["age"].max()) <= 89
    assert summary.n_age_capped == 1


def test_physician_is_blanked() -> None:
    df = _sample_df()
    out, summary = deidentify_dataframe(df)
    assert all(out["physician"].astype(str) == "")
    assert summary.n_actor_blanked == 2  # 2 non-empty physician rows


def test_dates_generalised_to_year() -> None:
    df = _sample_df()
    out, summary = deidentify_dataframe(df)
    assert summary.n_dates_truncated >= 2  # 2 non-null dates
    assert "2025" in out["timestamp_tz"].astype(str).tolist()


def test_clinical_columns_pass_through_unchanged() -> None:
    df = _sample_df()
    out, _ = deidentify_dataframe(df)
    assert (out["csf_glucose"] == df["csf_glucose"]).all()
    assert (out["risk_label"] == df["risk_label"]).all()
    assert (out["sex"] == df["sex"]).all()


def test_bypass_via_kwarg() -> None:
    df = _sample_df()
    out, summary = deidentify_dataframe(df, bypass=True)
    assert summary.bypassed is True
    assert int(out["age"].max()) == 95  # untouched


def test_bypass_via_env_var() -> None:
    df = _sample_df()
    with patch.dict(os.environ, {SKIP_DEIDENT_ENV: "1"}):
        _, summary = deidentify_dataframe(df)
    assert summary.bypassed is True


def test_load_tabular_safe_harbor_matches_existing_shape(tmp_path: Path, isolated_audit_path: Path) -> None:
    """The drop-in loader must produce the same (X, y) shape as ml.training.load_tabular."""
    df = _sample_df()
    csv = tmp_path / "log.csv"
    df.to_csv(csv, index=False)

    from ml.training import load_tabular
    Xa, ya, fa = load_tabular(str(csv))
    Xb, yb, fb, summary = load_tabular_safe_harbor(str(csv))

    assert Xa.shape == Xb.shape
    assert ya.shape == yb.shape
    assert fa == fb
    assert summary.n_rows == 3


def test_load_emits_audit_entry(tmp_path: Path, isolated_audit_path: Path) -> None:
    df = _sample_df()
    csv = tmp_path / "log.csv"
    df.to_csv(csv, index=False)
    load_tabular_safe_harbor(str(csv))
    assert isolated_audit_path.exists()
    entries = [json.loads(line) for line in isolated_audit_path.read_text().splitlines() if line.strip()]
    deident_entries = [e for e in entries if e["event_type"] == "data_verified"]
    assert len(deident_entries) >= 1
    assert deident_entries[0]["metadata"]["n_rows"] == 3
    status, _ = verify_persisted_chain(isolated_audit_path)
    assert status == IntegrityStatus.VALID
