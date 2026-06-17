"""
Phase 7.3 - IRB gate integration tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ml.audit_hooks import AUDIT_PATH_ENV, reset_audit_log
from ml.irb_gate import (
    IRB_BYPASS_ENV,
    IRB_PATH_ENV,
    IRBDecision,
    IRBGateBlocked,
    check_irb_or_raise,
    evaluate_irb_record,
    is_dataset_synthetic,
)


@pytest.fixture()
def tmp_audit(tmp_path: Path):
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


def _approved_record() -> dict:
    return {
        "irb_protocol_id": "WSU-2026-0042",
        "irb_status": "approved",
        "approval_date": "2026-03-01",
        "expiration_date": "2027-03-01",
        "principal_investigator": "Jordan Montenegro",
    }


def test_synthetic_dataset_passes_without_irb_record(tmp_audit: Path) -> None:
    df = pd.DataFrame({"source": ["simulated", "simulated", "synthetic_from_yoder2010"]})
    decision = check_irb_or_raise(df=df, path=Path("/nonexistent/no.json"))
    assert decision.permitted is True
    assert decision.status == "synthetic"


def test_synthetic_detection_rejects_mixed() -> None:
    df = pd.DataFrame({"source": ["simulated", "real_ehr", "simulated"]})
    assert is_dataset_synthetic(df) is False


def test_synthetic_detection_accepts_known_prefixes() -> None:
    for prefix in ("simulated", "synthetic", "bridge", "mimic_iv"):
        df = pd.DataFrame({"source": [f"{prefix}_v2"] * 3})
        assert is_dataset_synthetic(df) is True


def test_real_dataset_with_approved_irb_passes(tmp_path: Path, tmp_audit: Path) -> None:
    rec = tmp_path / "irb.json"
    rec.write_text(json.dumps(_approved_record()))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}):
        decision = check_irb_or_raise(df=df)
    assert decision.permitted is True
    assert decision.status == "approved"


def test_real_dataset_with_revisions_requested_blocks(tmp_path: Path, tmp_audit: Path) -> None:
    rec = tmp_path / "irb.json"
    bad = _approved_record()
    bad["irb_status"] = "revisions_requested"
    rec.write_text(json.dumps(bad))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with (
        patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}),
        pytest.raises(IRBGateBlocked, match="not in the permitted set"),
    ):
        check_irb_or_raise(df=df)


def test_real_dataset_with_expired_blocks(tmp_path: Path, tmp_audit: Path) -> None:
    rec = tmp_path / "irb.json"
    bad = _approved_record()
    bad["irb_status"] = "expired"
    rec.write_text(json.dumps(bad))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}), pytest.raises(IRBGateBlocked):
        check_irb_or_raise(df=df)


def test_real_dataset_with_missing_record_blocks(tmp_path: Path, tmp_audit: Path) -> None:
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with (
        patch.dict(os.environ, {IRB_PATH_ENV: str(tmp_path / "missing.json")}),
        pytest.raises(IRBGateBlocked, match="No IRB record"),
    ):
        check_irb_or_raise(df=df)


def test_malformed_json_blocks(tmp_path: Path, tmp_audit: Path) -> None:
    rec = tmp_path / "irb.json"
    rec.write_text("not valid json {")
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}), pytest.raises(IRBGateBlocked, match="not valid JSON"):
        check_irb_or_raise(df=df)


def test_conditionally_approved_passes(tmp_path: Path, tmp_audit: Path) -> None:
    rec = tmp_path / "irb.json"
    r = _approved_record()
    r["irb_status"] = "conditionally-approved"
    rec.write_text(json.dumps(r))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}):
        decision = check_irb_or_raise(df=df)
    assert decision.permitted is True


def test_bypass_env_var_short_circuits(tmp_path: Path, tmp_audit: Path) -> None:
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_BYPASS_ENV: "1", IRB_PATH_ENV: str(tmp_path / "no.json")}):
        decision = check_irb_or_raise(df=df)
    assert decision.permitted is True
    assert decision.status == "bypassed"


def test_evaluate_returns_decision_object(tmp_path: Path) -> None:
    rec = tmp_path / "irb.json"
    rec.write_text(json.dumps(_approved_record()))
    decision = evaluate_irb_record(rec)
    assert isinstance(decision, IRBDecision)
    assert decision.permitted is True
    assert decision.record["irb_protocol_id"] == "WSU-2026-0042"


# --- Q7.B - assert AuditEventType emission on the IRB gate path --------------
# These tests close two D-finding gaps surfaced in the Phase 4.5 discovery
# audit (commit 3fd05ed): ACCESS_DENIED and IRB_STATUS_CHANGE were emitted
# from ml/irb_gate.py:173 in production but no test asserted on the emitted
# event_type. Behavior is already covered above - this just tightens the
# audit-emission contract.


def test_approved_record_emits_irb_status_change_audit_event(
    tmp_path: Path, tmp_audit: Path
) -> None:
    rec = tmp_path / "irb.json"
    rec.write_text(json.dumps(_approved_record()))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}):
        check_irb_or_raise(df=df)

    assert tmp_audit.exists(), "audit log file must be written on IRB gate evaluation"
    entries = [json.loads(line) for line in tmp_audit.read_text().splitlines() if line.strip()]
    irb_entries = [e for e in entries if e["event_type"] == "irb_status_change"]
    assert irb_entries, (
        "approved IRB record must emit an AuditEventType.IRB_STATUS_CHANGE entry; "
        "regression of this assertion would mean the audit chain silently drops "
        "the approval transition."
    )
    assert irb_entries[0]["actor"] == "ml.irb_gate.check_irb_or_raise"
    assert irb_entries[0]["metadata"]["permitted"] is True


def test_blocked_record_emits_access_denied_audit_event(
    tmp_path: Path, tmp_audit: Path
) -> None:
    rec = tmp_path / "irb.json"
    bad = _approved_record()
    bad["irb_status"] = "revisions_requested"
    rec.write_text(json.dumps(bad))
    df = pd.DataFrame({"source": ["real_ehr"] * 3})
    with patch.dict(os.environ, {IRB_PATH_ENV: str(rec)}), pytest.raises(IRBGateBlocked):
        check_irb_or_raise(df=df)

    assert tmp_audit.exists()
    entries = [json.loads(line) for line in tmp_audit.read_text().splitlines() if line.strip()]
    denied_entries = [e for e in entries if e["event_type"] == "access_denied"]
    assert denied_entries, (
        "blocked IRB record must emit an AuditEventType.ACCESS_DENIED entry; "
        "regression would mean the audit chain silently drops the rejection."
    )
    assert denied_entries[0]["metadata"]["permitted"] is False
    assert denied_entries[0]["metadata"]["status"] == "revisions_requested"
