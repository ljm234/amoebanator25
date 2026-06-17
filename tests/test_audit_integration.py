"""
Phase 7.1 - integration tests for audit-trail wiring.

Verifies that:
  * a full training run emits the five expected event types
  * the persisted JSONL chain is hash-valid (verify_chain → VALID)
  * tampering with the file is detected (verify_chain → TAMPERED)
  * AMOEBANATOR_AUDIT_PATH env var redirects writes
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.audit_hooks import (
    AUDIT_PATH_ENV,
    default_audit_path,
    get_audit_log,
    record_calibration_fit,
    record_data_loaded,
    record_model_saved,
    record_train_completed,
    record_train_started,
    reset_audit_log,
    verify_persisted_chain,
)
from ml.data.audit_trail import IntegrityStatus


@pytest.fixture()
def isolated_audit_path(tmp_path: Path):
    """Yield a fresh audit path with the singleton + env var reset."""
    path = tmp_path / "audit.jsonl"
    reset_audit_log()
    old_env = os.environ.get(AUDIT_PATH_ENV)
    os.environ[AUDIT_PATH_ENV] = str(path)
    try:
        yield path
    finally:
        if old_env is None:
            os.environ.pop(AUDIT_PATH_ENV, None)
        else:
            os.environ[AUDIT_PATH_ENV] = old_env
        reset_audit_log()


def test_default_audit_path_uses_repo_default_when_env_unset() -> None:
    old_env = os.environ.pop(AUDIT_PATH_ENV, None)
    try:
        p = default_audit_path()
        assert p.name == "audit.jsonl"
        assert "outputs" in p.parts
        assert "audit" in p.parts
    finally:
        if old_env is not None:
            os.environ[AUDIT_PATH_ENV] = old_env


def test_env_var_overrides_default_path(tmp_path: Path) -> None:
    custom = tmp_path / "custom_audit.jsonl"
    with patch.dict(os.environ, {AUDIT_PATH_ENV: str(custom)}):
        assert default_audit_path() == custom.resolve()


def test_record_helpers_persist_to_jsonl(isolated_audit_path: Path) -> None:
    record_data_loaded(resource="test.csv", n_rows=30, n_features=10)
    record_train_started(resource="outputs/model", n_train=24, n_val=6)
    record_train_completed(resource="outputs/model", metrics={"auc": 1.0})
    assert isolated_audit_path.exists()
    lines = isolated_audit_path.read_text().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    event_types = [p["event_type"] for p in parsed]
    assert event_types == ["data_received", "session_start", "session_end"]


def test_chain_is_hash_valid_after_run(isolated_audit_path: Path) -> None:
    record_data_loaded(resource="test.csv", n_rows=30, n_features=10)
    record_train_started(resource="outputs/model", n_train=24, n_val=6)
    record_calibration_fit(resource="outputs/model", temperature=0.4, n_val=6)
    record_model_saved(resource="outputs/model", save_path="outputs/model/model.pt")
    record_train_completed(resource="outputs/model", metrics={"auc": 1.0})
    status, tampered = verify_persisted_chain(isolated_audit_path)
    assert status == IntegrityStatus.VALID
    assert tampered == []


def test_tampering_detected(isolated_audit_path: Path) -> None:
    """Mutating an entry's metadata after the fact must surface as TAMPERED."""
    record_data_loaded(resource="orig.csv", n_rows=30, n_features=10)
    record_train_started(resource="outputs/model", n_train=24, n_val=6)
    record_train_completed(resource="outputs/model", metrics={"auc": 0.9})

    lines = isolated_audit_path.read_text().splitlines()
    middle = json.loads(lines[1])
    middle["metadata"]["n_train"] = 9999  # tamper
    lines[1] = json.dumps(middle, sort_keys=True)
    isolated_audit_path.write_text("\n".join(lines) + "\n")

    status, tampered = verify_persisted_chain(isolated_audit_path)
    assert status != IntegrityStatus.VALID
    # The mutated entry's hash is now stale → either it itself is flagged or
    # every later entry's previous_hash mismatch is flagged.
    assert tampered, "tampered list should be non-empty"


def test_singleton_resumes_existing_chain(isolated_audit_path: Path) -> None:
    record_data_loaded(resource="first.csv", n_rows=10, n_features=5)
    reset_audit_log()  # simulate a process restart
    log = get_audit_log()
    assert len(log.entries) == 1
    record_train_started(resource="outputs/model", n_train=8, n_val=2)
    log2 = get_audit_log()
    assert len(log2.entries) == 2
    status, tampered = verify_persisted_chain(isolated_audit_path)
    assert status == IntegrityStatus.VALID
    assert tampered == []


def test_full_training_run_emits_all_five_event_types(isolated_audit_path: Path, tmp_path: Path) -> None:
    """End-to-end: invoking ml.training.train_and_save must produce the five canonical events."""
    from ml.training import train_and_save
    out_model = tmp_path / "model"
    train_and_save(model_dir=str(out_model))
    assert isolated_audit_path.exists()
    parsed = [json.loads(line) for line in isolated_audit_path.read_text().splitlines() if line.strip()]
    event_types = {p["event_type"] for p in parsed}
    assert {"data_received", "session_start", "configuration_change", "data_released", "session_end"} <= event_types


def test_chain_remains_valid_after_full_training_run(isolated_audit_path: Path, tmp_path: Path) -> None:
    from ml.training import train_and_save
    train_and_save(model_dir=str(tmp_path / "model"))
    status, tampered = verify_persisted_chain(isolated_audit_path)
    assert status == IntegrityStatus.VALID
    assert tampered == []
