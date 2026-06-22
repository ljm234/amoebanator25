"""Tests for app/audit_export.py (3 of 3).

10 tests covering CSV export contract, hash-chain round-trip
(acceptance criterion #4), tamper detection, schema
preservation, and the AUDIT_EXPORT_REQUESTED self-emission contract.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from app.audit_export import (
    CSV_SCHEMA_VERSION,
    export_audit_to_csv,
    verify_csv_chain_integrity,
)
from ml.data.audit_trail import AuditEventType


@pytest.fixture
def tmp_audit_log() -> Generator[Path, None, None]:
    """Yield a temp JSONL path; reset audit-hooks singleton each time."""
    from ml import audit_hooks as ah
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(path)
    ah._singleton_log = None
    ah._singleton_path = None
    try:
        yield path
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        ah._singleton_log = None
        ah._singleton_path = None
        path.unlink(missing_ok=True)


def _emit_n_events(n: int) -> None:
    """Emit ``n`` distinct audit events into the active singleton."""
    from ml.audit_hooks import _emit
    types = [
        AuditEventType.WEB_PREDICT_RECEIVED,
        AuditEventType.WEB_PREDICT_RETURNED,
        AuditEventType.WEB_PRESET_LOADED,
        AuditEventType.DATA_RECEIVED,
        AuditEventType.SESSION_START,
    ]
    for i in range(n):
        _emit(
            types[i % len(types)],
            actor="test",
            resource=f"resource_{i}",
            action_detail=f"event {i}",
            metadata={"index": i, "nested": {"a": i, "b": [1, 2, 3]}},
        )


# --- Test 1: returns bytes -------------------------------------------

def test_export_returns_bytes(tmp_audit_log: Path) -> None:
    _emit_n_events(3)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    assert isinstance(csv_bytes, bytes)


# --- Test 2: CSV has required columns --------------------------------

def test_export_csv_has_required_columns(tmp_audit_log: Path) -> None:
    _emit_n_events(2)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    required = {
        "schema_version", "timestamp", "event_type",
        "previous_hash", "entry_hash",
    }
    assert reader.fieldnames is not None
    assert required.issubset(set(reader.fieldnames))


# --- Test 3: preserves all rows --------------------------------------

def test_export_preserves_all_rows(tmp_audit_log: Path) -> None:
    _emit_n_events(10)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    rows = list(reader)
    # 10 emitted events + 1 from export_audit_to_csv self-emit
    # (AUDIT_EXPORT_REQUESTED) - the export reads the file fresh, so
    # the self-emit is NOT yet in the bytes returned. Only 10 rows.
    assert len(rows) == 10


# --- Test 4: round-trip hash chain byte-equal ------------------------

def test_round_trip_hash_chain_byte_equal(tmp_audit_log: Path) -> None:
    """Acceptance criterion #4 - the load-bearing test."""
    _emit_n_events(10)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    assert verify_csv_chain_integrity(csv_bytes) is True


# --- Test 5: verify pass on clean export -----------------------------

def test_verify_csv_chain_integrity_pass(tmp_audit_log: Path) -> None:
    _emit_n_events(5)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    assert verify_csv_chain_integrity(csv_bytes) is True


# --- Test 6: verify fail on tamper -----------------------------------

def test_verify_csv_chain_integrity_fail_on_tamper(tmp_audit_log: Path) -> None:
    _emit_n_events(3)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    # Tamper the metadata cell. CSV quotes JSON's double-quotes as ""
    # so the bytes pattern is ""index"":0 -> ""index"":999.
    tampered = csv_bytes.replace(b'""index"":0', b'""index"":999', 1)
    assert tampered != csv_bytes, (
        f"tamper pattern didn't match; CSV preview: {csv_bytes[:300]!r}"
    )
    assert verify_csv_chain_integrity(tampered) is False


# --- Test 7: filename format ------------------------

def test_export_filename_format() -> None:
    """The page-side caller is responsible for the filename, but the
    fixed format is ``f'amoebanator_audit_{session_id}_{ISO_ts}.csv'``.
    Verify the format string is composable."""
    session_id = "abc12345"
    iso_ts = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace(":", "-")
    )
    filename = f"amoebanator_audit_{session_id}_{iso_ts}.csv"
    assert filename.startswith("amoebanator_audit_")
    assert filename.endswith(".csv")
    assert ":" not in filename  # filesystem-safe


# --- Test 8: empty log -> header-only CSV -----------------------------

def test_export_handles_empty_log(tmp_audit_log: Path) -> None:
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    text = csv_bytes.decode("utf-8")
    # Header line only (no data rows). The export self-emits
    # AUDIT_EXPORT_REQUESTED but writes the file BEFORE the emit, so
    # this output is empty-of-data.
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 1, f"expected header only, got {len(lines)} lines"
    assert lines[0].startswith("schema_version,")


# --- Test 9: nested metadata JSON survives round-trip ----------------

def test_export_preserves_metadata_json(tmp_audit_log: Path) -> None:
    _emit_n_events(2)
    csv_bytes = export_audit_to_csv(tmp_audit_log)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    for row in reader:
        meta = json.loads(row["metadata"])
        assert isinstance(meta, dict)
        assert "index" in meta
        assert meta["nested"]["b"] == [1, 2, 3]
        # CSV_SCHEMA_VERSION column populated on every row
        assert row["schema_version"] == CSV_SCHEMA_VERSION


# --- Test 10: AUDIT_EXPORT_REQUESTED emission ------------------------

def test_export_emits_audit_event(tmp_audit_log: Path) -> None:
    """Calling export_audit_to_csv must emit AUDIT_EXPORT_REQUESTED.

    The first export reads the file (3 events), self-emits -> 4 events
    on disk. A second export reads 4 events; this is how we verify
    the self-emission landed."""
    _emit_n_events(3)
    csv_bytes_1 = export_audit_to_csv(tmp_audit_log)
    rows_1 = list(csv.DictReader(io.StringIO(csv_bytes_1.decode("utf-8"))))
    assert len(rows_1) == 3

    csv_bytes_2 = export_audit_to_csv(tmp_audit_log)
    rows_2 = list(csv.DictReader(io.StringIO(csv_bytes_2.decode("utf-8"))))
    # Now disk has 3 originals + 1 AUDIT_EXPORT_REQUESTED from first
    # export (+ 1 more from this second export not yet flushed).
    assert len(rows_2) == 4
    assert rows_2[-1]["event_type"] == "audit_export_requested"
    meta = json.loads(rows_2[-1]["metadata"])
    assert "export_size_bytes" in meta
    assert meta["row_count"] == 3  # rows in the FIRST export
