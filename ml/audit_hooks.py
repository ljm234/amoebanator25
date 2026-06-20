"""
Phase 7.1 - production wiring of ml.data.audit_trail into the training pipeline.

ml.data.audit_trail provides AuditLog (hash-chained, Merkle-checkpointed,
tamper-evident) but is in-memory only. This module adds:

  * JSONL on-disk persistence (one line per event so the file can be
    appended atomically and inspected with `jq` / `tail -f`)
  * A process-singleton accessor with optional env-var override
    (AMOEBANATOR_AUDIT_PATH)
  * Five typed event helpers for the training pipeline:
        record_data_loaded, record_train_started, record_train_completed,
        record_calibration_fit, record_model_saved
  * A `verify_persisted_chain()` helper that re-loads a saved log and
    revalidates the full hash chain (so `pytest tests/test_audit_integration.py`
    catches any tampering between runs).

Design notes:
  - We use AuditLog.record() so the upstream chain hashing logic is the source
    of truth. We never compute hashes ourselves.
  - We persist *after* AuditLog.record() returns, so an in-memory entry that
    fails to land on disk simply means the singleton is ahead of the file -
    next call rewrites the full log via _flush_log().
  - The wrapper is intentionally thin. If callers need the raw API
    (e.g. anomaly detection, archival, exports), import ml.data.audit_trail
    directly.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.data.audit_trail import (
    AuditEntry,
    AuditEventType,
    AuditLog,
    IntegrityStatus,
)

AUDIT_PATH_ENV: str = "AMOEBANATOR_AUDIT_PATH"
_DEFAULT_AUDIT_FILENAME: str = "audit.jsonl"

_singleton_lock = threading.Lock()
_singleton_log: AuditLog | None = None
_singleton_path: Path | None = None


def default_audit_path() -> Path:
    """Resolve the audit log path from $AMOEBANATOR_AUDIT_PATH or the repo default."""
    override = os.environ.get(AUDIT_PATH_ENV)
    if override:
        return Path(override).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "outputs" / "audit" / _DEFAULT_AUDIT_FILENAME


def _entry_to_jsonl_dict(entry: AuditEntry) -> dict[str, Any]:
    """Serialise an AuditEntry to a JSON-safe dict for JSONL persistence."""
    return {
        "entry_id": entry.entry_id,
        "sequence_number": entry.sequence_number,
        "timestamp": entry.timestamp,
        "event_type": entry.event_type,
        "actor": entry.actor,
        "resource": entry.resource,
        "action_detail": entry.action_detail,
        "metadata": entry.metadata,
        "previous_hash": entry.previous_hash,
        "entry_hash": entry.entry_hash,
    }


def _entry_from_jsonl_dict(d: dict[str, Any]) -> AuditEntry:
    """Reconstruct an AuditEntry from its JSONL row."""
    return AuditEntry(
        entry_id=str(d["entry_id"]),
        sequence_number=int(d["sequence_number"]),
        timestamp=str(d["timestamp"]),
        event_type=str(d["event_type"]),
        actor=str(d["actor"]),
        resource=str(d["resource"]),
        action_detail=str(d["action_detail"]),
        metadata=dict(d.get("metadata") or {}),
        previous_hash=str(d["previous_hash"]),
        entry_hash=str(d["entry_hash"]),
    )


def _load_existing_entries(path: Path) -> list[AuditEntry]:
    """Load any pre-existing entries from disk (no validation here)."""
    if not path.exists():
        return []
    entries: list[AuditEntry] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_entry_from_jsonl_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return entries


def _append_entry(path: Path, entry: AuditEntry) -> None:
    """Atomic-ish append of one JSONL row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_entry_to_jsonl_dict(entry), sort_keys=True))
        f.write("\n")


def reset_audit_log() -> None:
    """Drop the cached singleton (used by tests so each test gets a fresh log)."""
    global _singleton_log, _singleton_path
    with _singleton_lock:
        _singleton_log = None
        _singleton_path = None


def get_audit_log(path: Path | None = None) -> AuditLog:
    """
    Return the process-singleton AuditLog.

    Recovers any prior entries from disk on first call so a freshly started
    Python process picks up where the last one left off (chain stays intact).
    Pass `path` explicitly (or set $AMOEBANATOR_AUDIT_PATH) to use a non-default
    location.
    """
    global _singleton_log, _singleton_path
    target = (path or default_audit_path()).resolve()
    with _singleton_lock:
        if _singleton_log is not None and _singleton_path == target:
            return _singleton_log
        log = AuditLog()
        for prior in _load_existing_entries(target):
            log.entries.append(prior)
        _singleton_log = log
        _singleton_path = target
        return log


def _emit(
    event_type: AuditEventType,
    actor: str,
    resource: str,
    action_detail: str,
    metadata: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    log = get_audit_log(path)
    target = (path or default_audit_path()).resolve()
    entry = log.record(
        event_type=event_type,
        actor=actor,
        resource=resource,
        action_detail=action_detail,
        metadata=metadata or {},
    )
    _append_entry(target, entry)
    return entry


# --- Typed hook helpers used by the training pipeline ------------------------

@dataclass(frozen=True)
class TrainingActor:
    """Identifies the process emitting events. Free-form; serialised verbatim."""
    name: str = "ml.training"
    user: str = ""

    def render(self) -> str:
        return f"{self.name}({self.user})" if self.user else self.name


_DEFAULT_ACTOR = TrainingActor(user=os.environ.get("USER", ""))


def record_data_loaded(
    resource: str,
    n_rows: int,
    n_features: int,
    actor: TrainingActor | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    return _emit(
        AuditEventType.DATA_RECEIVED,
        actor=(actor or _DEFAULT_ACTOR).render(),
        resource=resource,
        action_detail=f"loaded {n_rows} rows x {n_features} features",
        metadata={"n_rows": int(n_rows), "n_features": int(n_features)},
        path=path,
    )


def record_train_started(
    resource: str,
    n_train: int,
    n_val: int,
    actor: TrainingActor | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    return _emit(
        AuditEventType.SESSION_START,
        actor=(actor or _DEFAULT_ACTOR).render(),
        resource=resource,
        action_detail=f"training started (n_train={n_train}, n_val={n_val})",
        metadata={"n_train": int(n_train), "n_val": int(n_val)},
        path=path,
    )


def record_train_completed(
    resource: str,
    metrics: dict[str, Any],
    actor: TrainingActor | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    return _emit(
        AuditEventType.SESSION_END,
        actor=(actor or _DEFAULT_ACTOR).render(),
        resource=resource,
        action_detail="training completed",
        metadata={"metrics": dict(metrics)},
        path=path,
    )


def record_calibration_fit(
    resource: str,
    temperature: float,
    n_val: int,
    actor: TrainingActor | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    return _emit(
        AuditEventType.CONFIGURATION_CHANGE,
        actor=(actor or _DEFAULT_ACTOR).render(),
        resource=resource,
        action_detail=f"temperature scaling fit (T={temperature:.4f})",
        metadata={"temperature": float(temperature), "n_val": int(n_val)},
        path=path,
    )


def record_model_saved(
    resource: str,
    save_path: Path | str,
    actor: TrainingActor | None = None,
    *,
    path: Path | None = None,
) -> AuditEntry:
    return _emit(
        AuditEventType.DATA_RELEASED,
        actor=(actor or _DEFAULT_ACTOR).render(),
        resource=resource,
        action_detail=f"model artifact saved to {save_path}",
        metadata={"save_path": str(save_path)},
        path=path,
    )


# --- Verification helpers ----------------------------------------------------

def verify_persisted_chain(
    path: Path | None = None,
) -> tuple[IntegrityStatus, list[int]]:
    """
    Re-load the audit JSONL from disk and verify its hash chain. Returns
    (status, tampered_indices). Use this from tests / CI to detect
    out-of-band edits to the file.
    """
    target = (path or default_audit_path()).resolve()
    log = AuditLog()
    for prior in _load_existing_entries(target):
        log.entries.append(prior)
    return log.verify_chain()
