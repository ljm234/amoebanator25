"""CSV export + chain-integrity verification for the audit log.

Q13.A locked feature: in-UI ``st.download_button`` lets a reviewer
export the current session's audit chain as CSV and verify integrity
post-download against a cloned repo. This converts HF Space's ephemeral
filesystem (audit log wipes on container restart) into an explicit
audit-portability feature.

Two public functions:

- ``export_audit_to_csv(jsonl_path)``         - read the JSONL audit log
                                                 at ``jsonl_path``,
                                                 return CSV bytes
                                                 preserving every column
                                                 plus a ``schema_version``.
                                                 Emits
                                                 ``AUDIT_EXPORT_REQUESTED``.
- ``verify_csv_chain_integrity(csv_bytes)``    - re-parse the CSV, walk
                                                 the chain, return True
                                                 iff every row's
                                                 ``entry_hash`` matches
                                                 the canonical hash
                                                 recomputation.

The Mini-1 closure gate criterion #4 round-trip test:
write 10 events → ``export_audit_to_csv`` → re-parse → byte-equal hash
chain.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ml.audit_hooks import _emit, _load_existing_entries
from ml.data.audit_trail import AuditEventType, _compute_entry_hash


# CSV format version; bump when columns / serialisation changes.
# Independent of the JSONL log's schema (the JSONL has no version field
# yet - that's a separate cleanup tracked as Mini-1 spec-gap-4).
CSV_SCHEMA_VERSION: str = "1"


# Column order is the contract - downstream verify_csv_chain_integrity
# and any external reviewer-side tooling reads in this exact order.
_CSV_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "entry_id",
    "sequence_number",
    "timestamp",
    "event_type",
    "actor",
    "resource",
    "action_detail",
    "metadata",       # JSON-encoded for CSV cell safety
    "previous_hash",
    "entry_hash",
)


def export_audit_to_csv(jsonl_path: Path) -> bytes:
    """Read the JSONL audit log at ``jsonl_path``; return CSV-encoded bytes.

    No filtering, no truncation - every entry survives the round-trip.
    The CSV is UTF-8 encoded with ``\\r\\n`` line endings (RFC 4180).
    Metadata dicts are JSON-encoded into their cell so the CSV stays
    flat-table-friendly while preserving all nested structure.

    Side effect: emits ``AuditEventType.AUDIT_EXPORT_REQUESTED`` with
    metadata ``{"export_size_bytes": <int>, "row_count": <int>}`` so
    audit-of-audits is itself in the chain.
    """
    entries = _load_existing_entries(jsonl_path)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(_CSV_COLUMNS)
    for e in entries:
        writer.writerow([
            CSV_SCHEMA_VERSION,
            e.entry_id,
            e.sequence_number,
            e.timestamp,
            e.event_type,
            e.actor,
            e.resource,
            e.action_detail,
            json.dumps(e.metadata, sort_keys=True, separators=(",", ":")),
            e.previous_hash,
            e.entry_hash,
        ])
    csv_bytes = buf.getvalue().encode("utf-8")
    _emit(
        AuditEventType.AUDIT_EXPORT_REQUESTED,
        actor="app.audit_export",
        resource=str(jsonl_path),
        action_detail=f"exported {len(entries)} entries to CSV",
        metadata={
            "export_size_bytes": len(csv_bytes),
            "row_count": len(entries),
            "csv_schema_version": CSV_SCHEMA_VERSION,
        },
    )
    return csv_bytes


def verify_csv_chain_integrity(csv_bytes: bytes) -> bool:
    """Re-parse ``csv_bytes`` and verify the hash chain row-by-row.

    For each row: recompute ``entry_hash`` via ``_compute_entry_hash``
    using the same field set as the original chain, and assert it
    matches the stored ``entry_hash`` cell.

    Returns ``True`` iff every row passes AND the ``previous_hash`` of
    each row equals the ``entry_hash`` of the prior row (chain links
    intact). Returns ``False`` on any mismatch - fail-closed because a
    silent True on a tampered chain defeats the audit's whole purpose.

    Empty CSV (header row only, no entries) returns ``True`` -
    vacuously correct.
    """
    text = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or tuple(reader.fieldnames) != _CSV_COLUMNS:
        return False

    prior_hash: str | None = None
    for row in reader:
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

        recomputed = _compute_entry_hash(
            entry_id=row["entry_id"],
            sequence_number=int(row["sequence_number"]),
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            actor=row["actor"],
            resource=row["resource"],
            action_detail=row["action_detail"],
            metadata=metadata,
            previous_hash=row["previous_hash"],
        )
        if recomputed != row["entry_hash"]:
            return False

        if prior_hash is not None and row["previous_hash"] != prior_hash:
            return False
        prior_hash = row["entry_hash"]

    return True
