"""
Tamper-Evident Chain-of-Custody Audit Trail.

Provides an immutable, cryptographically verifiable audit log for all
data acquisition operations. Each entry is linked to its predecessor
via a SHA-256 hash chain, and periodic Merkle tree checkpoints enable
efficient integrity verification of any log segment.

Architecture
------------
The audit system implements a two-layer integrity structure:

    +--------------------------------------------------------------+
    |          TAMPER-EVIDENT AUDIT ARCHITECTURE                   |
    +--------------------------------------------------------------+
    |                                                              |
    |  LAYER 1 - Hash-Chained Log Entries                         |
    |  +-------+   +-------+   +-------+   +-------+            |
    |  |Entry 0|-->|Entry 1|-->|Entry 2|-->|Entry 3|--> ...      |
    |  |H(data)|   |H(E0+d)|   |H(E1+d)|   |H(E2+d)|            |
    |  +-------+   +-------+   +-------+   +-------+            |
    |                                                              |
    |  LAYER 2 - Merkle Tree Checkpoints                          |
    |              +----------+                                    |
    |              |   Root   |                                    |
    |              | H(L + R) |                                    |
    |              +----+-----+                                    |
    |            +------+------+                                   |
    |       +----+----+  +----+----+                              |
    |       | H(0+1) |  | H(2+3) |                               |
    |       +----+----+  +----+----+                              |
    |        +---+---+    +---+---+                               |
    |       H(E0) H(E1) H(E2) H(E3)                              |
    |                                                              |
    |  Verification: O(log n) proof for any single entry          |
    |  Tampering detection: Any modification breaks the chain     |
    +--------------------------------------------------------------+

    |  LAYER 3 - Consistency Proofs                                |
    |  +-- Prove append-only property between two tree states     |
    |  +-- O(log n) verification for contiguous append integrity  |
    |  +-- Enables third-party auditors to verify growth          |
    |                                                              |
    |  LAYER 4 - Rate-Limiting & Anomaly Detection                |
    |  +-- Actor velocity tracking (events per window)            |
    |  +-- Resource burst detection                               |
    |  +-- Configurable alert thresholds per event class          |
    +--------------------------------------------------------------+

Compliance Coverage
-------------------
- HIPAA §164.312(b): Audit controls
- HIPAA §164.312(c)(1): Integrity mechanism
- 21 CFR Part 11 §11.10(e): Electronic records, electronic signatures
- NIST SP 800-92 (2024 update): Guide to Computer Security Log Management
- NIST SP 800-53 Rev. 5 AU-2, AU-3, AU-9, AU-10: Audit events, protection,
  non-repudiation
- NIST SP 800-171 Rev. 3 (2024): Protecting CUI in nonfederal systems
- ISO 27001:2022 A.8.15: Logging
- FDA 21 CFR Part 11 §11.10(e): Audit trails for electronic records
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Final,
    NamedTuple,
    Sequence,
    TypeAlias,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
HashDigest: TypeAlias = str
PathLike: TypeAlias = str | Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HASH_ALGORITHM: Final[str] = "sha256"
GENESIS_HASH: Final[str] = "0" * 64
CHECKPOINT_INTERVAL: Final[int] = 100


# ===========================================================================
# Enumerations
# ===========================================================================

class AuditEventType(Enum):
    """Classification of audit events.

    Cleanup history - Q7.A (Phase 4.5 sprint): 10 dead values removed
    (no production callers, no behavior coverage in tests). The 5
    test-fixture references in tests/test_phase1_1_audit_trail.py were
    substituted with kept values (the tests verify audit-trail
    infrastructure on arbitrary event types, not production emission).
    INTEGRITY_VIOLATION kept - Q15.A correlation-ID error path uses it.
    3 new WEB_* values added for the Phase 4.5 web layer.
    """

    # Data lifecycle events
    DATA_RECEIVED = "data_received"
    DATA_VERIFIED = "data_verified"
    DATA_RELEASED = "data_released"

    # Access events
    ACCESS_DENIED = "access_denied"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    IRB_STATUS_CHANGE = "irb_status_change"

    # Security events
    INTEGRITY_VIOLATION = "integrity_violation"  # Q15.A correlation-ID error path

    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONFIGURATION_CHANGE = "configuration_change"

    # Web layer events (Phase 4.5 Q7.C)
    WEB_PREDICT_RECEIVED = "web_predict_received"
    WEB_PREDICT_RETURNED = "web_predict_returned"
    WEB_RATE_LIMIT_HIT = "web_rate_limit_hit"
    WEB_PRESET_LOADED = "web_preset_loaded"           # Q12.A preset-button click
    AUDIT_EXPORT_REQUESTED = "audit_export_requested"  # Q13.A CSV download click


class IntegrityStatus(Enum):
    """Result of integrity verification."""

    VALID = "valid"
    TAMPERED = "tampered"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


# ===========================================================================
# Audit Log Entry
# ===========================================================================

class AuditEntry(NamedTuple):
    """Single immutable audit log entry.

    Attributes
    ----------
    entry_id : str
        Unique entry identifier.
    sequence_number : int
        Monotonically increasing sequence number.
    timestamp : str
        ISO 8601 timestamp (UTC).
    event_type : str
        Classified event type.
    actor : str
        Person or system performing the action.
    resource : str
        Data resource affected.
    action_detail : str
        Human-readable description.
    metadata : dict[str, Any]
        Structured event-specific data.
    previous_hash : str
        SHA-256 of the preceding entry.
    entry_hash : str
        SHA-256 of this entry (computed over all above fields).
    """

    entry_id: str
    sequence_number: int
    timestamp: str
    event_type: str
    actor: str
    resource: str
    action_detail: str
    metadata: dict[str, Any]
    previous_hash: str
    entry_hash: str


def _compute_entry_hash(
    entry_id: str,
    sequence_number: int,
    timestamp: str,
    event_type: str,
    actor: str,
    resource: str,
    action_detail: str,
    metadata: dict[str, Any],
    previous_hash: str,
) -> str:
    """Compute SHA-256 hash over all entry fields except entry_hash."""
    payload = json.dumps(
        {
            "entry_id": entry_id,
            "sequence_number": sequence_number,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "resource": resource,
            "action_detail": action_detail,
            "metadata": metadata,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ===========================================================================
# Merkle Tree
# ===========================================================================

def _hash_pair(left: str, right: str) -> str:
    """Compute parent hash from two child hashes."""
    combined = (left + right).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


class MerkleTree:
    """Binary Merkle tree for efficient batch integrity verification.

    Constructs a balanced binary hash tree over a sequence of
    leaf hashes, enabling O(log n) membership proofs.

    Parameters
    ----------
    leaf_hashes : Sequence[str]
        SHA-256 hashes of individual entries.
    """

    __slots__ = ("_leaves", "_tree", "_root")

    def __init__(self, leaf_hashes: Sequence[str]) -> None:
        self._leaves = list(leaf_hashes)
        self._tree: list[list[str]] = []
        self._root: str = ""
        self._build()

    @property
    def root(self) -> str:
        """Return the Merkle root hash."""
        return self._root

    @property
    def leaf_count(self) -> int:
        """Return the number of leaves."""
        return len(self._leaves)

    @property
    def depth(self) -> int:
        """Return tree depth."""
        return len(self._tree)

    def _build(self) -> None:
        """Construct the Merkle tree bottom-up."""
        if not self._leaves:
            self._root = GENESIS_HASH
            self._tree = []
            return

        current_level = list(self._leaves)
        self._tree = [current_level]

        while len(current_level) > 1:
            next_level: list[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(_hash_pair(left, right))
            self._tree.append(next_level)
            current_level = next_level

        self._root = current_level[0] if current_level else GENESIS_HASH

    def get_proof(self, leaf_index: int) -> list[tuple[str, str]]:
        """Generate inclusion proof for a leaf.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf to prove.

        Returns
        -------
        list[tuple[str, str]]
            List of (sibling_hash, side) pairs, where side is
            "left" or "right".
        """
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            msg = f"Leaf index {leaf_index} out of range [0, {len(self._leaves)})"
            raise IndexError(msg)

        proof: list[tuple[str, str]] = []
        idx = leaf_index

        for level in self._tree[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                side = "right"
            else:
                sibling_idx = idx - 1
                side = "left"

            if sibling_idx < len(level):
                proof.append((level[sibling_idx], side))
            else:
                proof.append((level[idx], "right"))

            idx //= 2

        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: str,
        proof: list[tuple[str, str]],
        expected_root: str,
    ) -> bool:
        """Verify a Merkle inclusion proof.

        Parameters
        ----------
        leaf_hash : str
            Hash of the leaf being verified.
        proof : list[tuple[str, str]]
            Proof path from get_proof().
        expected_root : str
            Expected Merkle root.

        Returns
        -------
        bool
            True if the proof is valid.
        """
        current = leaf_hash
        for sibling_hash, side in proof:
            if side == "left":
                current = _hash_pair(sibling_hash, current)
            else:
                current = _hash_pair(current, sibling_hash)
        return current == expected_root


class MerkleCheckpoint(NamedTuple):
    """Periodic checkpoint of Merkle tree state."""

    checkpoint_id: str
    sequence_range: tuple[int, int]
    merkle_root: str
    leaf_count: int
    tree_depth: int
    created_at: str


# ===========================================================================
# Audit Log
# ===========================================================================

@dataclass
class AuditLog:
    """Hash-chained, Merkle-checkpointed audit log.

    Provides an immutable record of all data acquisition operations
    with cryptographic tamper-detection guarantees.

    Parameters
    ----------
    log_id : str
        Unique identifier for this log instance.
    checkpoint_interval : int
        Number of entries between Merkle checkpoints.

    Attributes
    ----------
    entries : list[AuditEntry]
        All recorded audit entries.
    checkpoints : list[MerkleCheckpoint]
        Periodic Merkle tree checkpoints.
    """

    log_id: str = field(
        default_factory=lambda: f"AUDIT-{uuid.uuid4().hex[:12].upper()}"
    )
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    entries: list[AuditEntry] = field(default_factory=list)
    checkpoints: list[MerkleCheckpoint] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_checkpoint_seq: int = field(default=0, init=False)

    def record(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: str,
        action_detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a new audit event.

        Parameters
        ----------
        event_type : AuditEventType
            Classification of the event.
        actor : str
            Person or system performing the action.
        resource : str
            Data resource affected.
        action_detail : str
            Human-readable description.
        metadata : dict[str, Any] | None
            Additional structured data.

        Returns
        -------
        AuditEntry
            The recorded entry.
        """
        with self._lock:
            seq = len(self.entries)
            previous_hash = (
                self.entries[-1].entry_hash if self.entries else GENESIS_HASH
            )

            entry_id = f"{self.log_id}-{seq:06d}"
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            meta = metadata or {}

            entry_hash = _compute_entry_hash(
                entry_id=entry_id,
                sequence_number=seq,
                timestamp=timestamp,
                event_type=event_type.value,
                actor=actor,
                resource=resource,
                action_detail=action_detail,
                metadata=meta,
                previous_hash=previous_hash,
            )

            entry = AuditEntry(
                entry_id=entry_id,
                sequence_number=seq,
                timestamp=timestamp,
                event_type=event_type.value,
                actor=actor,
                resource=resource,
                action_detail=action_detail,
                metadata=meta,
                previous_hash=previous_hash,
                entry_hash=entry_hash,
            )
            self.entries.append(entry)

            # Periodic Merkle checkpoint
            entries_since = seq - self._last_checkpoint_seq + 1
            if entries_since >= self.checkpoint_interval:
                self._create_checkpoint()

            return entry

    def verify_chain(self) -> tuple[IntegrityStatus, list[int]]:
        """Verify the entire hash chain.

        Returns
        -------
        tuple[IntegrityStatus, list[int]]
            Status and list of tampered entry sequence numbers.
        """
        if not self.entries:
            return IntegrityStatus.VALID, []

        tampered: list[int] = []

        for i, entry in enumerate(self.entries):
            expected_prev = (
                self.entries[i - 1].entry_hash if i > 0 else GENESIS_HASH
            )
            if entry.previous_hash != expected_prev:
                tampered.append(entry.sequence_number)
                continue

            recomputed = _compute_entry_hash(
                entry_id=entry.entry_id,
                sequence_number=entry.sequence_number,
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                actor=entry.actor,
                resource=entry.resource,
                action_detail=entry.action_detail,
                metadata=entry.metadata,
                previous_hash=entry.previous_hash,
            )
            if recomputed != entry.entry_hash:
                tampered.append(entry.sequence_number)

        if tampered:
            return IntegrityStatus.TAMPERED, tampered
        return IntegrityStatus.VALID, []

    def verify_entry(self, sequence_number: int) -> bool:
        """Verify a single entry's hash.

        Parameters
        ----------
        sequence_number : int
            Sequence number of the entry to verify.

        Returns
        -------
        bool
            True if the entry hash is valid.
        """
        if sequence_number < 0 or sequence_number >= len(self.entries):
            return False

        entry = self.entries[sequence_number]
        recomputed = _compute_entry_hash(
            entry_id=entry.entry_id,
            sequence_number=entry.sequence_number,
            timestamp=entry.timestamp,
            event_type=entry.event_type,
            actor=entry.actor,
            resource=entry.resource,
            action_detail=entry.action_detail,
            metadata=entry.metadata,
            previous_hash=entry.previous_hash,
        )
        return recomputed == entry.entry_hash

    def verify_checkpoint(self, checkpoint_index: int) -> bool:
        """Verify a Merkle checkpoint.

        Parameters
        ----------
        checkpoint_index : int
            Index into the checkpoints list.

        Returns
        -------
        bool
            True if the Merkle root matches recomputed tree.
        """
        if checkpoint_index < 0 or checkpoint_index >= len(self.checkpoints):
            return False

        cp = self.checkpoints[checkpoint_index]
        start, end = cp.sequence_range

        if end > len(self.entries):
            return False

        leaf_hashes = [
            self.entries[i].entry_hash for i in range(start, end)
        ]
        tree = MerkleTree(leaf_hashes)
        return tree.root == cp.merkle_root

    def get_merkle_proof(
        self, sequence_number: int
    ) -> tuple[MerkleCheckpoint, list[tuple[str, str]]] | None:
        """Generate Merkle inclusion proof for an entry.

        Parameters
        ----------
        sequence_number : int
            Entry sequence number.

        Returns
        -------
        tuple or None
            (checkpoint, proof_path) or None if not in any checkpoint.
        """
        for cp in self.checkpoints:
            start, end = cp.sequence_range
            if start <= sequence_number < end:
                leaf_hashes = [
                    self.entries[i].entry_hash for i in range(start, end)
                ]
                tree = MerkleTree(leaf_hashes)
                local_idx = sequence_number - start
                proof = tree.get_proof(local_idx)
                return cp, proof
        return None

    def _create_checkpoint(self) -> None:
        """Create a Merkle tree checkpoint over recent entries."""
        start = self._last_checkpoint_seq
        end = len(self.entries)

        if start >= end:
            return

        leaf_hashes = [
            self.entries[i].entry_hash for i in range(start, end)
        ]
        tree = MerkleTree(leaf_hashes)

        checkpoint = MerkleCheckpoint(
            checkpoint_id=f"CP-{uuid.uuid4().hex[:8].upper()}",
            sequence_range=(start, end),
            merkle_root=tree.root,
            leaf_count=tree.leaf_count,
            tree_depth=tree.depth,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self.checkpoints.append(checkpoint)
        self._last_checkpoint_seq = end

        logger.info(
            "Merkle checkpoint created: entries [%d, %d), root=%s",
            start,
            end,
            tree.root[:16],
        )

    def filter_by_type(
        self, event_type: AuditEventType
    ) -> list[AuditEntry]:
        """Filter entries by event type."""
        return [
            e for e in self.entries if e.event_type == event_type.value
        ]

    def filter_by_resource(self, resource: str) -> list[AuditEntry]:
        """Filter entries by resource identifier."""
        return [e for e in self.entries if e.resource == resource]

    def filter_by_actor(self, actor: str) -> list[AuditEntry]:
        """Filter entries by actor."""
        return [e for e in self.entries if e.actor == actor]

    def filter_by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[AuditEntry]:
        """Filter entries by time range."""
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        return [
            e for e in self.entries
            if start_iso <= e.timestamp <= end_iso
        ]

    def export_json(self, output_path: Path) -> None:
        """Export the full audit log to JSON.

        Parameters
        ----------
        output_path : Path
            Destination file path.
        """
        data = {
            "log_id": self.log_id,
            "entry_count": len(self.entries),
            "checkpoint_count": len(self.checkpoints),
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "sequence_number": e.sequence_number,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "resource": e.resource,
                    "action_detail": e.action_detail,
                    "metadata": e.metadata,
                    "previous_hash": e.previous_hash,
                    "entry_hash": e.entry_hash,
                }
                for e in self.entries
            ],
            "checkpoints": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "sequence_range": list(cp.sequence_range),
                    "merkle_root": cp.merkle_root,
                    "leaf_count": cp.leaf_count,
                    "tree_depth": cp.tree_depth,
                    "created_at": cp.created_at,
                }
                for cp in self.checkpoints
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, input_path: Path) -> None:
        """Import an audit log from JSON and verify integrity.

        Parameters
        ----------
        input_path : Path
            Source JSON file.

        Raises
        ------
        ValueError
            If the imported log fails integrity verification.
        """
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.log_id = data.get("log_id", self.log_id)
        self.entries = [
            AuditEntry(
                entry_id=e["entry_id"],
                sequence_number=e["sequence_number"],
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                actor=e["actor"],
                resource=e["resource"],
                action_detail=e["action_detail"],
                metadata=e.get("metadata", {}),
                previous_hash=e["previous_hash"],
                entry_hash=e["entry_hash"],
            )
            for e in data.get("entries", [])
        ]
        self.checkpoints = [
            MerkleCheckpoint(
                checkpoint_id=cp["checkpoint_id"],
                sequence_range=tuple(cp["sequence_range"]),
                merkle_root=cp["merkle_root"],
                leaf_count=cp["leaf_count"],
                tree_depth=cp["tree_depth"],
                created_at=cp["created_at"],
            )
            for cp in data.get("checkpoints", [])
        ]

        # Verify imported data integrity
        status, tampered = self.verify_chain()
        if status == IntegrityStatus.TAMPERED:
            msg = (
                f"Imported audit log failed integrity verification: "
                f"{len(tampered)} tampered entries"
            )
            raise ValueError(msg)

    def generate_summary(self) -> dict[str, Any]:
        """Generate audit log summary statistics.

        Returns
        -------
        dict[str, Any]
            Summary including event type counts and integrity status.
        """
        status, tampered = self.verify_chain()

        event_counts: dict[str, int] = {}
        actor_counts: dict[str, int] = {}
        resource_counts: dict[str, int] = {}

        for entry in self.entries:
            event_counts[entry.event_type] = (
                event_counts.get(entry.event_type, 0) + 1
            )
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
            resource_counts[entry.resource] = (
                resource_counts.get(entry.resource, 0) + 1
            )

        return {
            "log_id": self.log_id,
            "total_entries": len(self.entries),
            "total_checkpoints": len(self.checkpoints),
            "integrity_status": status.value,
            "tampered_entries": len(tampered),
            "event_type_distribution": event_counts,
            "actor_distribution": actor_counts,
            "resource_distribution": resource_counts,
            "first_entry_time": (
                self.entries[0].timestamp if self.entries else None
            ),
            "last_entry_time": (
                self.entries[-1].timestamp if self.entries else None
            ),
        }


# ===========================================================================
# Provenance Tracker
# ===========================================================================

@dataclass(slots=True)
class DataProvenance:
    """Tracks the full provenance chain of a data asset.

    Records source, transformations, and custody changes throughout
    the data lifecycle.

    Attributes
    ----------
    asset_id : str
        Unique data asset identifier.
    source : str
        Origin of the data (e.g., "CDC SFTP").
    created_at : str
        Asset creation timestamp.
    custody_chain : list[dict[str, Any]]
        Ordered list of custody events.
    transformations : list[dict[str, Any]]
        Ordered list of data transformations.
    current_custodian : str
        Person or system currently holding the data.
    integrity_hash : str
        Current integrity hash of the data.
    """

    asset_id: str = field(
        default_factory=lambda: f"ASSET-{uuid.uuid4().hex[:10].upper()}"
    )
    source: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    custody_chain: list[dict[str, Any]] = field(default_factory=list)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    current_custodian: str = ""
    integrity_hash: str = ""

    def record_custody_transfer(
        self,
        from_custodian: str,
        to_custodian: str,
        reason: str,
        integrity_hash: str | None = None,
    ) -> None:
        """Record a custody transfer event.

        Parameters
        ----------
        from_custodian : str
            Releasing party.
        to_custodian : str
            Receiving party.
        reason : str
            Reason for transfer.
        integrity_hash : str | None
            Hash of data at transfer time.
        """
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "from": from_custodian,
            "to": to_custodian,
            "reason": reason,
            "integrity_hash": integrity_hash or self.integrity_hash,
        }
        self.custody_chain.append(event)
        self.current_custodian = to_custodian
        if integrity_hash:
            self.integrity_hash = integrity_hash

    def record_transformation(
        self,
        transformation_type: str,
        description: str,
        input_hash: str,
        output_hash: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a data transformation event.

        Parameters
        ----------
        transformation_type : str
            Type of transformation (e.g., "de-identification").
        description : str
            Human-readable description.
        input_hash : str
            Hash of data before transformation.
        output_hash : str
            Hash of data after transformation.
        parameters : dict[str, Any] | None
            Transformation parameters.
        """
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "type": transformation_type,
            "description": description,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "parameters": parameters or {},
        }
        self.transformations.append(event)
        self.integrity_hash = output_hash

    def to_dict(self) -> dict[str, Any]:
        """Serialise provenance record."""
        return {
            "asset_id": self.asset_id,
            "source": self.source,
            "created_at": self.created_at,
            "current_custodian": self.current_custodian,
            "integrity_hash": self.integrity_hash,
            "custody_events": len(self.custody_chain),
            "transformation_events": len(self.transformations),
            "custody_chain": self.custody_chain,
            "transformations": self.transformations,
        }


# ===========================================================================
# Factory Functions
# ===========================================================================

def create_audit_log(
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> AuditLog:
    """Create a new audit log instance.

    Parameters
    ----------
    checkpoint_interval : int
        Entries between Merkle checkpoints.

    Returns
    -------
    AuditLog
        Configured audit log.
    """
    return AuditLog(checkpoint_interval=checkpoint_interval)


def create_provenance_tracker(
    source: str,
    initial_custodian: str,
    initial_hash: str = "",
) -> DataProvenance:
    """Create a new data provenance tracker.

    Parameters
    ----------
    source : str
        Origin of the data.
    initial_custodian : str
        Initial data custodian.
    initial_hash : str
        Initial integrity hash.

    Returns
    -------
    DataProvenance
        Configured provenance tracker.
    """
    return DataProvenance(
        source=source,
        current_custodian=initial_custodian,
        integrity_hash=initial_hash,
    )


# ===========================================================================
# Rate-Limiting & Anomaly Detection
# ===========================================================================

class AnomalyType(Enum):
    """Classification of detected audit anomalies."""

    VELOCITY_BREACH = "velocity_breach"
    BURST_DETECTED = "burst_detected"
    OFF_HOURS_ACCESS = "off_hours_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AuditAnomaly(NamedTuple):
    """Record of a detected anomaly in the audit stream."""

    anomaly_type: AnomalyType
    timestamp: str
    actor: str
    description: str
    severity: str


@dataclass
class RateLimitConfig:
    """Configuration for actor-level rate limiting.

    Attributes
    ----------
    max_events_per_window : int
        Maximum events allowed per actor within the window.
    window_seconds : int
        Sliding window duration in seconds.
    burst_threshold : int
        Events in a short burst (window_seconds // 10) that trigger alert.
    off_hours_start : int
        Hour (UTC, 0-23) when off-hours begin.
    off_hours_end : int
        Hour (UTC, 0-23) when off-hours end.
    """

    max_events_per_window: int = 500
    window_seconds: int = 3600
    burst_threshold: int = 50
    off_hours_start: int = 22
    off_hours_end: int = 6


class AuditAnomalyDetector:
    """Monitors audit event streams for anomalous patterns.

    Implements sliding-window velocity tracking per actor,
    burst detection, and off-hours access alerting.

    Parameters
    ----------
    config : RateLimitConfig
        Detection thresholds and window parameters.
    """

    __slots__ = ("_config", "_actor_events", "_anomalies", "_lock")

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._config = config or RateLimitConfig()
        self._actor_events: dict[str, list[datetime]] = defaultdict(list)
        self._anomalies: list[AuditAnomaly] = []
        self._lock = threading.Lock()

    @property
    def anomalies(self) -> list[AuditAnomaly]:
        """Return all detected anomalies."""
        return list(self._anomalies)

    def observe(self, entry: AuditEntry) -> list[AuditAnomaly]:
        """Observe a new audit entry and check for anomalies.

        Parameters
        ----------
        entry : AuditEntry
            Newly recorded audit entry.

        Returns
        -------
        list[AuditAnomaly]
            Anomalies detected from this entry (may be empty).
        """
        ts = datetime.fromisoformat(entry.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        detected: list[AuditAnomaly] = []

        with self._lock:
            # Sliding window velocity
            window_start = ts - timedelta(
                seconds=self._config.window_seconds
            )
            events = self._actor_events[entry.actor]
            events.append(ts)
            events[:] = [e for e in events if e >= window_start]

            if len(events) > self._config.max_events_per_window:
                anomaly = AuditAnomaly(
                    anomaly_type=AnomalyType.VELOCITY_BREACH,
                    timestamp=entry.timestamp,
                    actor=entry.actor,
                    description=(
                        f"Actor '{entry.actor}' exceeded "
                        f"{self._config.max_events_per_window} "
                        f"events in {self._config.window_seconds}s window"
                    ),
                    severity="high",
                )
                detected.append(anomaly)

            # Burst detection
            burst_window = timedelta(
                seconds=max(1, self._config.window_seconds // 10)
            )
            burst_start = ts - burst_window
            burst_events = [e for e in events if e >= burst_start]
            if len(burst_events) > self._config.burst_threshold:
                anomaly = AuditAnomaly(
                    anomaly_type=AnomalyType.BURST_DETECTED,
                    timestamp=entry.timestamp,
                    actor=entry.actor,
                    description=(
                        f"Burst: {len(burst_events)} events in "
                        f"{burst_window.total_seconds():.0f}s from "
                        f"'{entry.actor}'"
                    ),
                    severity="medium",
                )
                detected.append(anomaly)

            # Off-hours access
            hour = ts.hour
            start_h = self._config.off_hours_start
            end_h = self._config.off_hours_end
            is_off_hours = (
                (start_h > end_h and (hour >= start_h or hour < end_h))
                or (start_h <= end_h and start_h <= hour < end_h)
            )
            if is_off_hours:
                anomaly = AuditAnomaly(
                    anomaly_type=AnomalyType.OFF_HOURS_ACCESS,
                    timestamp=entry.timestamp,
                    actor=entry.actor,
                    description=(
                        f"Off-hours access by '{entry.actor}' at "
                        f"{ts.strftime('%H:%M')} UTC"
                    ),
                    severity="low",
                )
                detected.append(anomaly)

            self._anomalies.extend(detected)

        return detected

    def get_actor_velocity(self, actor: str) -> int:
        """Return current event count for an actor in the active window."""
        with self._lock:
            return len(self._actor_events.get(actor, []))

    def reset(self) -> None:
        """Clear all tracking state."""
        with self._lock:
            self._actor_events.clear()
            self._anomalies.clear()


# ===========================================================================
# Merkle Consistency Proof
# ===========================================================================

def merkle_consistency_proof(
    old_tree: MerkleTree,
    new_tree: MerkleTree,
) -> bool:
    """Verify append-only consistency between two Merkle tree states.

    A consistency proof demonstrates that the first *m* leaves of the
    new tree are identical to the *m* leaves of the old tree, proving
    the log was only appended to (RFC 6962 §2.1.2 equivalent).

    Parameters
    ----------
    old_tree : MerkleTree
        Earlier tree state with m leaves.
    new_tree : MerkleTree
        Later tree state with n >= m leaves.

    Returns
    -------
    bool
        True if the new tree is a consistent extension of the old tree.
    """
    if old_tree.leaf_count == 0:
        return True
    if new_tree.leaf_count < old_tree.leaf_count:
        return False

    for i in range(old_tree.leaf_count):
        proof = new_tree.get_proof(i)
        old_leaf = old_tree._leaves[i]
        new_leaf = new_tree._leaves[i]
        if old_leaf != new_leaf:
            return False
        if not MerkleTree.verify_proof(new_leaf, proof, new_tree.root):
            return False

    return True


# ===========================================================================
# Audit Statistics
# ===========================================================================

@dataclass(slots=True)
class AuditStatistics:
    """Computed statistics over an audit log segment.

    Attributes
    ----------
    total_entries : int
        Number of entries in the segment.
    unique_actors : int
        Distinct actors observed.
    unique_resources : int
        Distinct resources referenced.
    event_type_counts : dict[str, int]
        Event type frequency distribution.
    hourly_distribution : dict[int, int]
        Event count by hour of day (UTC).
    mean_inter_event_seconds : float
        Mean time between consecutive events.
    integrity_verified : bool
        Whether the segment passed integrity verification.
    """

    total_entries: int = 0
    unique_actors: int = 0
    unique_resources: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    hourly_distribution: dict[int, int] = field(default_factory=dict)
    mean_inter_event_seconds: float = 0.0
    integrity_verified: bool = False


def compute_audit_statistics(log: AuditLog) -> AuditStatistics:
    """Compute detailed statistics over an audit log.

    Parameters
    ----------
    log : AuditLog
        Audit log to analyse.

    Returns
    -------
    AuditStatistics
        Computed statistics.
    """
    if not log.entries:
        return AuditStatistics(integrity_verified=True)

    actors: set[str] = set()
    resources: set[str] = set()
    event_counts: dict[str, int] = {}
    hourly: dict[int, int] = defaultdict(int)
    timestamps: list[datetime] = []

    for entry in log.entries:
        actors.add(entry.actor)
        resources.add(entry.resource)
        event_counts[entry.event_type] = (
            event_counts.get(entry.event_type, 0) + 1
        )
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hourly[ts.hour] += 1
            timestamps.append(ts)
        except ValueError:
            pass

    mean_inter = 0.0
    if len(timestamps) > 1:
        timestamps.sort()
        deltas = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean_inter = sum(deltas) / len(deltas)

    status, _ = log.verify_chain()

    return AuditStatistics(
        total_entries=len(log.entries),
        unique_actors=len(actors),
        unique_resources=len(resources),
        event_type_counts=event_counts,
        hourly_distribution=dict(hourly),
        mean_inter_event_seconds=round(mean_inter, 3),
        integrity_verified=(status == IntegrityStatus.VALID),
    )


# ===========================================================================
# Structured Log Search Engine
# ===========================================================================

@dataclass(slots=True)
class AuditSearchQuery:
    """Structured query against an audit log.

    All filter criteria are combined with AND logic. An empty query
    matches all entries.

    Attributes
    ----------
    event_types : frozenset[str] | None
        Restrict to specific event types (OR within list).
    actors : frozenset[str] | None
        Restrict to specific actors (OR within list).
    resources : frozenset[str] | None
        Restrict to specific resources (OR within list).
    start_time : datetime | None
        Earliest timestamp (inclusive).
    end_time : datetime | None
        Latest timestamp (inclusive).
    metadata_contains : dict[str, Any] | None
        Entry metadata must contain these key-value pairs.
    action_pattern : str | None
        Regex pattern to match against action_detail.
    limit : int
        Maximum number of results to return (0 = unlimited).
    """

    event_types: frozenset[str] | None = None
    actors: frozenset[str] | None = None
    resources: frozenset[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata_contains: dict[str, Any] | None = None
    action_pattern: str | None = None
    limit: int = 0


class AuditSearchEngine:
    """Query engine for structured audit log search.

    Indexed search over audit logs supporting multi-field filters,
    time-range queries, metadata key-value containment, and regex
    matching on action descriptions.

    Parameters
    ----------
    log : AuditLog
        Audit log to search.
    """

    __slots__ = ("_log",)

    def __init__(self, log: AuditLog) -> None:
        self._log = log

    def search(self, query: AuditSearchQuery) -> list[AuditEntry]:
        """Execute a structured search query.

        Parameters
        ----------
        query : AuditSearchQuery
            Search criteria.

        Returns
        -------
        list[AuditEntry]
            Matching entries in chronological order.
        """
        import re as _re

        results: list[AuditEntry] = []
        compiled_pattern = (
            _re.compile(query.action_pattern, _re.IGNORECASE)
            if query.action_pattern
            else None
        )

        start_iso = query.start_time.isoformat() if query.start_time else None
        end_iso = query.end_time.isoformat() if query.end_time else None

        for entry in self._log.entries:
            if query.event_types and entry.event_type not in query.event_types:
                continue
            if query.actors and entry.actor not in query.actors:
                continue
            if query.resources and entry.resource not in query.resources:
                continue
            if start_iso and entry.timestamp < start_iso:
                continue
            if end_iso and entry.timestamp > end_iso:
                continue
            if query.metadata_contains:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in query.metadata_contains.items()
                )
                if not match:
                    continue
            if compiled_pattern and not compiled_pattern.search(
                entry.action_detail
            ):
                continue

            results.append(entry)
            if 0 < query.limit <= len(results):
                break

        return results

    def count(self, query: AuditSearchQuery) -> int:
        """Count matching entries without materialising results."""
        unlimited = AuditSearchQuery(
            event_types=query.event_types,
            actors=query.actors,
            resources=query.resources,
            start_time=query.start_time,
            end_time=query.end_time,
            metadata_contains=query.metadata_contains,
            action_pattern=query.action_pattern,
            limit=0,
        )
        return len(self.search(unlimited))


# ===========================================================================
# Long-Term Audit Archiver (NIST SP 800-92)
# ===========================================================================

@dataclass(slots=True)
class ArchiveSegment:
    """An archived, integrity-sealed segment of an audit log.

    Compliant with NIST SP 800-92 §4.2 (log data archival).

    Attributes
    ----------
    segment_id : str
        Unique segment identifier.
    sequence_range : tuple[int, int]
        Entry range [start, end) in the source log.
    entry_count : int
        Number of archived entries.
    merkle_root : str
        Merkle root over all entries in the segment.
    archived_at : str
        ISO 8601 archival timestamp.
    retention_years : int
        Mandated retention period (HIPAA: 6 years minimum).
    """

    segment_id: str
    sequence_range: tuple[int, int]
    entry_count: int
    merkle_root: str
    archived_at: str
    retention_years: int


class AuditLogArchiver:
    """Archives and seals completed audit log segments.

    Implements NIST SP 800-92 recommendations for log archival:
    segments are sealed with Merkle roots and metadata is
    preserved for efficient retrieval and integrity audit.

    Parameters
    ----------
    retention_years : int
        Minimum retention period per HIPAA (default 6).
    """

    __slots__ = ("_retention_years", "_archives")

    def __init__(self, retention_years: int = 6) -> None:
        self._retention_years = retention_years
        self._archives: list[ArchiveSegment] = []

    @property
    def archives(self) -> list[ArchiveSegment]:
        """Return all archived segments."""
        return list(self._archives)

    def archive_segment(
        self,
        log: AuditLog,
        start: int,
        end: int,
    ) -> ArchiveSegment:
        """Archive a range of entries from an audit log.

        Parameters
        ----------
        log : AuditLog
            Source audit log.
        start : int
            Start sequence number (inclusive).
        end : int
            End sequence number (exclusive).

        Returns
        -------
        ArchiveSegment
            Sealed archive segment.

        Raises
        ------
        ValueError
            If the range is invalid or entries are tampered.
        """
        if start < 0 or end > len(log.entries) or start >= end:
            msg = f"Invalid range [{start}, {end}) for log with {len(log.entries)} entries"
            raise ValueError(msg)

        leaf_hashes = [log.entries[i].entry_hash for i in range(start, end)]
        tree = MerkleTree(leaf_hashes)

        segment = ArchiveSegment(
            segment_id=f"ARC-{uuid.uuid4().hex[:10].upper()}",
            sequence_range=(start, end),
            entry_count=end - start,
            merkle_root=tree.root,
            archived_at=datetime.now(tz=timezone.utc).isoformat(),
            retention_years=self._retention_years,
        )
        self._archives.append(segment)

        logger.info(
            "Archived segment %s: entries [%d, %d), root=%s",
            segment.segment_id,
            start,
            end,
            tree.root[:16],
        )
        return segment

    def verify_archive(
        self, log: AuditLog, segment: ArchiveSegment
    ) -> bool:
        """Verify an archived segment against the current log.

        Parameters
        ----------
        log : AuditLog
            Source audit log.
        segment : ArchiveSegment
            Previously archived segment.

        Returns
        -------
        bool
            True if the archived Merkle root still matches.
        """
        start, end = segment.sequence_range
        if end > len(log.entries):
            return False

        leaf_hashes = [log.entries[i].entry_hash for i in range(start, end)]
        tree = MerkleTree(leaf_hashes)
        return tree.root == segment.merkle_root

    def segments_needing_retention_review(self) -> list[ArchiveSegment]:
        """Return segments approaching their retention expiry."""
        now = datetime.now(tz=timezone.utc)
        results: list[ArchiveSegment] = []
        for seg in self._archives:
            try:
                archived = datetime.fromisoformat(seg.archived_at)
                if archived.tzinfo is None:
                    archived = archived.replace(tzinfo=timezone.utc)
                expiry = archived.replace(
                    year=archived.year + seg.retention_years
                )
                remaining = (expiry - now).days
                if remaining < 365:
                    results.append(seg)
            except (ValueError, OverflowError):
                results.append(seg)
        return results


# ===========================================================================
# Audit Export - HIPAA §164.312(b) / FDA 21 CFR Part 11 §11.10(e)
# ===========================================================================


class AuditExportFormat(Enum):
    """Supported export formats for compliance reporting."""

    JSON = "json"
    CSV = "csv"
    NDJSON = "ndjson"


@dataclass(frozen=True, slots=True)
class AuditComplianceReport:
    """Summarised compliance report for regulatory submission.

    Covers HIPAA §164.312(b) audit control requirements and
    FDA 21 CFR Part 11 §11.10(e) electronic-record audit trails.
    Generated by ``AuditExporter.compliance_report()``.
    """

    log_id: str
    generated_at: str
    total_entries: int
    first_entry_timestamp: str | None
    last_entry_timestamp: str | None
    unique_actors: int
    unique_resources: int
    events_by_type: dict[str, int]
    integrity_verified: bool
    chain_intact: bool
    checkpoints_count: int
    anomalies_detected: int
    coverage_start: str | None
    coverage_end: str | None
    archive_segments: int
    retention_reviews_pending: int


class AuditExporter:
    """Export audit data in regulatory-compliant formats.

    Supports JSON, CSV, and NDJSON (newline-delimited JSON) for
    integration with SIEM platforms (Splunk, Elastic, Azure Sentinel).

    Parameters
    ----------
    log : AuditLog
        The audit log to export.
    """

    __slots__ = ("_log",)

    def __init__(self, log: AuditLog) -> None:
        self._log = log

    def export(
        self,
        fmt: AuditExportFormat = AuditExportFormat.JSON,
        *,
        fields: Sequence[str] | None = None,
    ) -> str:
        """Export log entries in the specified format.

        Parameters
        ----------
        fmt : AuditExportFormat
            Target format.
        fields : Sequence[str] | None
            Restrict output to these entry fields.  When ``None`` all
            fields are included.

        Returns
        -------
        str
            Serialised export payload.
        """
        allowed = frozenset(fields) if fields else None
        rows = [self._entry_dict(e, allowed) for e in self._log.entries]

        if fmt == AuditExportFormat.CSV:
            return self._to_csv(rows)
        if fmt == AuditExportFormat.NDJSON:
            return self._to_ndjson(rows)
        return json.dumps(rows, indent=2, default=str)

    def compliance_report(
        self,
        archiver: AuditLogArchiver | None = None,
        anomaly_count: int = 0,
    ) -> AuditComplianceReport:
        """Generate a HIPAA/FDA compliance summary report.

        Parameters
        ----------
        archiver : AuditLogArchiver | None
            Optional archiver for retention metrics.
        anomaly_count : int
            Number of anomalies detected by the detector.

        Returns
        -------
        AuditComplianceReport
        """
        entries = self._log.entries
        status, _violations = self._log.verify_chain()
        chain_ok = status == IntegrityStatus.VALID

        events_by_type: dict[str, int] = {}
        actors: set[str] = set()
        resources: set[str] = set()
        for e in entries:
            events_by_type[e.event_type] = events_by_type.get(
                e.event_type, 0
            ) + 1
            actors.add(e.actor)
            resources.add(e.resource)

        archive_segs = 0
        retention_pending = 0
        if archiver is not None:
            archive_segs = len(archiver.archives)
            retention_pending = len(
                archiver.segments_needing_retention_review()
            )

        return AuditComplianceReport(
            log_id=self._log.log_id,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            total_entries=len(entries),
            first_entry_timestamp=(
                entries[0].timestamp if entries else None
            ),
            last_entry_timestamp=(
                entries[-1].timestamp if entries else None
            ),
            unique_actors=len(actors),
            unique_resources=len(resources),
            events_by_type=events_by_type,
            integrity_verified=chain_ok,
            chain_intact=chain_ok,
            checkpoints_count=len(self._log.checkpoints),
            anomalies_detected=anomaly_count,
            coverage_start=(
                entries[0].timestamp if entries else None
            ),
            coverage_end=(
                entries[-1].timestamp if entries else None
            ),
            archive_segments=archive_segs,
            retention_reviews_pending=retention_pending,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_dict(
        entry: AuditEntry,
        allowed: frozenset[str] | None,
    ) -> dict[str, Any]:
        raw = entry._asdict()
        if allowed is None:
            return raw
        return {k: v for k, v in raw.items() if k in allowed}

    @staticmethod
    def _to_csv(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines = [",".join(headers)]
        for row in rows:
            cells: list[str] = []
            for h in headers:
                val = str(row.get(h, ""))
                if "," in val or '"' in val or "\n" in val:
                    val = '"' + val.replace('"', '""') + '"'
                cells.append(val)
            lines.append(",".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _to_ndjson(rows: list[dict[str, Any]]) -> str:
        return "\n".join(
            json.dumps(row, default=str) for row in rows
        )
