"""
Audit Trail Module - Comprehensive Test Suite.

Tests cover:
  - Hash-chained audit entry recording
  - Chain integrity verification and tamper detection
  - Merkle tree construction, proofs, and verification
  - Merkle checkpoint creation and validation
  - AuditLog filtering (by type, resource, actor, time range)
  - JSON export/import with integrity verification
  - DataProvenance custody chain and transformation tracking
  - Factory functions
  - Summary generation
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ml.data.audit_trail import (
    GENESIS_HASH,
    AuditComplianceReport,
    AuditEntry,
    AuditEventType,
    AuditExportFormat,
    AuditExporter,
    AuditLog,
    AuditLogArchiver,
    AuditSearchEngine,
    AuditSearchQuery,
    DataProvenance,
    IntegrityStatus,
    MerkleCheckpoint,
    MerkleTree,
    _compute_entry_hash,
    _hash_pair,
    create_audit_log,
    create_provenance_tracker,
)


# ===========================================================================
# Hash Function Tests
# ===========================================================================


class TestHashFunctions:
    """Low-level hash computation tests."""

    def test_compute_entry_hash_deterministic(self) -> None:
        h1 = _compute_entry_hash(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
        )
        h2 = _compute_entry_hash(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_different_input_different_hash(self) -> None:
        h1 = _compute_entry_hash("E-001", 0, "t1", "e", "a", "r", "d", {}, GENESIS_HASH)
        h2 = _compute_entry_hash("E-002", 0, "t1", "e", "a", "r", "d", {}, GENESIS_HASH)
        assert h1 != h2

    def test_hash_pair(self) -> None:
        result = _hash_pair("abc", "def")
        assert len(result) == 64
        assert _hash_pair("abc", "def") == _hash_pair("abc", "def")

    def test_hash_pair_order_matters(self) -> None:
        assert _hash_pair("abc", "def") != _hash_pair("def", "abc")


# ===========================================================================
# AuditEntry Tests
# ===========================================================================


class TestAuditEntry:
    """AuditEntry NamedTuple tests."""

    def test_entry_is_namedtuple(self) -> None:
        entry = AuditEntry(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
        )
        assert entry.entry_id == "E-001"
        assert entry.sequence_number == 0

    def test_entry_immutable(self) -> None:
        entry = AuditEntry("E", 0, "t", "e", "a", "r", "d", {}, "p", "h")
        with pytest.raises(AttributeError):
            entry.actor = "modified"  # type: ignore[misc]


# ===========================================================================
# AuditEventType Tests
# ===========================================================================


class TestAuditEventType:
    """Audit event classification enum tests."""

    def test_data_lifecycle_events(self) -> None:
        assert AuditEventType.DATA_RECEIVED.value == "data_received"
        assert AuditEventType.DATA_VERIFIED.value == "data_verified"
        assert AuditEventType.DATA_RELEASED.value == "data_released"

    def test_security_events(self) -> None:
        assert AuditEventType.INTEGRITY_VIOLATION.value == "integrity_violation"

    def test_total_event_types(self) -> None:
        # Cleanup: 3 data-lifecycle + 1 access + 2 compliance + 1 security
        # + 3 system + 5 web (3 + 2 audit_export) = 15
        # (was 20 pre-cleanup)
        assert len(AuditEventType) == 15


# ===========================================================================
# Merkle Tree Tests
# ===========================================================================


class TestMerkleTree:
    """Binary Merkle tree construction and proof verification."""

    def test_empty_tree(self) -> None:
        tree = MerkleTree([])
        assert tree.root == GENESIS_HASH
        assert tree.leaf_count == 0

    def test_single_leaf(self) -> None:
        tree = MerkleTree(["abcdef" * 10 + "abcd"])
        assert tree.leaf_count == 1
        assert tree.root != GENESIS_HASH

    def test_two_leaves(self) -> None:
        h1 = "a" * 64
        h2 = "b" * 64
        tree = MerkleTree([h1, h2])
        assert tree.leaf_count == 2
        assert tree.root == _hash_pair(h1, h2)

    def test_power_of_two_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(8)]
        tree = MerkleTree(leaves)
        assert tree.leaf_count == 8
        assert tree.depth == 4  # 3 inner levels + leaf level

    def test_odd_number_of_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(5)]
        tree = MerkleTree(leaves)
        assert tree.leaf_count == 5
        assert tree.root != GENESIS_HASH

    def test_proof_for_first_leaf(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        assert len(proof) > 0
        assert MerkleTree.verify_proof(leaves[0], proof, tree.root) is True

    def test_proof_for_last_leaf(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(3)
        assert MerkleTree.verify_proof(leaves[3], proof, tree.root) is True

    def test_proof_for_all_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(8)]
        tree = MerkleTree(leaves)
        for i in range(8):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(leaves[i], proof, tree.root) is True

    def test_proof_invalid_leaf_rejected(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        fake_leaf = "f" * 64
        assert MerkleTree.verify_proof(fake_leaf, proof, tree.root) is False

    def test_proof_wrong_root_rejected(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        assert MerkleTree.verify_proof(leaves[0], proof, "0" * 64) is False

    def test_proof_index_out_of_range(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        with pytest.raises(IndexError):
            tree.get_proof(4)
        with pytest.raises(IndexError):
            tree.get_proof(-1)

    def test_depth_property(self) -> None:
        tree = MerkleTree(["a" * 64])
        assert tree.depth == 1


# ===========================================================================
# AuditLog Tests
# ===========================================================================


class TestAuditLog:
    """Hash-chained audit log tests."""

    def test_record_creates_entry(self) -> None:
        log = create_audit_log()
        entry = log.record(
            event_type=AuditEventType.DATA_RECEIVED,
            actor="system",
            resource="test.csv",
            action_detail="File ingested",
        )
        assert entry.sequence_number == 0
        assert entry.event_type == "data_received"
        assert entry.previous_hash == GENESIS_HASH
        assert len(entry.entry_hash) == 64

    def test_chain_links(self) -> None:
        log = create_audit_log()
        e1 = log.record(AuditEventType.DATA_RECEIVED, "sys", "a.csv", "Received")
        e2 = log.record(AuditEventType.DATA_VERIFIED, "sys", "a.csv", "Verified")
        assert e2.previous_hash == e1.entry_hash

    def test_verify_chain_valid(self) -> None:
        log = create_audit_log()
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"file_{i}", "Test")
        status, tampered = log.verify_chain()
        assert status == IntegrityStatus.VALID
        assert tampered == []

    def test_verify_chain_empty(self) -> None:
        log = create_audit_log()
        status, tampered = log.verify_chain()
        assert status == IntegrityStatus.VALID

    def test_verify_chain_detects_tamper(self) -> None:
        log = create_audit_log()
        for i in range(5):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"file_{i}", "Test")

        # Tamper with entry 2
        original = log.entries[2]
        tampered_entry = AuditEntry(
            entry_id=original.entry_id,
            sequence_number=original.sequence_number,
            timestamp=original.timestamp,
            event_type=original.event_type,
            actor="ATTACKER",
            resource=original.resource,
            action_detail=original.action_detail,
            metadata=original.metadata,
            previous_hash=original.previous_hash,
            entry_hash=original.entry_hash,  # hash no longer matches
        )
        log.entries[2] = tampered_entry

        status, tampered_list = log.verify_chain()
        assert status == IntegrityStatus.TAMPERED
        assert 2 in tampered_list

    def test_verify_entry(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "a.csv", "Test")
        assert log.verify_entry(0) is True

    def test_verify_entry_invalid_index(self) -> None:
        log = create_audit_log()
        assert log.verify_entry(0) is False
        assert log.verify_entry(-1) is False

    def test_checkpoint_creation(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        assert len(log.checkpoints) >= 1

    def test_verify_checkpoint(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        assert log.verify_checkpoint(0) is True

    def test_verify_checkpoint_invalid_index(self) -> None:
        log = create_audit_log()
        assert log.verify_checkpoint(0) is False
        assert log.verify_checkpoint(-1) is False

    def test_get_merkle_proof(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        result = log.get_merkle_proof(2)
        assert result is not None
        checkpoint, proof = result
        leaf_hash = log.entries[2].entry_hash
        assert MerkleTree.verify_proof(leaf_hash, proof, checkpoint.merkle_root)

    def test_get_merkle_proof_no_checkpoint(self) -> None:
        log = create_audit_log(checkpoint_interval=1000)
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        assert log.get_merkle_proof(0) is None

    def test_filter_by_type(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "a", "Received")
        log.record(AuditEventType.DATA_VERIFIED, "sys", "a", "Verified")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "b", "Received")
        filtered = log.filter_by_type(AuditEventType.DATA_RECEIVED)
        assert len(filtered) == 2

    def test_filter_by_resource(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "file_a.csv", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "file_b.csv", "Test")
        filtered = log.filter_by_resource("file_a.csv")
        assert len(filtered) == 1

    def test_filter_by_actor(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "bob", "f", "Test")
        assert len(log.filter_by_actor("alice")) == 1

    def test_filter_by_time_range(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(minutes=1)
        end = now + timedelta(minutes=1)
        filtered = log.filter_by_time_range(start, end)
        assert len(filtered) == 1

    def test_record_with_metadata(self) -> None:
        log = create_audit_log()
        # Substitution: ENCRYPTION_APPLIED -> DATA_VERIFIED
        # (test verifies metadata persistence; event type is arbitrary fixture)
        entry = log.record(
            AuditEventType.DATA_VERIFIED,
            actor="system",
            resource="data.bin",
            action_detail="AES-256-GCM applied",
            metadata={"algorithm": "AES-256-GCM", "key_id": "K-001"},
        )
        assert entry.metadata["algorithm"] == "AES-256-GCM"


# ===========================================================================
# JSON Export / Import Tests
# ===========================================================================


class TestAuditLogPersistence:
    """JSON serialisation and integrity-verified import tests."""

    def test_export_import_roundtrip(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            log.export_json(path)
            assert path.exists()

            log2 = AuditLog()
            log2.import_json(path)
            assert len(log2.entries) == 10
            assert log2.log_id == log.log_id

    def test_import_verifies_integrity(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            log.export_json(path)

            # Tamper with JSON
            with path.open("r") as f:
                data = json.load(f)
            data["entries"][0]["actor"] = "TAMPERED"
            with path.open("w") as f:
                json.dump(data, f)

            log2 = AuditLog()
            with pytest.raises(ValueError, match="integrity verification"):
                log2.import_json(path)

    def test_export_creates_directories(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "audit.json"
            log.export_json(path)
            assert path.exists()


# ===========================================================================
# Summary Generation Tests
# ===========================================================================


class TestAuditLogSummary:
    """Audit log summary statistics tests."""

    def test_summary_structure(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test")
        log.record(AuditEventType.DATA_VERIFIED, "bob", "f2", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test")

        summary = log.generate_summary()
        assert summary["total_entries"] == 3
        assert summary["integrity_status"] == "valid"
        assert summary["event_type_distribution"]["data_received"] == 2
        assert summary["actor_distribution"]["alice"] == 2
        assert summary["resource_distribution"]["f1"] == 2

    def test_summary_empty_log(self) -> None:
        log = create_audit_log()
        summary = log.generate_summary()
        assert summary["total_entries"] == 0
        assert summary["first_entry_time"] is None
        assert summary["last_entry_time"] is None

    def test_summary_timestamps(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.SESSION_START, "sys", "session", "Start")
        log.record(AuditEventType.SESSION_END, "sys", "session", "End")
        summary = log.generate_summary()
        assert summary["first_entry_time"] is not None
        assert summary["last_entry_time"] is not None


# ===========================================================================
# DataProvenance Tests
# ===========================================================================


class TestDataProvenance:
    """Data provenance custody chain and transformation tracking."""

    def test_factory_creates_provenance(self) -> None:
        prov = create_provenance_tracker(
            source="CDC SFTP",
            initial_custodian="system",
            initial_hash="abc123",
        )
        assert prov.source == "CDC SFTP"
        assert prov.current_custodian == "system"
        assert prov.integrity_hash == "abc123"

    def test_custody_transfer(self) -> None:
        prov = DataProvenance(source="CDC SFTP", current_custodian="system")
        prov.record_custody_transfer(
            from_custodian="system",
            to_custodian="researcher",
            reason="Data released for analysis",
            integrity_hash="hash_at_transfer",
        )
        assert prov.current_custodian == "researcher"
        assert len(prov.custody_chain) == 1
        assert prov.integrity_hash == "hash_at_transfer"

    def test_multiple_transfers(self) -> None:
        prov = DataProvenance(source="CDC", current_custodian="sys")
        prov.record_custody_transfer("sys", "alice", "Analysis", "h1")
        prov.record_custody_transfer("alice", "bob", "Review", "h2")
        prov.record_custody_transfer("bob", "archive", "Archival", "h3")
        assert len(prov.custody_chain) == 3
        assert prov.current_custodian == "archive"
        assert prov.integrity_hash == "h3"

    def test_custody_transfer_preserves_hash_when_none(self) -> None:
        prov = DataProvenance(
            source="CDC", current_custodian="sys", integrity_hash="original"
        )
        prov.record_custody_transfer("sys", "alice", "Transfer")
        # integrity_hash should remain "original" since none provided
        assert prov.custody_chain[-1]["integrity_hash"] == "original"

    def test_record_transformation(self) -> None:
        prov = DataProvenance(source="CDC", integrity_hash="input_hash")
        prov.record_transformation(
            transformation_type="de-identification",
            description="Safe Harbor + k-anonymity applied",
            input_hash="input_hash",
            output_hash="output_hash",
            parameters={"k": 5, "method": "safe_harbor"},
        )
        assert len(prov.transformations) == 1
        assert prov.integrity_hash == "output_hash"
        assert prov.transformations[0]["type"] == "de-identification"

    def test_multiple_transformations(self) -> None:
        prov = DataProvenance(source="CDC", integrity_hash="h0")
        prov.record_transformation("deidentify", "Safe Harbor", "h0", "h1")
        prov.record_transformation("anonymize", "k-Anonymity", "h1", "h2")
        prov.record_transformation("perturb", "Differential Privacy", "h2", "h3")
        assert len(prov.transformations) == 3
        assert prov.integrity_hash == "h3"

    def test_to_dict(self) -> None:
        prov = create_provenance_tracker("CDC SFTP", "system", "h0")
        prov.record_custody_transfer("system", "researcher", "Release", "h1")
        prov.record_transformation("deidentify", "Applied", "h0", "h1")

        d = prov.to_dict()
        assert d["source"] == "CDC SFTP"
        assert d["custody_events"] == 1
        assert d["transformation_events"] == 1
        assert "custody_chain" in d
        assert "transformations" in d

    def test_asset_id_auto_generated(self) -> None:
        prov = DataProvenance()
        assert prov.asset_id.startswith("ASSET-")

    def test_created_at_auto_set(self) -> None:
        prov = DataProvenance()
        assert prov.created_at != ""


# ===========================================================================
# IntegrityStatus Tests
# ===========================================================================


class TestIntegrityStatus:
    """Integrity status enum tests."""

    def test_values(self) -> None:
        assert IntegrityStatus.VALID.value == "valid"
        assert IntegrityStatus.TAMPERED.value == "tampered"
        assert IntegrityStatus.INCOMPLETE.value == "incomplete"
        assert IntegrityStatus.UNKNOWN.value == "unknown"
        assert len(IntegrityStatus) == 4


# ===========================================================================
# MerkleCheckpoint Tests
# ===========================================================================


class TestMerkleCheckpoint:
    """MerkleCheckpoint NamedTuple tests."""

    def test_checkpoint_fields(self) -> None:
        cp = MerkleCheckpoint(
            checkpoint_id="CP-001",
            sequence_range=(0, 10),
            merkle_root="a" * 64,
            leaf_count=10,
            tree_depth=4,
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert cp.checkpoint_id == "CP-001"
        assert cp.sequence_range == (0, 10)
        assert cp.leaf_count == 10

    def test_checkpoint_immutable(self) -> None:
        cp = MerkleCheckpoint("id", (0, 5), "root", 5, 3, "ts")
        with pytest.raises(AttributeError):
            cp.merkle_root = "modified"  # type: ignore[misc]


# ===========================================================================
# Coverage Gap Tests - previously uncovered lines
# ===========================================================================


class TestCoverageGaps:
    """Tests targeting previously uncovered code paths."""

    def test_merkle_proof_single_leaf_self_sibling(self) -> None:
        """Line 302: sibling_idx == idx when odd leaf count, last leaf."""
        leaves = [f"{i:064x}" for i in range(3)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(2)
        assert len(proof) > 0
        assert MerkleTree.verify_proof(leaves[2], proof, tree.root)

    def test_verify_chain_previous_hash_mismatch(self) -> None:
        """Lines 474-475: detect previous_hash link tampering."""
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "a", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "b", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "c", "Test")

        # Tamper: change previous_hash of entry 1 without changing hash
        original = log.entries[1]
        tampered = AuditEntry(
            entry_id=original.entry_id,
            sequence_number=original.sequence_number,
            timestamp=original.timestamp,
            event_type=original.event_type,
            actor=original.actor,
            resource=original.resource,
            action_detail=original.action_detail,
            metadata=original.metadata,
            previous_hash="0" * 64,  # broken link
            entry_hash=original.entry_hash,
        )
        log.entries[1] = tampered

        status, tampered_list = log.verify_chain()
        assert status == IntegrityStatus.TAMPERED
        assert 1 in tampered_list

    def test_verify_checkpoint_end_exceeds_entries(self) -> None:
        """Line 545: checkpoint.sequence_range end > len(entries)."""
        log = create_audit_log(checkpoint_interval=5)
        for i in range(6):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")

        assert len(log.checkpoints) >= 1
        # Mutate the checkpoint range to extend beyond entries
        cp = log.checkpoints[0]
        fake_cp = MerkleCheckpoint(
            checkpoint_id=cp.checkpoint_id,
            sequence_range=(cp.sequence_range[0], len(log.entries) + 100),
            merkle_root=cp.merkle_root,
            leaf_count=cp.leaf_count,
            tree_depth=cp.tree_depth,
            created_at=cp.created_at,
        )
        log.checkpoints[0] = fake_cp
        assert log.verify_checkpoint(0) is False

    def test_create_checkpoint_start_ge_end(self) -> None:
        """Line 586: _create_checkpoint returns early if start >= end."""
        log = create_audit_log(checkpoint_interval=1000)
        # Force start == end by manually adjusting internal state
        log._last_checkpoint_seq = 0
        log._create_checkpoint()
        assert len(log.checkpoints) == 0


# ===========================================================================
# Anomaly Detector Tests
# ===========================================================================


class TestAuditAnomalyDetector:
    """AuditAnomalyDetector sliding-window and burst tests."""

    def test_no_anomaly_below_threshold(self) -> None:
        from ml.data.audit_trail import AuditAnomalyDetector, RateLimitConfig
        config = RateLimitConfig(
            max_events_per_window=100,
            burst_threshold=20,
            off_hours_start=0,
            off_hours_end=0,
        )
        detector = AuditAnomalyDetector(config)

        log = create_audit_log()
        entry = log.record(AuditEventType.DATA_RECEIVED, "operator", "f", "Test")
        anomalies = detector.observe(entry)
        assert anomalies == []

    def test_velocity_breach_detected(self) -> None:
        from ml.data.audit_trail import (
            AnomalyType, AuditAnomalyDetector, RateLimitConfig,
        )
        config = RateLimitConfig(
            max_events_per_window=3, window_seconds=3600, burst_threshold=100,
        )
        detector = AuditAnomalyDetector(config)

        log = create_audit_log()
        for i in range(5):
            entry = log.record(AuditEventType.DATA_RECEIVED, "fast_user", f"f_{i}", "Test")
            detector.observe(entry)

        all_anomalies = detector.anomalies
        velocity_hits = [
            a for a in all_anomalies
            if a.anomaly_type == AnomalyType.VELOCITY_BREACH
        ]
        assert len(velocity_hits) > 0

    def test_burst_detected(self) -> None:
        from ml.data.audit_trail import (
            AnomalyType, AuditAnomalyDetector, RateLimitConfig,
        )
        config = RateLimitConfig(
            max_events_per_window=1000, burst_threshold=2, window_seconds=600,
        )
        detector = AuditAnomalyDetector(config)

        log = create_audit_log()
        for i in range(5):
            entry = log.record(AuditEventType.DATA_RECEIVED, "burster", f"f_{i}", "Test")
            detector.observe(entry)

        burst_hits = [
            a for a in detector.anomalies
            if a.anomaly_type == AnomalyType.BURST_DETECTED
        ]
        assert len(burst_hits) > 0

    def test_off_hours_access_detected(self) -> None:
        from ml.data.audit_trail import (
            AnomalyType, AuditAnomalyDetector, RateLimitConfig, AuditEntry,
        )
        config = RateLimitConfig(off_hours_start=22, off_hours_end=6)
        detector = AuditAnomalyDetector(config)

        off_hours_entry = AuditEntry(
            entry_id="E-OFF",
            sequence_number=0,
            timestamp="2025-01-15T02:30:00+00:00",  # 2:30 AM UTC
            event_type="data_received",
            actor="night_owl",
            resource="test.csv",
            action_detail="Late access",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
        )

        anomalies = detector.observe(off_hours_entry)
        off_hours_hits = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.OFF_HOURS_ACCESS
        ]
        assert len(off_hours_hits) == 1

    def test_get_actor_velocity(self) -> None:
        from ml.data.audit_trail import AuditAnomalyDetector
        detector = AuditAnomalyDetector()

        log = create_audit_log()
        for i in range(3):
            entry = log.record(AuditEventType.DATA_RECEIVED, "tracked_user", f"f_{i}", "Test")
            detector.observe(entry)

        assert detector.get_actor_velocity("tracked_user") == 3
        assert detector.get_actor_velocity("unknown_user") == 0

    def test_reset_clears_state(self) -> None:
        from ml.data.audit_trail import AuditAnomalyDetector
        detector = AuditAnomalyDetector()

        log = create_audit_log()
        entry = log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        detector.observe(entry)
        assert detector.get_actor_velocity("sys") == 1

        detector.reset()
        assert detector.get_actor_velocity("sys") == 0
        assert detector.anomalies == []

    def test_default_config(self) -> None:
        from ml.data.audit_trail import AuditAnomalyDetector
        detector = AuditAnomalyDetector()
        # Should not raise
        log = create_audit_log()
        entry = log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        detector.observe(entry)

    def test_off_hours_contiguous_range(self) -> None:
        """Off-hours when start <= end (e.g. 8 to 17 counts as off-hours)."""
        from ml.data.audit_trail import (
            AnomalyType, AuditAnomalyDetector, RateLimitConfig, AuditEntry,
        )
        config = RateLimitConfig(off_hours_start=8, off_hours_end=17)
        detector = AuditAnomalyDetector(config)

        entry = AuditEntry(
            entry_id="E-DAY",
            sequence_number=0,
            timestamp="2025-01-15T12:00:00+00:00",
            event_type="data_received",
            actor="day_worker",
            resource="test.csv",
            action_detail="Daytime access",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="b" * 64,
        )
        anomalies = detector.observe(entry)
        off_hours_hits = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.OFF_HOURS_ACCESS
        ]
        assert len(off_hours_hits) == 1


# ===========================================================================
# Merkle Consistency Proof Tests
# ===========================================================================


class TestMerkleConsistencyProof:
    """Tests for append-only Merkle tree consistency verification."""

    def test_consistent_extension(self) -> None:
        from ml.data.audit_trail import merkle_consistency_proof
        old_leaves = [f"{i:064x}" for i in range(4)]
        new_leaves = old_leaves + [f"{i:064x}" for i in range(4, 8)]
        old_tree = MerkleTree(old_leaves)
        new_tree = MerkleTree(new_leaves)
        assert merkle_consistency_proof(old_tree, new_tree) is True

    def test_empty_old_tree_consistent(self) -> None:
        from ml.data.audit_trail import merkle_consistency_proof
        old_tree = MerkleTree([])
        new_tree = MerkleTree(["a" * 64])
        assert merkle_consistency_proof(old_tree, new_tree) is True

    def test_new_tree_smaller_than_old(self) -> None:
        from ml.data.audit_trail import merkle_consistency_proof
        old_tree = MerkleTree([f"{i:064x}" for i in range(5)])
        new_tree = MerkleTree([f"{i:064x}" for i in range(3)])
        assert merkle_consistency_proof(old_tree, new_tree) is False

    def test_modified_leaf_inconsistent(self) -> None:
        from ml.data.audit_trail import merkle_consistency_proof
        old_leaves = [f"{i:064x}" for i in range(4)]
        modified = [f"{i:064x}" for i in range(4)] + ["f" * 64]
        modified[2] = "e" * 64  # tamper
        old_tree = MerkleTree(old_leaves)
        new_tree = MerkleTree(modified)
        assert merkle_consistency_proof(old_tree, new_tree) is False

    def test_same_tree_consistent(self) -> None:
        from ml.data.audit_trail import merkle_consistency_proof
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        assert merkle_consistency_proof(tree, tree) is True


# ===========================================================================
# Audit Statistics Tests
# ===========================================================================


class TestAuditStatistics:
    """Tests for compute_audit_statistics."""

    def test_empty_log_statistics(self) -> None:
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        stats = compute_audit_statistics(log)
        assert stats.total_entries == 0
        assert stats.integrity_verified is True

    def test_populated_log_statistics(self) -> None:
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test")
        log.record(AuditEventType.DATA_VERIFIED, "bob", "f2", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test2")

        stats = compute_audit_statistics(log)
        assert stats.total_entries == 3
        assert stats.unique_actors == 2
        assert stats.unique_resources == 2
        assert stats.event_type_counts["data_received"] == 2
        assert stats.integrity_verified is True
        assert stats.mean_inter_event_seconds >= 0.0

    def test_hourly_distribution(self) -> None:
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        stats = compute_audit_statistics(log)
        assert sum(stats.hourly_distribution.values()) == 1

    def test_tampered_log_integrity_false(self) -> None:
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        log.record(AuditEventType.DATA_VERIFIED, "sys", "f", "Test")

        original = log.entries[0]
        tampered = AuditEntry(
            entry_id=original.entry_id,
            sequence_number=original.sequence_number,
            timestamp=original.timestamp,
            event_type=original.event_type,
            actor="TAMPERED",
            resource=original.resource,
            action_detail=original.action_detail,
            metadata=original.metadata,
            previous_hash=original.previous_hash,
            entry_hash=original.entry_hash,
        )
        log.entries[0] = tampered

        stats = compute_audit_statistics(log)
        assert stats.integrity_verified is False


# ===========================================================================
# Final Coverage Completeness - targeting remaining uncovered lines
# ===========================================================================


class TestFinalCoverage:
    """Tests targeting the last uncovered lines."""

    def test_anomaly_detector_tz_naive_timestamp(self) -> None:
        """Line 1055: observe() with tz-naive timestamp entry."""
        from ml.data.audit_trail import AuditAnomalyDetector
        detector = AuditAnomalyDetector()
        # Entry with tz-naive timestamp (no +00:00)
        entry = AuditEntry(
            entry_id="E-NAIVE",
            sequence_number=0,
            timestamp="2025-01-15T12:00:00",  # no timezone
            event_type="data_received",
            actor="naive_user",
            resource="test.csv",
            action_detail="Test",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="c" * 64,
        )
        anomalies = detector.observe(entry)
        # Should not crash, should process normally
        assert isinstance(anomalies, list)

    def test_merkle_consistency_proof_verify_fails(self) -> None:
        """Line 1177: verify_proof fails when root is corrupted but leaves match."""
        from ml.data.audit_trail import merkle_consistency_proof
        old_leaves = [f"{i:064x}" for i in range(3)]
        new_leaves = [f"{i:064x}" for i in range(3)] + ["f" * 64]
        old_tree = MerkleTree(old_leaves)
        new_tree = MerkleTree(new_leaves)
        # Corrupt the new tree's ROOT (not leaves), so leaves still match
        # but verify_proof against new root fails
        new_tree._root = "0" * 64
        result = merkle_consistency_proof(old_tree, new_tree)
        assert result is False

    def test_compute_statistics_tz_naive_entry(self) -> None:
        """Lines 1248, 1251-1252: tz-naive timestamp + ValueError path."""
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        # Record a real entry (tz-aware)
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f1", "Test")

        # Manually add an entry with tz-naive timestamp
        naive_entry = AuditEntry(
            entry_id="E-NAIVE",
            sequence_number=1,
            timestamp="2025-06-15T10:30:00",  # no timezone
            event_type="data_verified",
            actor="sys",
            resource="f2",
            action_detail="Test naive",
            metadata={},
            previous_hash=log.entries[0].entry_hash,
            entry_hash="d" * 64,
        )
        log.entries.append(naive_entry)

        stats = compute_audit_statistics(log)
        assert stats.total_entries == 2

    def test_compute_statistics_invalid_timestamp(self) -> None:
        """Line 1252: ValueError branch for unparseable timestamp."""
        from ml.data.audit_trail import compute_audit_statistics
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        # Manually add entry with invalid timestamp
        bad_entry = AuditEntry(
            entry_id="E-BAD",
            sequence_number=1,
            timestamp="not-a-timestamp-at-all",
            event_type="data_verified",
            actor="sys",
            resource="f2",
            action_detail="Bad ts",
            metadata={},
            previous_hash=log.entries[0].entry_hash,
            entry_hash="e" * 64,
        )
        log.entries.append(bad_entry)

        stats = compute_audit_statistics(log)
        assert stats.total_entries == 2
        # Only 1 valid timestamp parsed
        assert sum(stats.hourly_distribution.values()) == 1


# ===========================================================================
# Audit Search Engine Tests
# ===========================================================================


class TestAuditSearchEngine:
    """Structured log search engine tests."""

    @pytest.fixture()
    def populated_log(self) -> AuditLog:
        log = create_audit_log()
        log.record(
            AuditEventType.DATA_RECEIVED, "alice", "file1.csv",
            "Received specimen data", metadata={"batch": "B001"},
        )
        log.record(
            AuditEventType.DATA_VERIFIED, "bob", "file1.csv",
            "Verified checksums", metadata={"batch": "B001"},
        )
        log.record(
            # Substitution: ACCESS_GRANTED -> DATA_RELEASED
            AuditEventType.DATA_RELEASED, "alice", "file2.csv",
            "Granted access to dataset", metadata={"role": "pi"},
        )
        log.record(
            # Substitution: DATA_DELETED -> SESSION_END
            AuditEventType.SESSION_END, "system", "file3.csv",
            "Purged expired data",
        )
        return log

    def test_search_all(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        results = engine.search(AuditSearchQuery())
        assert len(results) == 4

    def test_search_by_event_types(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(
            event_types=frozenset({"data_received", "data_verified"})
        )
        results = engine.search(query)
        assert len(results) == 2
        assert all(
            r.event_type in ("data_received", "data_verified")
            for r in results
        )

    def test_search_by_actors(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(actors=frozenset({"alice"}))
        results = engine.search(query)
        assert len(results) == 2

    def test_search_by_resources(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(resources=frozenset({"file1.csv"}))
        results = engine.search(query)
        assert len(results) == 2

    def test_search_by_metadata(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(metadata_contains={"batch": "B001"})
        results = engine.search(query)
        assert len(results) == 2

    def test_search_by_action_pattern(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(action_pattern="checksum")
        results = engine.search(query)
        assert len(results) == 1
        assert results[0].event_type == "data_verified"

    def test_search_with_limit(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(limit=2)
        results = engine.search(query)
        assert len(results) == 2

    def test_search_by_time_range(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        now = datetime.now(tz=timezone.utc)
        query = AuditSearchQuery(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        results = engine.search(query)
        assert len(results) == 4

    def test_search_time_range_excludes(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        far_past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        query = AuditSearchQuery(
            start_time=far_past,
            end_time=far_past + timedelta(hours=1),
        )
        results = engine.search(query)
        assert len(results) == 0

    def test_search_combined_filters(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(
            actors=frozenset({"alice"}),
            resources=frozenset({"file1.csv"}),
        )
        results = engine.search(query)
        assert len(results) == 1
        assert results[0].event_type == "data_received"

    def test_count(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(actors=frozenset({"alice"}))
        assert engine.count(query) == 2

    def test_metadata_no_match(self, populated_log: AuditLog) -> None:
        engine = AuditSearchEngine(populated_log)
        query = AuditSearchQuery(metadata_contains={"nonexistent": "value"})
        results = engine.search(query)
        assert len(results) == 0


# ===========================================================================
# Audit Log Archiver Tests
# ===========================================================================


class TestAuditLogArchiver:
    """NIST SP 800-92 compliant log archival tests."""

    def test_archive_segment(self) -> None:
        log = create_audit_log()
        for i in range(10):
            log.record(
                AuditEventType.DATA_RECEIVED, "user", f"file{i}.csv",
                f"Received file {i}",
            )

        archiver = AuditLogArchiver(retention_years=6)
        segment = archiver.archive_segment(log, 0, 10)

        assert segment.entry_count == 10
        assert segment.sequence_range == (0, 10)
        assert len(segment.merkle_root) == 64
        assert segment.retention_years == 6
        assert len(archiver.archives) == 1

    def test_verify_archive_valid(self) -> None:
        log = create_audit_log()
        for i in range(5):
            log.record(
                AuditEventType.DATA_VERIFIED, "sys", f"f{i}", f"Check {i}",
            )

        archiver = AuditLogArchiver()
        segment = archiver.archive_segment(log, 0, 5)
        assert archiver.verify_archive(log, segment) is True

    def test_verify_archive_tampered(self) -> None:
        log = create_audit_log()
        for i in range(5):
            log.record(
                AuditEventType.DATA_RECEIVED, "sys", f"f{i}", f"R {i}",
            )

        archiver = AuditLogArchiver()
        segment = archiver.archive_segment(log, 0, 5)

        # Tamper with an entry hash
        original = log.entries[2]
        tampered = AuditEntry(
            entry_id=original.entry_id,
            sequence_number=original.sequence_number,
            timestamp=original.timestamp,
            event_type=original.event_type,
            actor=original.actor,
            resource=original.resource,
            action_detail="TAMPERED",
            metadata=original.metadata,
            previous_hash=original.previous_hash,
            entry_hash="f" * 64,
        )
        log.entries[2] = tampered

        assert archiver.verify_archive(log, segment) is False

    def test_archive_invalid_range(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "u", "f", "test")
        archiver = AuditLogArchiver()

        with pytest.raises(ValueError, match="Invalid range"):
            archiver.archive_segment(log, 5, 10)

        with pytest.raises(ValueError, match="Invalid range"):
            archiver.archive_segment(log, 0, 0)

    def test_verify_archive_out_of_range(self) -> None:
        from ml.data.audit_trail import ArchiveSegment

        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "u", "f", "test")
        archiver = AuditLogArchiver()

        fake_segment = ArchiveSegment(
            segment_id="ARC-FAKE",
            sequence_range=(0, 999),
            entry_count=999,
            merkle_root="a" * 64,
            archived_at=datetime.now(tz=timezone.utc).isoformat(),
            retention_years=6,
        )
        assert archiver.verify_archive(log, fake_segment) is False

    def test_retention_review_empty(self) -> None:
        archiver = AuditLogArchiver()
        assert archiver.segments_needing_retention_review() == []

    def test_retention_review_near_expiry(self) -> None:
        from ml.data.audit_trail import ArchiveSegment

        archiver = AuditLogArchiver()
        # Create a segment archived almost 6 years ago
        old_date = datetime.now(tz=timezone.utc) - timedelta(days=365 * 6 - 100)
        old_segment = ArchiveSegment(
            segment_id="ARC-OLD",
            sequence_range=(0, 10),
            entry_count=10,
            merkle_root="b" * 64,
            archived_at=old_date.isoformat(),
            retention_years=6,
        )
        archiver._archives.append(old_segment)

        review = archiver.segments_needing_retention_review()
        assert len(review) == 1
        assert review[0].segment_id == "ARC-OLD"

    def test_retention_review_not_near_expiry(self) -> None:
        from ml.data.audit_trail import ArchiveSegment

        archiver = AuditLogArchiver()
        recent = ArchiveSegment(
            segment_id="ARC-NEW",
            sequence_range=(0, 5),
            entry_count=5,
            merkle_root="c" * 64,
            archived_at=datetime.now(tz=timezone.utc).isoformat(),
            retention_years=6,
        )
        archiver._archives.append(recent)
        assert archiver.segments_needing_retention_review() == []

    def test_retention_review_invalid_date(self) -> None:
        from ml.data.audit_trail import ArchiveSegment

        archiver = AuditLogArchiver()
        bad = ArchiveSegment(
            segment_id="ARC-BAD",
            sequence_range=(0, 1),
            entry_count=1,
            merkle_root="d" * 64,
            archived_at="not-a-date",
            retention_years=6,
        )
        archiver._archives.append(bad)
        review = archiver.segments_needing_retention_review()
        assert len(review) == 1

    def test_retention_review_tz_naive_archived_at(self) -> None:
        """Cover tz-naive archived_at -> replace(tzinfo=UTC) branch."""
        from ml.data.audit_trail import ArchiveSegment

        archiver = AuditLogArchiver()
        # Use a tz-naive ISO string (no +00:00 suffix) that expires soon
        naive_dt = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(
            days=365 * 6 - 100
        )
        seg = ArchiveSegment(
            segment_id="ARC-NAIVE",
            sequence_range=(0, 1),
            entry_count=1,
            merkle_root="e" * 64,
            archived_at=naive_dt.isoformat(),
            retention_years=6,
        )
        archiver._archives.append(seg)
        review = archiver.segments_needing_retention_review()
        assert len(review) == 1

    def test_search_excludes_entries_before_start(self) -> None:
        """Cover start_iso filter branch that skips early entries."""
        from ml.data.audit_trail import AuditEventType, AuditEntry, GENESIS_HASH

        log = AuditLog(log_id="SEARCH-PRE")
        # Manually inject entries with controlled timestamps
        early_entry = AuditEntry(
            entry_id="SEARCH-PRE-000000",
            sequence_number=0,
            timestamp="2020-01-01T00:00:00+00:00",
            # Substitution: ACCESS_GRANTED -> DATA_RECEIVED
            event_type=AuditEventType.DATA_RECEIVED.value,
            actor="analyst@lab.org",
            resource="/data/old.csv",
            action_detail="early read",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
        )
        late_entry = AuditEntry(
            entry_id="SEARCH-PRE-000001",
            sequence_number=1,
            timestamp="2024-06-01T00:00:00+00:00",
            # Substitution: ACCESS_GRANTED -> DATA_RECEIVED
            event_type=AuditEventType.DATA_RECEIVED.value,
            actor="analyst@lab.org",
            resource="/data/new.csv",
            action_detail="late read",
            metadata={},
            previous_hash="a" * 64,
            entry_hash="b" * 64,
        )
        log.entries.extend([early_entry, late_entry])

        engine = AuditSearchEngine(log)
        query = AuditSearchQuery(
            start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        results = engine.search(query)
        assert len(results) == 1
        assert results[0].action_detail == "late read"


# ===========================================================================
# AuditExportFormat Enum Tests
# ===========================================================================


class TestAuditExportFormat:
    """Verify export format enum values and membership."""

    def test_json_value(self) -> None:
        assert AuditExportFormat.JSON.value == "json"

    def test_csv_value(self) -> None:
        assert AuditExportFormat.CSV.value == "csv"

    def test_ndjson_value(self) -> None:
        assert AuditExportFormat.NDJSON.value == "ndjson"

    def test_member_count(self) -> None:
        assert len(AuditExportFormat) == 3


# ===========================================================================
# AuditComplianceReport Dataclass Tests
# ===========================================================================


class TestAuditComplianceReport:
    """Verify frozen dataclass properties and field access."""

    def test_construction_and_fields(self) -> None:
        report = AuditComplianceReport(
            log_id="LOG-COMPL-001",
            generated_at="2025-03-01T12:00:00+00:00",
            total_entries=42,
            first_entry_timestamp="2024-01-01T00:00:00+00:00",
            last_entry_timestamp="2025-02-28T23:59:59+00:00",
            unique_actors=5,
            unique_resources=12,
            events_by_type={"data_received": 20, "access_granted": 22},
            integrity_verified=True,
            chain_intact=True,
            checkpoints_count=3,
            anomalies_detected=0,
            coverage_start="2024-01-01T00:00:00+00:00",
            coverage_end="2025-02-28T23:59:59+00:00",
            archive_segments=2,
            retention_reviews_pending=0,
        )
        assert report.log_id == "LOG-COMPL-001"
        assert report.total_entries == 42
        assert report.unique_actors == 5
        assert report.unique_resources == 12
        assert report.integrity_verified is True
        assert report.chain_intact is True
        assert report.checkpoints_count == 3
        assert report.anomalies_detected == 0
        assert report.archive_segments == 2
        assert report.retention_reviews_pending == 0

    def test_frozen_immutability(self) -> None:
        report = AuditComplianceReport(
            log_id="LOG-FRZ",
            generated_at="2025-01-01T00:00:00+00:00",
            total_entries=0,
            first_entry_timestamp=None,
            last_entry_timestamp=None,
            unique_actors=0,
            unique_resources=0,
            events_by_type={},
            integrity_verified=True,
            chain_intact=True,
            checkpoints_count=0,
            anomalies_detected=0,
            coverage_start=None,
            coverage_end=None,
            archive_segments=0,
            retention_reviews_pending=0,
        )
        with pytest.raises(AttributeError):
            report.log_id = "ALTERED"  # type: ignore[misc]

    def test_none_timestamps_when_empty(self) -> None:
        report = AuditComplianceReport(
            log_id="LOG-EMPTY",
            generated_at="2025-06-01T00:00:00+00:00",
            total_entries=0,
            first_entry_timestamp=None,
            last_entry_timestamp=None,
            unique_actors=0,
            unique_resources=0,
            events_by_type={},
            integrity_verified=True,
            chain_intact=True,
            checkpoints_count=0,
            anomalies_detected=0,
            coverage_start=None,
            coverage_end=None,
            archive_segments=0,
            retention_reviews_pending=0,
        )
        assert report.first_entry_timestamp is None
        assert report.last_entry_timestamp is None
        assert report.coverage_start is None
        assert report.coverage_end is None


# ===========================================================================
# AuditExporter Tests
# ===========================================================================


def _build_populated_log() -> AuditLog:
    """Helper - construct a log with 3 entries across different types."""
    log = create_audit_log()
    log.record(
        event_type=AuditEventType.DATA_RECEIVED,
        actor="ingestion@pipeline.org",
        resource="/data/batch_001.csv",
        action_detail="Received 500 records from WHO portal",
    )
    log.record(
        # Substitution: ACCESS_GRANTED -> DATA_VERIFIED
        event_type=AuditEventType.DATA_VERIFIED,
        actor="researcher@university.edu",
        resource="/data/batch_001.csv",
        action_detail="Read access for IRB PRJ-2024-0019",
    )
    log.record(
        event_type=AuditEventType.COMPLIANCE_CHECK,
        actor="model-server@prod",
        resource="/models/amoeba-v3",
        action_detail="Inference batch 128 samples",
    )
    return log


class TestAuditExporter:
    """Comprehensive tests for AuditExporter - JSON, CSV, NDJSON."""

    def test_export_json_default(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        payload = exporter.export(AuditExportFormat.JSON)
        parsed = json.loads(payload)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["event_type"] == AuditEventType.DATA_RECEIVED.value
        assert parsed[1]["actor"] == "researcher@university.edu"
        assert parsed[2]["resource"] == "/models/amoeba-v3"

    def test_export_json_with_field_filter(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        payload = exporter.export(
            AuditExportFormat.JSON,
            fields=["entry_id", "actor"],
        )
        parsed = json.loads(payload)
        for row in parsed:
            assert set(row.keys()) == {"entry_id", "actor"}

    def test_export_csv_headers_and_rows(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        csv_text = exporter.export(AuditExportFormat.CSV)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 4  # header + 3 data rows
        headers = lines[0].split(",")
        assert "entry_id" in headers
        assert "actor" in headers
        assert "event_type" in headers

    def test_export_csv_field_filter(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        csv_text = exporter.export(
            AuditExportFormat.CSV,
            fields=["actor", "resource"],
        )
        lines = csv_text.strip().split("\n")
        assert lines[0] == "actor,resource"
        assert len(lines) == 4

    def test_export_csv_escapes_commas(self) -> None:
        log = create_audit_log()
        log.record(
            event_type=AuditEventType.DATA_RECEIVED,
            actor="system",
            resource="/data/file.csv",
            action_detail="Received 100, 200, and 300 rows",
        )
        exporter = AuditExporter(log)
        csv_text = exporter.export(AuditExportFormat.CSV)
        assert '"Received 100, 200, and 300 rows"' in csv_text

    def test_export_csv_escapes_quotes(self) -> None:
        log = create_audit_log()
        log.record(
            event_type=AuditEventType.DATA_RECEIVED,
            actor="system",
            resource="/data/file.csv",
            action_detail='Value with "quotes" inside',
        )
        exporter = AuditExporter(log)
        csv_text = exporter.export(AuditExportFormat.CSV)
        assert '""quotes""' in csv_text

    def test_export_ndjson_lines(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        ndjson_text = exporter.export(AuditExportFormat.NDJSON)
        lines = ndjson_text.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "entry_id" in parsed

    def test_export_ndjson_field_filter(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        ndjson_text = exporter.export(
            AuditExportFormat.NDJSON,
            fields=["actor", "event_type"],
        )
        for line in ndjson_text.strip().split("\n"):
            parsed = json.loads(line)
            assert set(parsed.keys()) == {"actor", "event_type"}

    def test_export_empty_log_json(self) -> None:
        log = create_audit_log()
        exporter = AuditExporter(log)
        payload = exporter.export(AuditExportFormat.JSON)
        assert json.loads(payload) == []

    def test_export_empty_log_csv(self) -> None:
        log = create_audit_log()
        exporter = AuditExporter(log)
        csv_text = exporter.export(AuditExportFormat.CSV)
        assert csv_text == ""

    def test_export_empty_log_ndjson(self) -> None:
        log = create_audit_log()
        exporter = AuditExporter(log)
        ndjson_text = exporter.export(AuditExportFormat.NDJSON)
        assert ndjson_text == ""

    def test_compliance_report_populated_log(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        report = exporter.compliance_report()
        assert isinstance(report, AuditComplianceReport)
        assert report.total_entries == 3
        assert report.unique_actors == 3
        assert report.unique_resources == 2
        assert report.integrity_verified is True
        assert report.chain_intact is True
        assert report.anomalies_detected == 0
        assert report.archive_segments == 0
        assert report.retention_reviews_pending == 0
        assert report.first_entry_timestamp is not None
        assert report.last_entry_timestamp is not None
        assert report.events_by_type[AuditEventType.DATA_RECEIVED.value] == 1
        # Substitution: ACCESS_GRANTED -> DATA_VERIFIED (matches fixture above)
        assert report.events_by_type[AuditEventType.DATA_VERIFIED.value] == 1
        assert report.events_by_type[AuditEventType.COMPLIANCE_CHECK.value] == 1

    def test_compliance_report_empty_log(self) -> None:
        log = create_audit_log()
        exporter = AuditExporter(log)
        report = exporter.compliance_report()
        assert report.total_entries == 0
        assert report.unique_actors == 0
        assert report.first_entry_timestamp is None
        assert report.coverage_start is None
        assert report.integrity_verified is True

    def test_compliance_report_with_anomaly_count(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        report = exporter.compliance_report(anomaly_count=7)
        assert report.anomalies_detected == 7

    def test_compliance_report_with_archiver(self) -> None:
        log = _build_populated_log()
        archiver = AuditLogArchiver(retention_years=6)
        exporter = AuditExporter(log)
        report = exporter.compliance_report(archiver=archiver)
        assert report.archive_segments >= 0
        assert report.retention_reviews_pending >= 0

    def test_compliance_report_log_id_matches(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        report = exporter.compliance_report()
        assert report.log_id == log.log_id

    def test_compliance_report_generated_at_iso_format(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        report = exporter.compliance_report()
        dt = datetime.fromisoformat(report.generated_at)
        assert dt.tzinfo is not None

    def test_compliance_report_checkpoints_count(self) -> None:
        log = _build_populated_log()
        log._create_checkpoint()
        exporter = AuditExporter(log)
        report = exporter.compliance_report()
        assert report.checkpoints_count == 1

    def test_export_json_default_format(self) -> None:
        log = _build_populated_log()
        exporter = AuditExporter(log)
        payload_default = exporter.export()
        payload_json = exporter.export(AuditExportFormat.JSON)
        assert payload_default == payload_json
