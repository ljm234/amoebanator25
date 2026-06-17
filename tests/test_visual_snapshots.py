"""Cumulative visual regression - Phase 4.5 Mini-2 T2.6.

Single canonical parametrized test covering all 4 pages. Each page's
captured AppTest markdown blob is compared against its committed
baseline at ``tests/_snapshots/<name>.md.snap``. Test fails if drift
exceeds 5% character delta - catches nav/disclaimer/banner
regressions that unit tests miss.

Mini-1 T1.7 had a single non-parametrized snapshot test for the
predict page; that test is preserved (closure-gate criterion #7
predates Mini-2). This file is the cumulative version covering all
4 pages.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from streamlit.testing.v1 import AppTest


SNAPSHOT_DIR = Path(__file__).parent / "_snapshots"

_PAGE_SNAPSHOTS: list[tuple[str, str]] = [
    ("pages/01_predict.py", "predict.md.snap"),
    ("pages/02_audit.py",   "audit.md.snap"),
    ("pages/03_about.py",   "about.md.snap"),
    ("pages/04_references.py", "references.md.snap"),
]


@pytest.fixture
def populated_audit_log() -> Generator[Path, None, None]:
    """Audit page needs a populated JSONL to render the table; otherwise
    it hits st.stop() at the empty-log guard and the snapshot is the
    info-pointer state, not the table state. Captured baseline is the
    table-state, so we pre-populate."""
    from ml import audit_hooks as ah
    from ml.audit_hooks import _emit
    from ml.data.audit_trail import AuditEventType

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(path)
    ah._singleton_log = None
    ah._singleton_path = None
    for i in range(3):
        _emit(
            AuditEventType.WEB_PREDICT_RECEIVED,
            actor="pi",
            resource=f"r{i}",
            action_detail=f"evt{i}",
            metadata={"i": i},
        )
    try:
        yield path
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        ah._singleton_log = None
        ah._singleton_path = None
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("page_path, snapshot_name", _PAGE_SNAPSHOTS)
def test_visual_snapshot_drift_under_5pct(
    page_path: str, snapshot_name: str, populated_audit_log: Path
) -> None:
    """Mini-2 cumulative gate #7 - every page's markdown stays within
    5% character delta of its committed baseline."""
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if not snapshot_path.exists():
        pytest.skip(f"baseline {snapshot_path} missing - run T2.6 capture")
    baseline = snapshot_path.read_text(encoding="utf-8")
    if not baseline:
        pytest.skip(f"baseline {snapshot_path} empty")

    at = AppTest.from_file(page_path)
    at.run(timeout=30)
    captured = "\n".join(m.value for m in at.markdown)

    longer = max(len(captured), len(baseline))
    shorter = min(len(captured), len(baseline))
    drift = (longer - shorter) / longer if longer else 0.0
    assert drift < 0.05, (
        f"{page_path} snapshot drift {drift:.1%} exceeds 5% threshold "
        f"(baseline={len(baseline)} chars, captured={len(captured)} chars)"
    )


def test_all_4_snapshots_present() -> None:
    """Source-level guard: all 4 baselines must exist in the snapshot dir."""
    for _, snap in _PAGE_SNAPSHOTS:
        path = SNAPSHOT_DIR / snap
        assert path.exists(), f"missing baseline: {path}"
        assert path.stat().st_size > 0, f"empty baseline: {path}"
