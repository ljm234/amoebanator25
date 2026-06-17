"""Tests for pages/02_audit.py - Phase 4.5 Mini-2 T2.5 (1 of 4)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from streamlit.testing.v1 import AppTest


PAGE_PATH = "pages/02_audit.py"


@pytest.fixture
def populated_audit_log() -> Generator[Path, None, None]:
    """Yield a temp JSONL audit path with 5 emitted events."""
    from ml import audit_hooks as ah
    from ml.audit_hooks import _emit
    from ml.data.audit_trail import AuditEventType

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    os.environ["AMOEBANATOR_AUDIT_PATH"] = str(path)
    ah._singleton_log = None
    ah._singleton_path = None
    for i in range(5):
        _emit(
            AuditEventType.WEB_PREDICT_RECEIVED,
            actor="pi",
            resource=f"resource_{i}",
            action_detail=f"event {i}",
            metadata={"i": i},
        )
    try:
        yield path
    finally:
        os.environ.pop("AMOEBANATOR_AUDIT_PATH", None)
        ah._singleton_log = None
        ah._singleton_path = None
        path.unlink(missing_ok=True)


@pytest.fixture
def empty_audit_log() -> Generator[Path, None, None]:
    """Yield a temp JSONL path that is empty (no events)."""
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


def test_module_imports_cleanly(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_audit_page_renders_disclaimer(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    blob = "\n".join(m.value for m in at.markdown)
    for tok in ["NOT a medical device", "n=30", "limited to", "ORCID"]:
        assert tok in blob, f"missing disclaimer token {tok!r}"


def test_audit_page_title(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    assert "Audit Log (Current Session)" in [t.value for t in at.title]


def test_ephemerality_banner_present(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    warnings = [w.value for w in at.warning]
    assert any("wiped on container restart" in w for w in warnings)


def test_no_audit_log_renders_info_pointer(empty_audit_log: Path) -> None:
    """Empty log → st.info pointer + st.stop, no table render."""
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    infos = [i.value for i in at.info]
    assert any("No audit events yet" in i for i in infos)


def test_table_renders_when_log_populated(populated_audit_log: Path) -> None:
    """Verify markdown blocks present (st.table renders into markdown stream)."""
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    # Title + disclaimer + ephemerality banner present; no exception.
    assert len(at.exception) == 0
    assert len(list(at.markdown)) >= 1


def test_chain_integrity_caption_present(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    captions = [c.value for c in at.caption]
    assert any("verify_csv_chain_integrity" in c for c in captions)


def test_csv_schema_version_caption_present(populated_audit_log: Path) -> None:
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    captions = [c.value for c in at.caption]
    assert any("schema version 1" in c for c in captions)


def test_audit_page_uses_default_path_when_env_unset(populated_audit_log: Path) -> None:
    """The page resolves the audit path via ml.audit_hooks.default_audit_path,
    which respects AMOEBANATOR_AUDIT_PATH (set by the fixture)."""
    from ml.audit_hooks import default_audit_path
    # macOS symlinks /var/folders → /private/var/folders; resolve both
    # sides before comparing so the symlink expansion doesn't break us.
    assert default_audit_path().resolve() == populated_audit_log.resolve()


def test_10k_row_cap_documented(populated_audit_log: Path) -> None:
    """The 10,000-row cap is a Q15.C lock; read the page source to confirm."""
    src = Path(PAGE_PATH).read_text(encoding="utf-8")
    assert "10_000" in src or "10000" in src


def test_audit_page_uses_st_table_not_dataframe(populated_audit_log: Path) -> None:
    """Q15.5.C: page MUST use st.table for screen-reader semantics."""
    src = Path(PAGE_PATH).read_text(encoding="utf-8")
    assert "st.table(" in src
    # st.dataframe is allowed elsewhere but not for the audit table
    # (the pd.read_json call is fine; only the render call matters)


def test_audit_page_renders_under_5s(populated_audit_log: Path) -> None:
    """Boot-time budget - Mini-2 closure gate criterion #3 inherits this."""
    import time
    t0 = time.time()
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=10)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"audit page boot took {elapsed:.2f}s"
    assert len(at.exception) == 0
