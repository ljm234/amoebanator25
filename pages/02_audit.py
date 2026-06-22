"""Audit log viewer.

Renders the current session's audit chain via ``st.table`` (HTML <table> semantics for screen readers, NOT
``st.dataframe``'s virtualised React grid). Capped at 10,000 rows;
full chain stays intact in the underlying JSONL file.

CSV download button emits a fresh export-bytes blob each
click, preserving every row + the hash chain pointers, so reviewers
can verify integrity post-download against a cloned repo.

The page is the load-bearing audit-portability feature: it converts
HF Space's ephemeral filesystem (audit log wipes on container
restart) into an explicit "download before stepping away" UX.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.audit_export import export_audit_to_csv
from app.disclaimer import render_disclaimer
from ml.audit_hooks import default_audit_path


_ROW_CAP = 10_000


st.set_page_config(page_title="Audit - Amoebanator 25")
render_disclaimer()

st.title("Audit Log (Current Session)")

# Ephemerality banner above the table.
st.warning(
    "Showing all events from current session. Earlier sessions wiped "
    "on container restart (HF free-tier ephemeral disk). "
    "Use 'Download session audit log' BEFORE stepping away from the "
    "demo for >30 minutes to preserve."
)

audit_path: Path = default_audit_path()

if not audit_path.exists() or audit_path.stat().st_size == 0:
    st.info(
        "No audit events yet. Run a prediction on the Predict page first, "
        "then return here to view the chain."
    )
    st.stop()


def _load_audit_df(path: Path) -> pd.DataFrame:
    """Load JSONL audit log into a flat DataFrame.

    Each line in the JSONL is a single AuditEntry; pd.read_json with
    lines=True handles the streaming parse cleanly.
    """
    return pd.read_json(path, lines=True)


df = _load_audit_df(audit_path)
total_rows = len(df)

if total_rows > _ROW_CAP:
    df_display = df.tail(_ROW_CAP)
    st.info(
        f"Showing last {_ROW_CAP:,} of {total_rows:,} entries (oldest "
        "entries trimmed for display; full chain still intact in the "
        "underlying file). Use Download CSV to export the full session."
    )
else:
    df_display = df

# st.table - true HTML <table> semantics for screen readers.
st.table(df_display)


# CSV download button.
session_id = st.session_state.get("session_id", "unknown")
ts = (
    datetime.datetime.now(datetime.timezone.utc)
    .isoformat()
    .replace(":", "-")
)
filename = f"amoebanator_audit_{session_id}_{ts}.csv"
csv_bytes = export_audit_to_csv(audit_path)
st.download_button(
    label="Download session audit log (CSV)",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
    key="download_audit_csv",
)

# Small footer surfacing the schema version + chain integrity contract
# so a reviewer reading the downloaded CSV knows what to verify.
st.caption(
    "Exported under CSV schema version 1. To verify chain integrity, "
    "run `app.audit_export.verify_csv_chain_integrity(<bytes>)` on the "
    "downloaded file from a cloned repo. Each row includes "
    "`previous_hash` + `entry_hash` columns; the verifier recomputes "
    "the hash and asserts the chain links."
)
