"""
Phase 7.2 - production wiring of ml.data.deidentification into the data load path.

`ml.data.deidentification.SafeHarborProcessor` implements 45 CFR §164.514(b)(2)
(removal of the 18 HIPAA identifier categories, age cap at 89, ZIP truncation
to 3 digits, date generalisation to year). This module wraps it so that every
training run that touches a real-data CSV first passes through the Safe Harbor
scrubber.

Decision: scrub at load time (not write time). Scrubbing at write time would
leave a brief window where unscrubbed PHI sits in memory inside the trainer;
scrubbing at load time means the model never sees identifiers it shouldn't.

Field mapping for the bundled simulated dataset
(outputs/diagnosis_log_pro.csv):

  * `case_id`      - opaque UUID, NOT a HIPAA identifier; passes through.
  * `physician`    - actor name; treated as a *user* identifier per the
                     Safe Harbor catch-all (b)(2)(ii). Always scrubbed.
  * `age`          - capped at 89 per (b)(2)(i)(C).
  * `sex`          - demographic, not an identifier; passes through.
  * `csf_*`        - clinical labs; pass through.
  * `symptoms`     - clinical free text; passes through (no PHI in the
                     bundled vocabulary).
  * `comments`     - free-text; subject to length-based scrubbing inside
                     SafeHarborProcessor.
  * `timestamp_tz` - date; truncated to year per (b)(2)(i)(C).
  * `risk_score`, `risk_label`, `pcr`, `microscopy`, `exposure` - clinical;
                     pass through.

The bundled CSV has `source="simulated"` on every row, so this scrub is a
no-op data-shape verification today; once a real-data CSV (e.g., MIMIC-IV
extracts) is dropped in, the scrub becomes load-bearing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.audit_hooks import _emit
from ml.data.audit_trail import AuditEventType
from ml.data.deidentification import SafeHarborConfig, SafeHarborProcessor


SKIP_DEIDENT_ENV: str = "AMOEBANATOR_SKIP_DEIDENT"

# Columns we explicitly preserve (the scrubber's free-text rule otherwise
# triggers length-based scrubbing on long-but-clinical fields).
_CLINICAL_PASSTHROUGH: frozenset[str] = frozenset({
    "case_id", "sex", "csf_glucose", "csf_protein", "csf_wbc",
    "symptoms", "pcr", "microscopy", "exposure", "risk_score", "risk_label",
    "source",
})

# Columns whose values we always blank because they identify a user/actor.
_ACTOR_COLUMNS: frozenset[str] = frozenset({"physician"})


@dataclass(frozen=True)
class DeidentSummary:
    """One-row summary returned alongside the scrubbed dataframe."""
    n_rows: int
    n_actor_blanked: int
    n_age_capped: int
    n_dates_truncated: int
    config_age_cap: int
    bypassed: bool


def _truncate_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = 0
    for col in df.columns:
        if "date" in col.lower() or "timestamp" in col.lower() or col.lower() == "timestamp_tz":
            mask = df[col].notna()
            if mask.any():
                df.loc[mask, col] = pd.to_datetime(df.loc[mask, col], errors="coerce", utc=True).dt.year.astype("Int64").astype(str)
                n += int(mask.sum())
    return df, n


def _cap_ages(df: pd.DataFrame, cap: int) -> tuple[pd.DataFrame, int]:
    if "age" not in df.columns:
        return df, 0
    ages = pd.to_numeric(df["age"], errors="coerce")
    over = ages.fillna(0).astype(float) > cap
    n = int(over.sum())
    if n:
        df.loc[over, "age"] = cap
    return df, n


def _blank_actors(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = 0
    for col in _ACTOR_COLUMNS:
        if col in df.columns:
            mask = df[col].notna() & (df[col].astype(str) != "")
            n += int(mask.sum())
            df[col] = ""
    return df, n


def deidentify_dataframe(
    df: pd.DataFrame,
    config: SafeHarborConfig | None = None,
    *,
    bypass: bool | None = None,
) -> tuple[pd.DataFrame, DeidentSummary]:
    """
    Apply Safe Harbor de-identification to a dataframe in place-ish (returns a
    copy). When `bypass=True` (or env var AMOEBANATOR_SKIP_DEIDENT=1) the call
    is a no-op and the summary records `bypassed=True` so downstream auditing
    can flag it.
    """
    if bypass is None:
        bypass = os.environ.get(SKIP_DEIDENT_ENV, "").strip() in {"1", "true", "TRUE", "yes"}
    if bypass:
        return df.copy(), DeidentSummary(
            n_rows=int(len(df)),
            n_actor_blanked=0,
            n_age_capped=0,
            n_dates_truncated=0,
            config_age_cap=(config or SafeHarborConfig()).age_cap,
            bypassed=True,
        )

    cfg = config or SafeHarborConfig()
    out = df.copy()
    out, n_actor = _blank_actors(out)
    out, n_age = _cap_ages(out, cfg.age_cap)
    out, n_dates = _truncate_dates(out)

    # Run the upstream SafeHarborProcessor on a per-row dict to catch any
    # remaining identifier categories (free-text scrubbing, etc.).
    proc = SafeHarborProcessor(cfg)
    cols = [c for c in out.columns if c not in _CLINICAL_PASSTHROUGH]
    if cols:
        scrubbed_rows: list[dict[str, Any]] = []
        for _, row in out[cols].iterrows():
            row_dict: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
            scrubbed_rows.append(proc.process_record(row_dict))
        scrubbed = pd.DataFrame(scrubbed_rows, index=out.index)
        for c in scrubbed.columns:
            out[c] = scrubbed[c]

    summary = DeidentSummary(
        n_rows=int(len(out)),
        n_actor_blanked=n_actor,
        n_age_capped=n_age,
        n_dates_truncated=n_dates,
        config_age_cap=cfg.age_cap,
        bypassed=False,
    )
    return out, summary


def load_tabular_safe_harbor(
    csv_path: str = "outputs/diagnosis_log_pro.csv",
    config: SafeHarborConfig | None = None,
    *,
    bypass: bool | None = None,
    emit_audit: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], DeidentSummary]:
    """
    Drop-in replacement for ml.training.load_tabular that scrubs the dataframe
    before vectorising. Returns (X, y, feature_names, summary) so callers can
    log the de-identification report.
    """
    df = pd.read_csv(csv_path)
    df, summary = deidentify_dataframe(df, config=config, bypass=bypass)

    all_symptoms: set[str] = set()
    for s in df["symptoms"].astype(str):
        for token in [t for t in s.split(";") if t]:
            all_symptoms.add(token)
    for sym in sorted(all_symptoms):
        df[f"sym_{sym}"] = df["symptoms"].astype(str).apply(lambda x, sym=sym: 1 if sym in x.split(";") else 0)
    feats = ["age", "csf_glucose", "csf_protein", "csf_wbc", "pcr", "microscopy", "exposure"] + [c for c in df.columns if c.startswith("sym_")]
    X = df[feats].fillna(0).astype(float).values
    y = (df["risk_label"].astype(str).str.lower() == "high").astype(int).values

    if emit_audit:
        _emit(
            AuditEventType.DATA_VERIFIED,
            actor="ml.data_loader.load_tabular_safe_harbor",
            resource=csv_path,
            action_detail="Safe Harbor de-identification applied",
            metadata={
                "n_rows": summary.n_rows,
                "n_actor_blanked": summary.n_actor_blanked,
                "n_age_capped": summary.n_age_capped,
                "n_dates_truncated": summary.n_dates_truncated,
                "bypassed": summary.bypassed,
            },
        )

    return X, y, feats, summary  # type: ignore[return-value]
