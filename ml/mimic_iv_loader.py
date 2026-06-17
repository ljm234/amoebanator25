"""
Phase 2.2 - MIMIC-IV CSF lab + diagnosis loader (scaffold).

Cannot be executed against real MIMIC-IV data without a signed PhysioNet
Data Use Agreement (see docs/USER_ASSIGNMENTS.md). Until that completes,
this module exposes:

  * Verified itemids and ICD-10 codes (from MIMIC-IV d_labitems and the
    FY2026 ICD-10-CM tabular list).
  * A loader that operates on PhysioNet-shaped CSV files and returns a
    cohort dataframe with the same column schema the Amoebanator pipeline
    expects.
  * `synthesize_mimic_shaped_csvs()` - produces tiny PhysioNet-shaped CSVs
    in a temp directory so the loader's row-extraction logic is tested
    end-to-end before real data arrives. The synthesised data is marked
    `subject_id` in a synthetic range so it cannot collide with real IDs.

Verified itemids (from PhysioNet d_labitems, MIMIC-IV demo v2.2 / v3.1):

  | itemid | label                      | use                                  |
  |--------|----------------------------|--------------------------------------|
  | 51790  | Glucose, CSF               | csf_glucose (mg/dL)                  |
  | 51802  | Total Protein, CSF         | csf_protein (mg/dL)                  |
  | 52286  | Total Nucleated Cells, CSF | csf_wbc surrogate (cells/µL)         |
  | 52281  | Polys                      | neutrophil % of CSF nucleated cells  |
  | 52264  | Lymphs                     | lymphocyte %                         |

CSF microbiology lives in `hosp.microbiologyevents` filtered on
`spec_type_desc == 'CSF;SPINAL FLUID'`. Gram stain results are stored in
`test_name == 'GRAM STAIN'` with text in `comments`/`test_text`.

ICD-10-CM codes (FY2026, effective 2025-10-01):

  * G00.0-G00.9  Bacterial meningitis (G00.x)
  * A87.0-A87.9  Viral meningitis (A87.x)
  * B60.2        Naegleriasis (PAM)

In MIMIC-IV `diagnoses_icd`, codes are stored without the dot
(e.g., "G001", "A870", "B602") and `icd_version` flags 9 vs 10.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CSF_ITEMIDS: dict[int, str] = {
    51790: "csf_glucose",
    51802: "csf_protein",
    52286: "csf_wbc",
    52281: "csf_polys_pct",
    52264: "csf_lymphs_pct",
}

CSF_SPEC_TYPE_DESC: str = "CSF;SPINAL FLUID"

ICD10_BACTERIAL_PREFIX: str = "G00"
ICD10_VIRAL_PREFIX: str = "A87"
ICD10_AMEBIC_PREFIX: str = "B60"
ICD10_NAEGLERIA_CODE: str = "B602"


@dataclass(frozen=True)
class MimicCohortConfig:
    """Configuration for the MIMIC-IV CSF cohort builder."""
    require_csf_lab: bool = True
    require_icd_meningitis: bool = True
    bacterial_codes: tuple[str, ...] = (
        "G000", "G001", "G002", "G003", "G008", "G009",
    )
    viral_codes: tuple[str, ...] = (
        "A870", "A871", "A872", "A878", "A879",
    )
    amebic_codes: tuple[str, ...] = ("B602",)


def _normalize_icd_code(code: object) -> str:
    """MIMIC-IV stores ICD-10 codes without dots; normalise comparisons."""
    if code is None:
        return ""
    s = str(code).strip().upper().replace(".", "")
    return s


def label_from_icd(
    diagnoses: pd.DataFrame,
    cfg: MimicCohortConfig | None = None,
) -> dict[int, str]:
    """
    Map subject_id → label in {"bacterial","viral","amebic","other"} given
    the diagnoses_icd table for one cohort. Multiple matching codes
    resolve in priority order: amebic > bacterial > viral > other.
    """
    cfg = cfg or MimicCohortConfig()
    needed_cols = {"subject_id", "icd_code", "icd_version"}
    missing = needed_cols - set(diagnoses.columns)
    if missing:
        raise ValueError(f"diagnoses df missing columns: {sorted(missing)}")

    icd10 = diagnoses[diagnoses["icd_version"] == 10].copy()
    icd10["code_norm"] = icd10["icd_code"].map(_normalize_icd_code)
    out: dict[int, str] = {}
    for sid, sub in icd10.groupby("subject_id"):
        sid_int = int(sid)  # type: ignore[arg-type]
        codes = set(sub["code_norm"].tolist())
        if codes & set(cfg.amebic_codes):
            out[sid_int] = "amebic"
        elif codes & set(cfg.bacterial_codes):
            out[sid_int] = "bacterial"
        elif codes & set(cfg.viral_codes):
            out[sid_int] = "viral"
        else:
            out[sid_int] = "other"
    return out


def csf_labs_per_subject(
    labevents: pd.DataFrame,
    itemids: Iterable[int] = tuple(CSF_ITEMIDS.keys()),
) -> pd.DataFrame:
    """
    Pivot CSF lab values from labevents long-form to one row per subject_id.
    Takes the median per (subject_id, itemid) when multiple measurements exist.
    """
    needed_cols = {"subject_id", "itemid", "valuenum"}
    missing = needed_cols - set(labevents.columns)
    if missing:
        raise ValueError(f"labevents df missing columns: {sorted(missing)}")
    targets = set(int(i) for i in itemids)
    df = labevents[labevents["itemid"].isin(targets)].copy()
    if df.empty:
        return pd.DataFrame(columns=["subject_id"] + [CSF_ITEMIDS[i] for i in targets])
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.dropna(subset=["valuenum"])
    pivot = (
        df.groupby(["subject_id", "itemid"])["valuenum"].median().unstack().reset_index()
    )
    rename = {iid: CSF_ITEMIDS[iid] for iid in CSF_ITEMIDS if iid in pivot.columns}
    pivot = pivot.rename(columns=rename)
    return pivot


def gram_culture_per_subject(microbiology: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce microbiologyevents (filtered for CSF) to one row per subject_id
    with two booleans: any_positive_culture, any_organism_seen_on_gram.
    """
    needed_cols = {"subject_id", "spec_type_desc"}
    missing = needed_cols - set(microbiology.columns)
    if missing:
        raise ValueError(f"microbiology df missing columns: {sorted(missing)}")
    df = microbiology[microbiology["spec_type_desc"].astype(str).str.upper() == CSF_SPEC_TYPE_DESC].copy()
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "any_positive_culture", "any_organism_on_gram"])
    if "org_name" not in df.columns:
        df["org_name"] = ""
    if "test_name" not in df.columns:
        df["test_name"] = ""
    df["org_name"] = df["org_name"].astype(str).fillna("")
    df["test_name"] = df["test_name"].astype(str).fillna("")
    df["_pos_culture"] = (df["org_name"].str.strip() != "") & (df["org_name"].str.lower() != "nan")
    df["_gram_seen"] = df["test_name"].str.upper().str.contains("GRAM")
    grouped = df.groupby("subject_id", as_index=False).agg(
        any_positive_culture=("_pos_culture", "any"),
        any_organism_on_gram=("_gram_seen", "any"),
    )
    grouped["any_positive_culture"] = grouped["any_positive_culture"].astype(bool)
    grouped["any_organism_on_gram"] = grouped["any_organism_on_gram"].astype(bool)
    return grouped


def assemble_cohort(
    labevents: pd.DataFrame,
    diagnoses: pd.DataFrame,
    microbiology: pd.DataFrame | None = None,
    cfg: MimicCohortConfig | None = None,
) -> pd.DataFrame:
    """
    Join labs + diagnosis labels + (optional) microbiology into a single
    Amoebanator-shaped row per subject_id. Returns columns the rest of the
    pipeline understands: age (placeholder NaN - comes from patients table),
    csf_glucose, csf_protein, csf_wbc, pcr (NaN - needs separate query),
    microscopy (from gram), exposure (NaN - not in MIMIC), risk_label
    (mapped from ICD), source="mimic_iv".
    """
    cfg = cfg or MimicCohortConfig()
    labs = csf_labs_per_subject(labevents)
    label_map = label_from_icd(diagnoses, cfg)
    if not label_map:
        return pd.DataFrame()
    base = labs if not labs.empty else pd.DataFrame({"subject_id": list(label_map.keys())})
    base = base.merge(
        pd.DataFrame({"subject_id": list(label_map.keys()), "icd_label": list(label_map.values())}),
        on="subject_id", how="outer",
    )
    if microbiology is not None and not microbiology.empty:
        mic = gram_culture_per_subject(microbiology)
        base = base.merge(mic, on="subject_id", how="left")
        base["microscopy"] = base["any_organism_on_gram"].fillna(False).astype(int)
    else:
        base["microscopy"] = 0

    if cfg.require_csf_lab:
        present_lab_cols = [c for c in CSF_ITEMIDS.values() if c in base.columns]
        if not present_lab_cols:
            return base.iloc[0:0].copy()
        has_lab = base[present_lab_cols].notna().any(axis=1)
        base = base[has_lab].copy()
    if cfg.require_icd_meningitis:
        base = base[base["icd_label"].isin({"bacterial", "viral", "amebic"})].copy()

    base["risk_label"] = np.where(base["icd_label"].isin({"bacterial", "amebic"}), "High", "Low")
    base["source"] = "mimic_iv"
    base["physician"] = "mimic_iv"
    base["pcr"] = np.nan
    base["exposure"] = np.nan
    base["age"] = np.nan
    base["symptoms"] = ""
    return base.reset_index(drop=True)


def synthesize_mimic_shaped_csvs(
    out_dir: Path,
    n_subjects: int = 30,
    seed: int = 0,
) -> dict[str, Path]:
    """
    Write tiny PhysioNet-shaped CSVs (labevents, diagnoses_icd,
    microbiologyevents) so the loader can be smoke-tested without DUA.
    Subject IDs start at 9_000_000 so they cannot overlap with real ones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sids = np.arange(9_000_000, 9_000_000 + n_subjects)

    lab_rows = []
    for sid in sids:
        for iid in CSF_ITEMIDS:
            lab_rows.append({
                "subject_id": int(sid),
                "hadm_id": int(sid) * 10,
                "itemid": int(iid),
                "valuenum": float(rng.uniform(0, 500)),
                "valueuom": "x",
                "charttime": "2173-01-01 00:00:00",
            })
    pd.DataFrame(lab_rows).to_csv(out_dir / "labevents.csv", index=False)

    dx_rows = []
    cfg = MimicCohortConfig()
    code_pool = list(cfg.bacterial_codes) + list(cfg.viral_codes) + list(cfg.amebic_codes) + ["Z000"]
    for sid in sids:
        code = rng.choice(code_pool)
        dx_rows.append({
            "subject_id": int(sid),
            "hadm_id": int(sid) * 10,
            "seq_num": 1,
            "icd_code": str(code),
            "icd_version": 10,
        })
    pd.DataFrame(dx_rows).to_csv(out_dir / "diagnoses_icd.csv", index=False)

    mb_rows = []
    for sid in sids:
        positive = rng.random() < 0.4
        mb_rows.append({
            "subject_id": int(sid),
            "spec_type_desc": "CSF;SPINAL FLUID",
            "test_name": "GRAM STAIN",
            "org_name": "STAPHYLOCOCCUS AUREUS" if positive else "",
        })
    pd.DataFrame(mb_rows).to_csv(out_dir / "microbiologyevents.csv", index=False)

    manifest = {
        "labevents": str(out_dir / "labevents.csv"),
        "diagnoses_icd": str(out_dir / "diagnoses_icd.csv"),
        "microbiologyevents": str(out_dir / "microbiologyevents.csv"),
        "n_subjects": int(n_subjects),
        "subject_id_range": [int(sids.min()), int(sids.max())],
        "schema_version": "mimic_iv_v22_compatible",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    skip = {"n_subjects", "subject_id_range", "schema_version"}
    return {k: Path(str(v)) for k, v in manifest.items() if k not in skip}


def load_cohort_from_csvs(
    labevents_csv: Path,
    diagnoses_csv: Path,
    microbiology_csv: Path | None = None,
    cfg: MimicCohortConfig | None = None,
) -> pd.DataFrame:
    """End-to-end: read CSVs, run assembly, return a cohort dataframe."""
    labs = pd.read_csv(labevents_csv)
    dx = pd.read_csv(diagnoses_csv)
    mic = pd.read_csv(microbiology_csv) if microbiology_csv and Path(microbiology_csv).exists() else None
    return assemble_cohort(labs, dx, mic, cfg)


def task_balance(cohort: pd.DataFrame) -> dict[str, int]:
    """One-line task summary: bacterial / viral / amebic / other counts."""
    if "icd_label" not in cohort.columns:
        return {}
    return {str(k): int(v) for k, v in cohort["icd_label"].value_counts().to_dict().items()}
