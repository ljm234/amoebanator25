"""Phase 2.2 - tests for ml.mimic_iv_loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.mimic_iv_loader import (
    CSF_ITEMIDS,
    CSF_SPEC_TYPE_DESC,
    ICD10_NAEGLERIA_CODE,
    MimicCohortConfig,
    _normalize_icd_code,
    assemble_cohort,
    csf_labs_per_subject,
    gram_culture_per_subject,
    label_from_icd,
    load_cohort_from_csvs,
    synthesize_mimic_shaped_csvs,
    task_balance,
)


def test_csf_itemids_include_glucose_protein_wbc() -> None:
    assert 51790 in CSF_ITEMIDS  # glucose
    assert 51802 in CSF_ITEMIDS  # protein
    assert 52286 in CSF_ITEMIDS  # total nucleated cells (WBC surrogate)


def test_normalize_icd_code_strips_dots_and_uppercases() -> None:
    assert _normalize_icd_code("g00.1") == "G001"
    assert _normalize_icd_code(" b60.2 ") == "B602"
    assert _normalize_icd_code(None) == ""


def test_naegleria_code_is_b602() -> None:
    assert ICD10_NAEGLERIA_CODE == "B602"


def test_label_from_icd_priority_order() -> None:
    """A subject with codes from multiple categories resolves amebic > bacterial > viral > other."""
    df = pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4],
        "icd_code": ["G001", "B602", "A871", "G002", "Z000"],
        "icd_version": [10, 10, 10, 10, 10],
    })
    out = label_from_icd(df)
    assert out[1] == "amebic"      # B602 wins over G001
    assert out[2] == "viral"
    assert out[3] == "bacterial"
    assert out[4] == "other"


def test_label_from_icd_ignores_icd9() -> None:
    df = pd.DataFrame({
        "subject_id": [1],
        "icd_code": ["G001"],
        "icd_version": [9],
    })
    out = label_from_icd(df)
    assert out == {}


def test_label_from_icd_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        label_from_icd(pd.DataFrame({"subject_id": [1]}))


def test_csf_labs_per_subject_pivots_correctly() -> None:
    df = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "itemid": [51790, 51802, 51790],
        "valuenum": [50.0, 100.0, 30.0],
    })
    pivot = csf_labs_per_subject(df)
    # Subject 1 has both glucose and protein
    s1 = pivot[pivot["subject_id"] == 1].iloc[0]
    assert s1["csf_glucose"] == 50.0
    assert s1["csf_protein"] == 100.0


def test_csf_labs_takes_median_for_repeated_measures() -> None:
    df = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "itemid": [51790, 51790, 51790],
        "valuenum": [40.0, 50.0, 60.0],
    })
    pivot = csf_labs_per_subject(df)
    assert pivot.iloc[0]["csf_glucose"] == 50.0  # median


def test_csf_labs_empty_returns_empty_df() -> None:
    df = pd.DataFrame({"subject_id": [1], "itemid": [99999], "valuenum": [1.0]})
    pivot = csf_labs_per_subject(df)
    assert pivot.empty or pivot.shape[0] == 0


def test_gram_culture_per_subject_filters_csf_only() -> None:
    df = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "spec_type_desc": [CSF_SPEC_TYPE_DESC, "BLOOD CULTURE", CSF_SPEC_TYPE_DESC],
        "test_name": ["GRAM STAIN", "GRAM STAIN", "GRAM STAIN"],
        "org_name": ["", "STAPH AUREUS", "STREP PNEUMONIAE"],
    })
    out = gram_culture_per_subject(df)
    # Subject 1 has CSF gram (no organism), subject 2 has CSF positive
    assert set(out["subject_id"].tolist()) == {1, 2}
    s2 = out[out["subject_id"] == 2].iloc[0]
    assert s2["any_positive_culture"] is np.True_ or bool(s2["any_positive_culture"]) is True


def test_assemble_cohort_handles_missing_microbiology() -> None:
    labs = pd.DataFrame({"subject_id": [1, 2], "itemid": [51790, 51790], "valuenum": [50.0, 30.0]})
    dx = pd.DataFrame({"subject_id": [1, 2], "icd_code": ["G001", "A871"], "icd_version": [10, 10]})
    cohort = assemble_cohort(labs, dx, microbiology=None)
    assert len(cohort) == 2
    assert set(cohort["icd_label"].tolist()) == {"bacterial", "viral"}
    assert (cohort["microscopy"] == 0).all()
    assert (cohort["source"] == "mimic_iv").all()


def test_synthesize_mimic_shaped_csvs_roundtrip(tmp_path: Path) -> None:
    paths = synthesize_mimic_shaped_csvs(tmp_path / "mimic", n_subjects=20, seed=0)
    assert all(p.exists() for p in paths.values())
    cohort = load_cohort_from_csvs(paths["labevents"], paths["diagnoses_icd"], paths["microbiologyevents"])
    assert len(cohort) > 0
    # All synthetic subject_ids must be in the 9_xxx_xxx range
    assert (cohort["subject_id"] >= 9_000_000).all()
    assert (cohort["subject_id"] < 9_001_000).all()


def test_synthesize_includes_amebic_codes(tmp_path: Path) -> None:
    paths = synthesize_mimic_shaped_csvs(tmp_path / "mimic", n_subjects=200, seed=42)
    cohort = load_cohort_from_csvs(paths["labevents"], paths["diagnoses_icd"], paths["microbiologyevents"])
    balance = task_balance(cohort)
    assert "bacterial" in balance
    assert "viral" in balance
    # With n=200 and 1 amebic code in pool of ~14, expect at least one
    assert balance.get("amebic", 0) > 0


def test_task_balance_handles_empty() -> None:
    assert task_balance(pd.DataFrame()) == {}


def test_assemble_requires_lab_when_configured() -> None:
    labs = pd.DataFrame({"subject_id": [], "itemid": [], "valuenum": []})
    dx = pd.DataFrame({"subject_id": [1], "icd_code": ["G001"], "icd_version": [10]})
    cfg = MimicCohortConfig(require_csf_lab=True)
    cohort = assemble_cohort(labs, dx, cfg=cfg)
    assert cohort.empty


def test_csf_labs_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        csf_labs_per_subject(pd.DataFrame({"subject_id": [1]}))
