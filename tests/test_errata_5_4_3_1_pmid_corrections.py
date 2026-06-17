"""Errata 5.4.3.1 tests: 3 catastrophic PMID corrections.

Pre-medRxiv NCBI E-utilities verification on 2026-05-11 revealed 3 PMIDs in
PMID_REGISTRY pointed to completely unrelated papers. Fixed in this errata:

- 29490180 -> 30089069  Tyler 'Acute Viral Encephalitis' NEJM 2018
  (was a NEJM Letter on breast cancer recurrence, Pan H 2017)
- 21088000 -> 20952256  Granerod 'Causes of encephalitis' Lancet ID 2010
  (was Thorne 2011 Nucleic Acids Res epigenetics paper)
- 16517432 -> 16675036  Whitley 'Herpes simplex encephalitis' Antiviral Res 2006
  (was J Asthma 2006 Danish skin test reactivity)

Clinical content invariance: the per-vignette hash of clinical-data fields
(history + vitals + exam + labs + csf + imaging + normalized diagnostic_tests
results stripped of citation strings) MUST be identical pre vs post errata.
Only PMID strings change.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.schemas.vignette import VignetteSchema  # noqa: E402
from scripts.vignettes.generate_pam_vignettes import PMID_REGISTRY  # noqa: E402


WRONG_TO_RIGHT: dict[str, str] = {
    "29490180": "30089069",  # Tyler 2018 NEJM
    "21088000": "20952256",  # Granerod 2010 Lancet ID
    "16517432": "16675036",  # Whitley 2006 Antiviral Res
}

CORRECTED_METADATA: dict[str, dict] = {
    "30089069": {
        "title_substr": "Acute Viral Encephalitis",
        "first_author": "Tyler",
        "year": 2018,
        "journal_substr": "N Engl J Med",
        "anchor_type": "review",
        "doi": "10.1056/NEJMra1708714",
    },
    "20952256": {
        "title_substr": "Causes of encephalitis",
        "first_author": "Granerod",
        "year": 2010,
        "journal_substr": "Lancet Infect Dis",
        "anchor_type": "cohort",
        "doi": "10.1016/S1473-3099(10)70222-X",
    },
    "16675036": {
        "title_substr": "Herpes simplex encephalitis",
        "first_author": "Whitley",
        "year": 2006,
        "journal_substr": "Antiviral Res",
        "anchor_type": "review",
        "doi": "10.1016/j.antiviral.2006.04.002",
    },
}


def _all_vignette_files() -> list[Path]:
    paths: list[Path] = []
    for d in (
        REPO / "data/vignettes/pam",
        REPO / "data/vignettes/v2/class_02_bacterial",
        REPO / "data/vignettes/v2/class_03_viral",
        REPO / "data/vignettes/v2/class_04_tb",
        REPO / "data/vignettes/v2/class_05_fungal",
        REPO / "data/vignettes/v2/class_06_gae",
    ):
        if d.is_dir():
            paths.extend(p for p in sorted(d.glob("*.json")) if p.name != "marginals.json")
    return paths


def _git_show_head(p: Path) -> str:
    rel = p.relative_to(REPO)
    out = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        capture_output=True, text=True, cwd=REPO, check=True,
    )
    return out.stdout


def _clinical_hash(json_text: str) -> str:
    """Hash of clinical-content subset.

    Includes: history, vitals, exam, labs, csf, imaging, and diagnostic_tests
    results with citation_pmid_or_doi STRIPPED (the citation field is the only
    PMID-bearing entry inside diagnostic_tests; everything else - test_name,
    result, sensitivity/specificity - is clinical data that must not drift).
    """
    data = json.loads(json_text)
    subset: dict = {k: data.get(k) for k in ("history", "vitals", "exam",
                                              "labs", "csf", "imaging")}
    dx = data.get("diagnostic_tests")
    if dx and "results" in dx:
        subset["diagnostic_tests"] = {
            "results": [
                {k: v for k, v in r.items() if k != "citation_pmid_or_doi"}
                for r in dx["results"]
            ]
        }
    canonical = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Wrong PMIDs purged from corpus
# ----------------------------------------------------------------------


@pytest.mark.parametrize("wrong_pmid", list(WRONG_TO_RIGHT.keys()))
def test_wrong_pmid_purged_from_all_vignettes(wrong_pmid):
    found = [p.name for p in _all_vignette_files()
             if wrong_pmid in p.read_text(encoding="utf-8")]
    assert not found, (
        f"wrong PMID {wrong_pmid} still present in {len(found)} vignettes: "
        f"{found[:5]}{'...' if len(found) > 5 else ''}"
    )


def test_pmid_29490180_not_in_any_vignette():
    found = [p.name for p in _all_vignette_files()
             if "29490180" in p.read_text(encoding="utf-8")]
    assert not found, f"old PMID 29490180 in {len(found)} vignettes"


def test_pmid_21088000_not_in_any_vignette():
    found = [p.name for p in _all_vignette_files()
             if "21088000" in p.read_text(encoding="utf-8")]
    assert not found, f"old PMID 21088000 in {len(found)} vignettes"


def test_pmid_16517432_not_in_any_vignette():
    found = [p.name for p in _all_vignette_files()
             if "16517432" in p.read_text(encoding="utf-8")]
    assert not found, f"old PMID 16517432 in {len(found)} vignettes"


# ----------------------------------------------------------------------
# Wrong PMIDs purged from registry
# ----------------------------------------------------------------------


@pytest.mark.parametrize("wrong_pmid", list(WRONG_TO_RIGHT.keys()))
def test_wrong_pmid_not_in_registry(wrong_pmid):
    assert wrong_pmid not in PMID_REGISTRY, (
        f"wrong PMID {wrong_pmid} still in PMID_REGISTRY"
    )


# ----------------------------------------------------------------------
# Corrected PMIDs present with correct NCBI-verified metadata
# ----------------------------------------------------------------------


def test_pmid_30089069_present_in_registry_with_tyler_2018_metadata():
    assert "30089069" in PMID_REGISTRY
    meta = PMID_REGISTRY["30089069"]
    assert "Tyler" in meta["authors_short"]
    assert meta["year"] == 2018
    journal = meta.get("journal", "") + " " + meta.get("journal_short_code", "")
    assert "N Engl J Med" in journal or "NEJM" in journal
    assert "Acute Viral Encephalitis".lower() in meta["title"].lower()
    assert meta["doi"] == "10.1056/NEJMra1708714"
    assert meta["anchor_type"] == "review"


def test_pmid_20952256_present_in_registry_with_granerod_2010_metadata():
    assert "20952256" in PMID_REGISTRY
    meta = PMID_REGISTRY["20952256"]
    assert "Granerod" in meta["authors_short"]
    assert meta["year"] == 2010
    assert "Lancet Infect Dis" in meta["journal"]
    assert "Causes of encephalitis".lower() in meta["title"].lower()
    assert meta["doi"] == "10.1016/S1473-3099(10)70222-X"
    assert meta["anchor_type"] == "cohort"


def test_pmid_16675036_present_in_registry_with_whitley_2006_metadata():
    assert "16675036" in PMID_REGISTRY
    meta = PMID_REGISTRY["16675036"]
    assert "Whitley" in meta["authors_short"]
    assert meta["year"] == 2006
    assert "Antiviral Res" in meta["journal"]
    assert "Herpes simplex encephalitis".lower() in meta["title"].lower()
    assert meta["doi"] == "10.1016/j.antiviral.2006.04.002"
    assert meta["anchor_type"] == "review"


# ----------------------------------------------------------------------
# Schema validation + clinical-content invariance
# ----------------------------------------------------------------------


def _affected_files_post_correction() -> list[Path]:
    """Files that contain any of the 3 corrected (right) PMIDs on disk now."""
    right_pmids = set(WRONG_TO_RIGHT.values())
    out = []
    for p in _all_vignette_files():
        text = p.read_text(encoding="utf-8")
        if any(rp in text for rp in right_pmids):
            out.append(p)
    return out


def test_corrected_vignettes_schema_validate_all():
    affected = _affected_files_post_correction()
    assert affected, "no vignettes found with corrected PMIDs (errata not applied?)"
    for p in affected:
        VignetteSchema.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _affected_files_pre_correction() -> list[Path]:
    """Files that contained any of the 3 WRONG PMIDs at HEAD (pre-errata commit)."""
    wrong_pmids = set(WRONG_TO_RIGHT.keys())
    affected = set()
    for wp in wrong_pmids:
        out = subprocess.run(
            ["git", "grep", "-l", wp, "HEAD", "--", "data/vignettes/"],
            capture_output=True, text=True, cwd=REPO,
        )
        if out.returncode != 0:
            continue
        for line in out.stdout.strip().split("\n"):
            if not line:
                continue
            _, _, rel = line.partition(":")
            affected.add(REPO / rel)
    return sorted(affected)


def test_corrected_vignettes_clinical_content_unchanged():
    """Pre-correction (git HEAD) vs post-correction (disk) clinical-content
    hashes MUST be identical for every affected vignette. Verifies only PMID
    strings changed; zero clinical drift."""
    affected = _affected_files_pre_correction()
    if not affected:
        pytest.skip(
            "errata 5.4.3.1 already committed to HEAD; no pre-correction "
            "files remain to diff; this migration check is obsolete once "
            "the corrected PMIDs are in HEAD"
        )
    drift = []
    for p in affected:
        try:
            pre = _git_show_head(p)
        except Exception as exc:
            pytest.skip(f"git show failed for {p.name}: {exc}")
        post = p.read_text(encoding="utf-8")
        h_pre = _clinical_hash(pre)
        h_post = _clinical_hash(post)
        if h_pre != h_post:
            drift.append((p.name, h_pre[:12], h_post[:12]))
    assert not drift, f"clinical drift in {len(drift)} files: {drift[:3]}"
