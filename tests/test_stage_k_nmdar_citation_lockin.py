"""Stage K addendum lock-in: anti-NMDAR fixture citation metadata is corrected.

The anti-NMDAR fixture anchor cited the wrong journal ("Case Rep Neurol Med") and
wrong first-author initial ("Keller A") for PMID 25400967, whose correct NLM citation
is Keller S, Roitman P, Ben-Hur T, Bonne O, Lotan A. Case Rep Psychiatry 2014;2014:868325
(DOI 10.1155/2014/868325). This pins the corrected journal and case_id across the
committed fixture and the builder dict so a regen or manual edit cannot restore it.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_NMDAR_FIX = _REPO_ROOT / "tests" / "schemas" / "fixtures" / "valid_nmdar_fixture.json"
_BUILDER_SRC = _REPO_ROOT / "scripts" / "vignettes" / "generate_subphase11_fixtures.py"
_WRONG_JOURNAL = ("Case Rep Neurol Med", "CaseRepNeurolMed")


def test_nmdar_fixture_citation_corrected():
    obj = json.load(io.open(_NMDAR_FIX, encoding="utf-8"))
    assert obj["case_id"] == "NMIM-001-CaseRepPsychiatry-2014-Keller"
    anchor = obj["adjudication"]["anchoring_documentation"]
    assert "Case Rep Psychiatry" in anchor
    for w in _WRONG_JOURNAL:
        assert w not in anchor
    assert "Keller A," not in anchor


def test_nmdar_builder_citation_corrected():
    import scripts.vignettes.generate_subphase11_fixtures as g

    nmdar = dict(g.FIXTURES)["valid_nmdar_fixture.json"]
    assert nmdar["case_id"] == "NMIM-001-CaseRepPsychiatry-2014-Keller"
    assert "Case Rep Psychiatry" in nmdar["adjudication"]["anchoring_documentation"]
    src = _BUILDER_SRC.read_text(encoding="utf-8")
    for w in _WRONG_JOURNAL:
        assert w not in src
