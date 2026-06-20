"""Roundtrip serialization test for VignetteSchema fixtures.

For each fixture: load JSON -> validate -> dump -> re-validate -> assert deep equality.
Confirms idempotent serialization across all 9 ClassLabel values.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.schemas.vignette import VignetteSchema

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATHS = sorted(FIXTURES_DIR.glob("valid_*_fixture.json"))


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_PATHS,
    ids=[p.stem for p in FIXTURE_PATHS],
)
def test_vignette_roundtrip(fixture_path: Path) -> None:
    """Load -> validate -> serialize -> re-validate -> assert equality."""
    raw = json.loads(fixture_path.read_text())
    obj1 = VignetteSchema.model_validate(raw)
    serialized = obj1.model_dump_json()
    obj2 = VignetteSchema.model_validate_json(serialized)
    assert obj1.model_dump() == obj2.model_dump(), (
        f"Roundtrip mismatch for {fixture_path.name}"
    )


def test_all_class_labels_covered() -> None:
    """Sanity check: 9 fixtures cover all 9 ClassLabel enum values."""
    seen_classes = set()
    for path in FIXTURE_PATHS:
        raw = json.loads(path.read_text())
        obj = VignetteSchema.model_validate(raw)
        seen_classes.add(obj.ground_truth_class.name)
    expected = {"PAM", "BACTERIAL", "VIRAL", "TUBERCULOUS", "CRYPTOCOCCAL_FUNGAL",
                "GAE", "NEUROCYSTICERCOSIS", "CEREBRAL_MALARIA_OR_SEVERE_ARBO",
                "NON_INFECTIOUS_MIMIC"}
    assert seen_classes == expected, f"Missing classes: {expected - seen_classes}"
