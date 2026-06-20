"""V1.5 -> V2.0 migration test skeleton.

Full migration script implementation deferred to Subphase 1.4 (data pipeline phase).
This file documents expected migration semantics and provides a skip marker
so the test scaffolding exists when 1.4 lands.
"""
from __future__ import annotations

import pytest

from ml.schemas.labels import ClassLabel


@pytest.mark.skip(reason="V1.5 migration script implemented in Subphase 1.4")
def test_v1_5_binary_to_v2_0_nine_class_migration() -> None:
    """V1.5 binary PAM/non-PAM -> V2.0 9-class differential.

    Expected migration semantics:
    - V1.5 vignette with binary_label=1 (PAM) -> V2.0 ground_truth_class=ClassLabel.PAM
    - V1.5 vignette with binary_label=0 (non-PAM) -> V2.0 ground_truth_class=ClassLabel.NON_INFECTIOUS_MIMIC
      as default fallback; physician adjudicator must reclassify into specific
      non-PAM class (BACTERIAL, VIRAL, TUBERCULOUS, etc.) before training use.
    - Schema version field migrates "1.5" -> "2.0".
    - All V1.5 fields preserved; new V2.0-only fields populated as None.

    Implementation in Subphase 1.4: ml/schemas/migrations/v1_5_to_v2_0.py
    """
    raise NotImplementedError("Migration script lands in Subphase 1.4")


def test_class_label_enum_stable() -> None:
    """Sanity check: ClassLabel enum order matches V2.0 schema lock."""
    expected_order = [
        ("PAM", 1),
        ("BACTERIAL", 2),
        ("VIRAL", 3),
        ("TUBERCULOUS", 4),
        ("CRYPTOCOCCAL_FUNGAL", 5),
        ("GAE", 6),
        ("NEUROCYSTICERCOSIS", 7),
        ("CEREBRAL_MALARIA_OR_SEVERE_ARBO", 8),
        ("NON_INFECTIOUS_MIMIC", 9),
    ]
    for name, value in expected_order:
        member = ClassLabel[name]
        assert member.value == value, f"{name} should be {value}, got {member.value}"
