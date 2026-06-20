"""
Phase 1.1 De-identification Module - Comprehensive Test Suite.

Tests cover:
  - Safe Harbor processing (18 identifiers, age cap, ZIP truncation,
    date generalisation, free-text scrubbing, pseudonymization)
  - k-Anonymity enforcement (equivalence classes, generalisation,
    suppression, information loss metric)
  - Differential privacy mechanisms (Laplace, Gaussian, Exponential)
  - Privacy budget tracking and allocation
  - Full pipeline orchestration across all three layers
  - Factory functions and report generation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from ml.data.deidentification import (
    DeidentificationConfig,
    DeidentificationMethod,
    ExponentialMechanism,
    GaussianMechanism,
    KAnonymityConfig,
    KAnonymityProcessor,
    LaplaceMechanism,
    PrivacyBudget,
    PrivacyLevel,
    SafeHarborConfig,
    SafeHarborProcessor,
    create_deidentification_pipeline,
)


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_clinical_record(**overrides: object) -> dict[str, object]:
    """Build a sample clinical record with optional overrides."""
    base: dict[str, object] = {
        "patient_id": "PAT-001",
        "name": "John Doe",
        "email": "jdoe@example.com",
        "ssn": "123-45-6789",
        "phone": "801-555-0100",
        "date_of_birth": "1985-06-15",
        "zip_code": "84408",
        "age": 38,
        "sex": "M",
        "geographic_region": "Utah, Western USA",
        "csf_glucose": 45.0,
        "csf_protein": 120.0,
        "csf_wbc": 350.0,
        "diagnosis": "suspected_pam",
        "collection_date": "2025-09-10T14:30:00",
    }
    base.update(overrides)
    return base


def _make_record_batch(n: int = 20) -> list[dict[str, object]]:
    """Build a batch of clinical records for k-anonymity testing."""
    records: list[dict[str, object]] = []
    ages = [25, 25, 30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70]
    sexes = ["M", "F"] * 10
    regions = ["Utah, Western USA"] * 10 + ["Arizona, Western USA"] * 10
    for i in range(min(n, 20)):
        rec = _make_clinical_record(
            patient_id=f"PAT-{i:03d}",
            name=f"Patient {i}",
            email=f"pat{i}@example.com",
            age=ages[i],
            sex=sexes[i],
            geographic_region=regions[i],
        )
        records.append(rec)
    return records


# ===========================================================================
# Safe Harbor Processor Tests
# ===========================================================================


class TestSafeHarborProcessor:
    """HIPAA Safe Harbor de-identification tests."""

    def test_removes_direct_identifiers(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        result = proc.process_record(record)
        assert "name" not in result
        assert "email" not in result
        assert "ssn" not in result
        assert "phone" not in result
        assert "date_of_birth" not in result

    def test_preserves_non_identifier_fields(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        result = proc.process_record(record)
        assert "diagnosis" in result
        assert "csf_glucose" in result
        assert "csf_protein" in result

    def test_age_cap_at_89(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(age=95)
        result = proc.process_record(record)
        assert result["age"] == 89

    def test_age_below_cap_unchanged(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(age=45)
        result = proc.process_record(record)
        assert result["age"] == 45

    def test_custom_age_cap(self) -> None:
        config = SafeHarborConfig(age_cap=80)
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(age=85)
        result = proc.process_record(record)
        assert result["age"] == 80

    def test_zip_truncation_to_3_digits(self) -> None:
        proc = SafeHarborProcessor()
        # Use "postal_code" - "zip_code" is a direct identifier that gets
        # removed before geographic truncation runs.  The truncation step
        # matches any surviving key containing "zip" or "postal".
        record = _make_clinical_record(postal_code="84408")
        result = proc.process_record(record)
        assert result["postal_code"] == "844"

    def test_zip_small_population_prefix(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(postal_code="03601")
        result = proc.process_record(record)
        assert result["postal_code"] == "000"

    def test_zip_short_code(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(postal_code="84")
        result = proc.process_record(record)
        assert result["postal_code"] == "000"

    def test_date_generalization_to_year(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025"

    def test_date_generalization_to_month(self) -> None:
        config = SafeHarborConfig(date_precision="month")
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025-09"

    def test_date_generalization_to_day(self) -> None:
        config = SafeHarborConfig(date_precision="day")
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025-09-10"

    def test_date_none_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date=None)
        result = proc.process_record(record)
        assert result["collection_date"] is None

    def test_date_invalid_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date="not-a-date")
        result = proc.process_record(record)
        assert result["collection_date"] is None

    def test_date_datetime_object(self) -> None:
        proc = SafeHarborProcessor()
        dt = datetime(2025, 9, 10, 14, 30, tzinfo=timezone.utc)
        record = _make_clinical_record(collection_date=dt)
        result = proc.process_record(record)
        assert result["collection_date"] == "2025"

    def test_free_text_scrubs_phone(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Contact patient at 801-555-1234 for follow-up"}
        result = proc.process_record(record)
        assert "801-555-1234" not in result["notes"]
        assert "[REDACTED]" in result["notes"]

    def test_free_text_scrubs_ssn(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "SSN is 123-45-6789 as confirmed"}
        result = proc.process_record(record)
        assert "123-45-6789" not in result["notes"]

    def test_free_text_scrubs_email_in_text(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Send results to user@hospital.org tomorrow"}
        result = proc.process_record(record)
        assert "user@hospital.org" not in result["notes"]

    def test_free_text_scrubs_dates(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Patient born on 06/15/1985 confirmed exposure"}
        result = proc.process_record(record)
        assert "06/15/1985" not in result["notes"]

    def test_actions_log_populated(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        proc.process_record(record)
        actions = proc.actions
        assert len(actions) > 0
        removal_actions = [
            a for a in actions if a.method == DeidentificationMethod.REMOVAL
        ]
        assert len(removal_actions) >= 4  # name, email, ssn, phone, etc.

    def test_process_batch(self) -> None:
        proc = SafeHarborProcessor()
        batch = _make_record_batch(5)
        results = proc.process_batch(batch)
        assert len(results) == 5
        for r in results:
            assert "name" not in r

    def test_pseudonymize(self) -> None:
        proc = SafeHarborProcessor()
        pseudo = proc._pseudonymize("John Doe")
        assert pseudo.startswith("PSEUDO_")
        assert len(pseudo) == 19  # PSEUDO_ + 12 hex chars

    def test_pseudonymize_deterministic(self) -> None:
        config = SafeHarborConfig(salt=b"fixed-salt-for-test")
        proc = SafeHarborProcessor(config)
        p1 = proc._pseudonymize("John Doe")
        p2 = proc._pseudonymize("John Doe")
        assert p1 == p2

    def test_date_non_string_non_datetime_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        result = proc._generalise_date(12345)
        assert result is None


# ===========================================================================
# k-Anonymity Tests
# ===========================================================================


class TestKAnonymityProcessor:
    """k-Anonymity enforcement tests."""

    def test_k_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            KAnonymityConfig(k=1)

    def test_enforce_returns_k_anonymous_dataset(self) -> None:
        config = KAnonymityConfig(
            k=2,
            quasi_identifiers=("age", "sex", "geographic_region"),
        )
        proc = KAnonymityProcessor(config)
        records = _make_record_batch(20)
        result = proc.enforce(records)
        # Every equivalence class should have >= k members
        classes = proc._compute_equivalence_classes(result)
        for key, count in classes.items():
            assert count >= 2, f"Equivalence class {key} has only {count} members"

    def test_suppressed_count_tracked(self) -> None:
        config = KAnonymityConfig(
            k=10,
            quasi_identifiers=("age", "sex"),
        )
        proc = KAnonymityProcessor(config)
        records = _make_record_batch(20)
        result = proc.enforce(records)
        # Some records may be suppressed
        assert proc.suppressed_count >= 0
        assert len(result) + proc.suppressed_count <= 20

    def test_information_loss_bounded(self) -> None:
        config = KAnonymityConfig(k=2)
        proc = KAnonymityProcessor(config)
        original = _make_record_batch(20)
        anonymised = proc.enforce(original)
        loss = proc.get_information_loss(original, anonymised)
        assert 0.0 <= loss <= 1.0

    def test_information_loss_empty_dataset(self) -> None:
        proc = KAnonymityProcessor()
        assert proc.get_information_loss([], []) == 0.0

    def test_generalisation_applies(self) -> None:
        config = KAnonymityConfig(
            k=5,
            quasi_identifiers=("age",),
            generalisation_hierarchies={
                "age": [
                    lambda v: (v // 10) * 10,
                    lambda _: "*",
                ],
            },
        )
        proc = KAnonymityProcessor(config)
        records = [{"age": 25 + i} for i in range(10)]
        result = proc.enforce(records)
        for r in result:
            if r["age"] != "*":
                assert isinstance(r["age"], int)
                assert r["age"] % 10 == 0

    def test_suppress_violations(self) -> None:
        config = KAnonymityConfig(
            k=5,
            quasi_identifiers=("age",),
            generalisation_hierarchies={"age": []},
        )
        proc = KAnonymityProcessor(config)
        # Only 1 record per age -> all must be suppressed
        records = [{"age": i} for i in range(5)]
        result = proc.enforce(records)
        assert len(result) == 0
        assert proc.suppressed_count == 5

    def test_already_k_anonymous(self) -> None:
        config = KAnonymityConfig(k=2, quasi_identifiers=("age",))
        proc = KAnonymityProcessor(config)
        records = [{"age": 30}] * 10  # all same QI
        result = proc.enforce(records)
        assert len(result) == 10

    def test_generalisation_error_handling(self) -> None:
        def _bad_generaliser(_: object) -> str:
            raise TypeError("deliberate generalisation failure")

        config = KAnonymityConfig(
            k=2,
            quasi_identifiers=("age",),
            generalisation_hierarchies={
                "age": [_bad_generaliser],
            },
        )
        proc = KAnonymityProcessor(config)
        # Unique ages -> each equivalence class has size 1, which is < k=2.
        # The generaliser raises TypeError (caught by the handler),
        # collapsing every value to "*", which makes a single equivalence
        # class of size 5 >= k=2.
        records = [{"age": i} for i in range(5)]
        result = proc.enforce(records)
        assert all(r["age"] == "*" for r in result)


# ===========================================================================
# Differential Privacy Mechanism Tests
# ===========================================================================


class TestLaplaceMechanism:
    """Laplace noise mechanism tests."""

    def test_invalid_sensitivity_raises(self) -> None:
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            LaplaceMechanism(sensitivity=0, epsilon=1.0)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            LaplaceMechanism(sensitivity=1.0, epsilon=0)

    def test_scale_computation(self) -> None:
        mech = LaplaceMechanism(sensitivity=2.0, epsilon=0.5)
        assert mech.scale == pytest.approx(4.0)

    def test_add_noise_changes_value(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        noised = mech.add_noise(100.0)
        assert noised != 100.0

    def test_add_noise_mean_converges(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        samples = [mech.add_noise(100.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean - 100.0) < 0.5  # should be close to true value

    def test_add_noise_batch(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        values = np.array([10.0, 20.0, 30.0])
        noised = mech.add_noise_batch(values)
        assert noised.shape == (3,)
        assert not np.array_equal(values, noised)

    def test_confidence_interval(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        lower, upper = mech.confidence_interval(100.0, confidence=0.95)
        assert lower < 100.0
        assert upper > 100.0
        assert upper - lower > 0

    def test_confidence_interval_wider_at_lower_epsilon(self) -> None:
        mech_tight = LaplaceMechanism(sensitivity=1.0, epsilon=2.0)
        mech_wide = LaplaceMechanism(sensitivity=1.0, epsilon=0.1)
        ci_tight = mech_tight.confidence_interval(0.0, 0.95)
        ci_wide = mech_wide.confidence_interval(0.0, 0.95)
        width_tight = ci_tight[1] - ci_tight[0]
        width_wide = ci_wide[1] - ci_wide[0]
        assert width_wide > width_tight


class TestGaussianMechanism:
    """Gaussian noise mechanism tests."""

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=0, epsilon=1.0)
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=0)
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=0)

    def test_sigma_computation(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert mech.sigma > 0

    def test_add_noise(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        noised = mech.add_noise(50.0)
        assert noised != 50.0

    def test_add_noise_batch(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        values = np.array([10.0, 20.0, 30.0])
        noised = mech.add_noise_batch(values)
        assert noised.shape == (3,)

    def test_noise_mean_converges(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        samples = [mech.add_noise(0.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean) < 0.5


class TestExponentialMechanism:
    """Exponential mechanism for categorical selection tests."""

    def test_select_from_candidates(self) -> None:
        mech = ExponentialMechanism(epsilon=1.0, seed=42)
        candidates = ["cat_A", "cat_B", "cat_C"]
        utilities = [10.0, 1.0, 1.0]
        selected = mech.select(candidates, utilities)
        assert selected in candidates

    def test_high_utility_selected_more_often(self) -> None:
        mech = ExponentialMechanism(epsilon=5.0, seed=42)
        candidates = ["best", "worst"]
        utilities = [100.0, 0.0]
        counts = {"best": 0, "worst": 0}
        for _ in range(1000):
            s = mech.select(candidates, utilities)
            counts[s] += 1
        assert counts["best"] > counts["worst"]

    def test_handles_zero_weights(self) -> None:
        mech = ExponentialMechanism(epsilon=1.0, seed=42)
        candidates = ["a", "b"]
        utilities = [-1e10, -1e10]  # very low -> near-zero weights
        selected = mech.select(candidates, utilities)
        assert selected in candidates


# ===========================================================================
# Privacy Budget Tests
# ===========================================================================


class TestPrivacyBudget:
    """Privacy budget tracking tests."""

    def test_initial_state(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        assert pb.remaining_epsilon == pytest.approx(1.0)
        assert pb.spent_epsilon == 0.0

    def test_allocate_succeeds(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        assert pb.allocate("age", 0.3) is True
        assert pb.remaining_epsilon == pytest.approx(0.7)

    def test_allocate_exceeds_budget(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        pb.allocate("age", 0.8)
        assert pb.allocate("csf_wbc", 0.5) is False

    def test_reset(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        pb.allocate("age", 0.5)
        pb.reset()
        assert pb.remaining_epsilon == pytest.approx(1.0)
        assert pb.field_budgets == {}

    def test_remaining_epsilon_never_negative(self) -> None:
        pb = PrivacyBudget(total_epsilon=0.1)
        pb.allocate("age", 0.1)
        assert pb.remaining_epsilon >= 0.0


# ===========================================================================
# De-identification Pipeline Tests
# ===========================================================================


class TestDeidentificationPipeline:
    """Full pipeline orchestration tests."""

    def test_safe_harbor_only(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.SAFE_HARBOR_ONLY,
        )
        records = _make_record_batch(5)
        result = pipeline.process(records)
        assert len(result) == 5
        for r in result:
            assert "name" not in r
        report = pipeline.report
        assert report.privacy_level == "safe_harbor"
        assert report.safe_harbor_actions > 0

    def test_k_anonymous(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.K_ANONYMOUS,
            k=2,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert len(result) <= 20
        report = pipeline.report
        assert report.privacy_level == "k_anonymous"

    def test_full_pipeline(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k=2,
            total_epsilon=1.0,
            seed=42,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert len(result) <= 20
        report = pipeline.report
        assert report.privacy_level == "full_pipeline"
        assert report.epsilon_spent > 0

    def test_report_to_dict(self) -> None:
        pipeline = create_deidentification_pipeline(seed=42)
        records = _make_record_batch(10)
        pipeline.process(records)
        d = pipeline.report.to_dict()
        assert "input_count" in d
        assert d["input_count"] == 10

    def test_pipeline_config_defaults(self) -> None:
        config = DeidentificationConfig()
        assert config.privacy_level == PrivacyLevel.FULL_PIPELINE
        assert config.k_anonymity.k == 5

    def test_report_timestamp(self) -> None:
        pipeline = create_deidentification_pipeline(seed=42)
        pipeline.process(_make_record_batch(5))
        assert pipeline.report.timestamp != ""

    def test_differentially_private_level(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.DIFFERENTIALLY_PRIVATE,
            k=2,
            total_epsilon=2.0,
            seed=42,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert pipeline.report.epsilon_spent > 0
        assert len(result) > 0

    def test_pipeline_handles_non_numeric_gracefully(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k=2,
            seed=42,
        )
        records = [{"age": "unknown", "sex": "M", "geographic_region": "Utah"}] * 10
        result = pipeline.process(records)
        assert len(result) > 0


# ===========================================================================
# Coverage Gap Tests - previously uncovered lines
# ===========================================================================


class TestCoverageGaps:
    """Tests targeting previously uncovered code paths."""

    def test_k_anon_suppress_violations_count(self) -> None:
        """Line 492: suppressed_count incremented per violating record."""
        config = KAnonymityConfig(
            k=5,
            quasi_identifiers=("age",),
            generalisation_hierarchies={"age": []},
        )
        proc = KAnonymityProcessor(config)
        # 3 records per age group, k=5 -> all suppressed
        records = (
            [{"age": 10}] * 3
            + [{"age": 20}] * 3
        )
        result = proc.enforce(records)
        assert len(result) == 0
        assert proc.suppressed_count == 6

    def test_pipeline_budget_allocation_break(self) -> None:
        """Line 950: budget.allocate returns False -> break out of loop."""
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k=2,
            total_epsilon=0.0001,  # very small budget to exhaust quickly
            seed=42,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        # Should still produce output even with exhausted budget
        assert isinstance(result, list)


# ===========================================================================
# l-Diversity Tests
# ===========================================================================


class TestLDiversityConfig:
    """LDiversityConfig validation tests."""

    def test_default_config(self) -> None:
        from ml.data.deidentification import LDiversityConfig
        config = LDiversityConfig()
        assert config.min_l == 3
        assert "diagnosis" in config.sensitive_attributes

    def test_l_less_than_2_raises(self) -> None:
        from ml.data.deidentification import LDiversityConfig
        with pytest.raises(ValueError, match="min_l must be >= 2"):
            LDiversityConfig(min_l=1)


class TestLDiversityProcessor:
    """LDiversityProcessor enforcement tests."""

    def test_check_diverse_dataset(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=2,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age", "sex"),
        )
        proc = LDiversityProcessor(config)
        records = [
            {"age": 30, "sex": "M", "diagnosis": "suspected_pam"},
            {"age": 30, "sex": "M", "diagnosis": "healthy"},
            {"age": 30, "sex": "M", "diagnosis": "other_infection"},
        ]
        assert proc.check(records) is True

    def test_check_non_diverse_dataset(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=3,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age",),
        )
        proc = LDiversityProcessor(config)
        # All same diagnosis in one QI group
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "pam"},
        ]
        assert proc.check(records) is False

    def test_enforce_suppresses_non_diverse_classes(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=2,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age",),
        )
        proc = LDiversityProcessor(config)
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "pam"},  # only 1 distinct -> violates l=2
            {"age": 40, "diagnosis": "healthy"},
            {"age": 40, "diagnosis": "pam"},  # 2 distinct -> OK
        ]
        result = proc.enforce(records)
        assert len(result) == 2  # only age=40 group survives
        assert proc.suppressed_count == 2

    def test_enforce_preserves_diverse_classes(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=2,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age",),
        )
        proc = LDiversityProcessor(config)
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "healthy"},
            {"age": 40, "diagnosis": "other"},
            {"age": 40, "diagnosis": "pam"},
        ]
        result = proc.enforce(records)
        assert len(result) == 4

    def test_suppressed_asterisk_ignored(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=2,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age",),
        )
        proc = LDiversityProcessor(config)
        records = [
            {"age": 30, "diagnosis": "*"},
            {"age": 30, "diagnosis": "*"},
        ]
        assert proc.check(records) is False

    def test_none_values_ignored(self) -> None:
        from ml.data.deidentification import LDiversityConfig, LDiversityProcessor
        config = LDiversityConfig(
            min_l=2,
            sensitive_attributes=("diagnosis",),
            quasi_identifiers=("age",),
        )
        proc = LDiversityProcessor(config)
        records: list[dict[str, Any]] = [
            {"age": 30, "diagnosis": None},
            {"age": 30, "diagnosis": "pam"},
        ]
        assert proc.check(records) is False  # only 1 non-null distinct

    def test_default_constructor(self) -> None:
        from ml.data.deidentification import LDiversityProcessor
        proc = LDiversityProcessor()
        assert proc.suppressed_count == 0


# ===========================================================================
# t-Closeness Tests
# ===========================================================================


class TestEarthMoversDistance:
    """earth_movers_distance_categorical utility tests."""

    def test_identical_distributions(self) -> None:
        from ml.data.deidentification import earth_movers_distance_categorical
        d = {"a": 0.5, "b": 0.5}
        assert earth_movers_distance_categorical(d, d) == pytest.approx(0.0)

    def test_disjoint_distributions(self) -> None:
        from ml.data.deidentification import earth_movers_distance_categorical
        g = {"a": 1.0}
        gl = {"b": 1.0}
        assert earth_movers_distance_categorical(g, gl) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        from ml.data.deidentification import earth_movers_distance_categorical
        g = {"a": 0.6, "b": 0.4}
        gl = {"a": 0.3, "b": 0.7}
        emd = earth_movers_distance_categorical(g, gl)
        assert 0.0 < emd < 1.0


class TestTCloseness:
    """check_t_closeness verification tests."""

    def test_empty_dataset_passes(self) -> None:
        from ml.data.deidentification import check_t_closeness
        assert check_t_closeness([], ("age",), "diagnosis", 0.5) is True

    def test_uniform_distribution_passes(self) -> None:
        from ml.data.deidentification import check_t_closeness
        records = [
            {"age": 30, "diagnosis": "A"},
            {"age": 30, "diagnosis": "B"},
            {"age": 40, "diagnosis": "A"},
            {"age": 40, "diagnosis": "B"},
        ]
        assert check_t_closeness(records, ("age",), "diagnosis", 1.0) is True

    def test_skewed_distribution_fails(self) -> None:
        from ml.data.deidentification import check_t_closeness
        # Group age=30 is 100% "A", but global is 50/50
        records = [
            {"age": 30, "diagnosis": "A"},
            {"age": 30, "diagnosis": "A"},
            {"age": 40, "diagnosis": "B"},
            {"age": 40, "diagnosis": "B"},
        ]
        # Global: 50% A, 50% B. Group age=30: 100% A -> EMD = 0.5
        # With t=0.3, this should fail
        assert check_t_closeness(records, ("age",), "diagnosis", 0.3) is False

    def test_large_t_passes(self) -> None:
        from ml.data.deidentification import check_t_closeness
        records = [
            {"age": 30, "diagnosis": "A"},
            {"age": 30, "diagnosis": "A"},
            {"age": 40, "diagnosis": "B"},
            {"age": 40, "diagnosis": "B"},
        ]
        assert check_t_closeness(records, ("age",), "diagnosis", 1.0) is True


# ===========================================================================
# Truncated Laplace Mechanism Tests
# ===========================================================================


class TestTruncatedLaplaceMechanism:
    """TruncatedLaplaceMechanism bounded-output tests."""

    def test_output_within_bounds(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        mech = TruncatedLaplaceMechanism(
            sensitivity=1.0, epsilon=0.5, lower=0.0, upper=120.0, seed=42,
        )
        for _ in range(100):
            v = mech.add_noise(60.0)
            assert 0.0 <= v <= 120.0

    def test_invalid_sensitivity_raises(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        with pytest.raises(ValueError, match="positive"):
            TruncatedLaplaceMechanism(sensitivity=0, epsilon=1.0, lower=0.0, upper=100.0)

    def test_invalid_epsilon_raises(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        with pytest.raises(ValueError, match="positive"):
            TruncatedLaplaceMechanism(sensitivity=1.0, epsilon=0, lower=0.0, upper=100.0)

    def test_invalid_bounds_raises(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        with pytest.raises(ValueError, match="Lower bound"):
            TruncatedLaplaceMechanism(sensitivity=1.0, epsilon=1.0, lower=100.0, upper=100.0)

    def test_scale_property(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        mech = TruncatedLaplaceMechanism(
            sensitivity=2.0, epsilon=0.5, lower=0.0, upper=100.0,
        )
        assert mech.scale == pytest.approx(4.0)

    def test_batch_within_bounds(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        mech = TruncatedLaplaceMechanism(
            sensitivity=1.0, epsilon=1.0, lower=10.0, upper=50.0, seed=42,
        )
        values = np.array([20.0, 30.0, 40.0])
        result = mech.add_noise_batch(values)
        assert result.shape == (3,)
        assert np.all(result >= 10.0)
        assert np.all(result <= 50.0)

    def test_noise_alters_value(self) -> None:
        from ml.data.deidentification import TruncatedLaplaceMechanism
        mech = TruncatedLaplaceMechanism(
            sensitivity=1.0, epsilon=1.0, lower=0.0, upper=200.0, seed=42,
        )
        v = mech.add_noise(100.0)
        assert v != 100.0  # extremely unlikely to be exact


# ===========================================================================
# Rényi DP Accountant Tests
# ===========================================================================


class TestRenyiDPAccountant:
    """RenyiDPAccountant composition tracking tests."""

    def test_initial_state(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        assert acc.mechanisms_applied == 0
        assert all(e == 0.0 for e in acc.rdp_epsilons)

    def test_add_laplace(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_laplace(sensitivity=1.0, epsilon=1.0)
        assert acc.mechanisms_applied == 1
        assert any(e > 0 for e in acc.rdp_epsilons)

    def test_add_gaussian(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_gaussian(sensitivity=1.0, sigma=1.0)
        assert acc.mechanisms_applied == 1
        assert any(e > 0 for e in acc.rdp_epsilons)

    def test_composition_increases_epsilon(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_laplace(1.0, 1.0)
        eps1 = acc.get_epsilon(delta=1e-5)
        acc.add_laplace(1.0, 1.0)
        eps2 = acc.get_epsilon(delta=1e-5)
        assert eps2 > eps1

    def test_get_epsilon_positive(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_gaussian(1.0, 1.0)
        eps = acc.get_epsilon(delta=1e-5)
        assert eps > 0.0
        assert eps < float("inf")

    def test_get_epsilon_zero_delta_returns_inf(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_laplace(1.0, 1.0)
        assert acc.get_epsilon(delta=0) == float("inf")

    def test_reset_clears_state(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_laplace(1.0, 1.0)
        acc.add_gaussian(1.0, 1.0)
        acc.reset()
        assert acc.mechanisms_applied == 0
        assert all(e == 0.0 for e in acc.rdp_epsilons)

    def test_more_noise_yields_lower_epsilon(self) -> None:
        """More noise (higher sigma) -> tighter privacy -> lower epsilon."""
        from ml.data.deidentification import RenyiDPAccountant
        acc_noisy = RenyiDPAccountant()
        acc_noisy.add_gaussian(sensitivity=1.0, sigma=10.0)
        acc_tight = RenyiDPAccountant()
        acc_tight.add_gaussian(sensitivity=1.0, sigma=0.1)
        assert acc_noisy.get_epsilon(1e-5) < acc_tight.get_epsilon(1e-5)

    def test_multiple_mechanisms_mixed(self) -> None:
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant()
        acc.add_laplace(1.0, 0.5)
        acc.add_gaussian(1.0, 2.0)
        acc.add_laplace(2.0, 1.0)
        assert acc.mechanisms_applied == 3
        eps = acc.get_epsilon(delta=1e-5)
        assert eps > 0.0

    def test_alpha_one_branch_in_add_laplace(self) -> None:
        """Line 1361: alpha == 1 branch in add_laplace."""
        from ml.data.deidentification import RenyiDPAccountant
        # Include alpha=1 in the orders
        acc = RenyiDPAccountant(alpha_orders=(1.0, 2.0, 4.0))
        acc.add_laplace(sensitivity=1.0, epsilon=1.0)
        assert acc.mechanisms_applied == 1
        # Alpha=1 should still produce a valid RDP epsilon
        assert acc.rdp_epsilons[0] > 0.0

    def test_get_epsilon_skips_alpha_le_1(self) -> None:
        """Line 1414: alpha <= 1 is skipped in get_epsilon."""
        from ml.data.deidentification import RenyiDPAccountant
        acc = RenyiDPAccountant(alpha_orders=(0.5, 1.0, 2.0, 4.0))
        acc.add_gaussian(sensitivity=1.0, sigma=1.0)
        eps = acc.get_epsilon(delta=1e-5)
        assert eps > 0.0
        assert eps < float("inf")


# ===========================================================================
# Final k-Anonymity Suppression Coverage
# ===========================================================================


class TestFinalCoverage:
    """Final tests for remaining uncovered lines."""

    def test_k_anon_suppress_keeps_valid_groups(self) -> None:
        """Line 502: Non-violating records are kept via result.append(record)."""
        config = KAnonymityConfig(
            k=3,
            quasi_identifiers=("age",),
            generalisation_hierarchies={"age": []},
        )
        proc = KAnonymityProcessor(config)
        # 4 records with age=30 (meets k=3) and 2 with age=99 (violates k=3)
        records = [
            {"age": 30} for _ in range(4)
        ] + [
            {"age": 99}, {"age": 99},
        ]
        result = proc.enforce(records)
        # The 4 age=30 records survive (hit line 502), the 2 age=99 are suppressed
        assert len(result) == 4
        assert proc.suppressed_count == 2
        assert all(r["age"] == 30 for r in result)

    def test_pipeline_budget_break_with_many_fields(self) -> None:
        """Line 960: break when budget.allocate returns False."""
        config = DeidentificationConfig(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k_anonymity=KAnonymityConfig(k=2),
            privacy_budget=PrivacyBudget(total_epsilon=0.0001, delta=1e-5),
            numeric_sensitivity={
                "csf_glucose": 50.0,
                "csf_protein": 100.0,
                "csf_wbc": 200.0,
                "age": 1.0,
                "extra_field_1": 1.0,
                "extra_field_2": 1.0,
            },
            seed=42,
        )
        from ml.data.deidentification import DeidentificationPipeline
        pipeline = DeidentificationPipeline(config)
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert isinstance(result, list)


# ===========================================================================
# Entropy l-Diversity Tests
# ===========================================================================


class TestEntropyLDiversityChecker:
    """Entropy-based l-diversity verification tests."""

    def test_min_l_validation(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        with pytest.raises(ValueError, match="min_l must be >= 2"):
            EntropyLDiversityChecker(min_l=1)

    def test_min_entropy_property(self) -> None:
        import math
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(min_l=3)
        assert abs(checker.min_entropy - math.log(3)) < 1e-10

    def test_uniform_distribution_passes(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=2,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "healthy"},
            {"age": 30, "diagnosis": "other"},
        ]
        assert checker.check(records) is True

    def test_skewed_distribution_fails(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=3,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        # 98 pam, 1 healthy, 1 other -> entropy < log(3)
        records = [{"age": 30, "diagnosis": "pam"}] * 98
        records.append({"age": 30, "diagnosis": "healthy"})
        records.append({"age": 30, "diagnosis": "other"})
        assert checker.check(records) is False

    def test_single_value_fails(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=2,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "pam"},
        ]
        assert checker.check(records) is False

    def test_class_entropies(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=2,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "healthy"},
            {"age": 40, "diagnosis": "pam"},
            {"age": 40, "diagnosis": "pam"},
        ]
        entropies = checker.class_entropies(records)
        assert len(entropies) == 2
        # age=30 has 2 distinct -> entropy = log(2)
        assert entropies[(30,)] > 0.5
        # age=40 has 1 distinct -> entropy = 0
        assert entropies[(40,)] == 0.0

    def test_nulls_and_stars_ignored(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=2,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        records: list[dict[str, Any]] = [
            {"age": 30, "diagnosis": None},
            {"age": 30, "diagnosis": "*"},
            {"age": 30, "diagnosis": "pam"},
        ]
        # Only 1 non-null non-star -> entropy 0 < log(2)
        assert checker.check(records) is False

    def test_empty_dataset(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(min_l=2)
        assert checker.check([]) is True

    def test_multiple_groups_mixed(self) -> None:
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            min_l=2,
            sensitive_attribute="diagnosis",
            quasi_identifiers=("age",),
        )
        records = [
            {"age": 30, "diagnosis": "pam"},
            {"age": 30, "diagnosis": "healthy"},
            {"age": 40, "diagnosis": "other"},
            {"age": 40, "diagnosis": "other"},
        ]
        # age=30 passes, age=40 fails
        assert checker.check(records) is False


# ===========================================================================
# Re-identification Risk Estimator Tests
# ===========================================================================


class TestReidentificationRiskEstimator:
    """Re-identification risk estimation tests."""

    def test_empty_dataset(self) -> None:
        from ml.data.deidentification import ReidentificationRiskEstimator
        estimator = ReidentificationRiskEstimator()
        report = estimator.estimate([])
        assert report.records_analysed == 0
        assert report.prosecutor_risk == 0.0
        assert report.journalist_risk == 0.0
        assert report.marketer_risk == 0.0

    def test_all_identical_records(self) -> None:
        from ml.data.deidentification import ReidentificationRiskEstimator
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age", "sex"),
        )
        records = [{"age": 30, "sex": "M"}] * 10
        report = estimator.estimate(records)
        assert report.prosecutor_risk == 0.1
        assert report.journalist_risk == 0.1
        assert report.marketer_risk == 0.0
        assert report.equivalence_classes == 1
        assert report.smallest_class_size == 10

    def test_all_unique_records(self) -> None:
        from ml.data.deidentification import ReidentificationRiskEstimator
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age",),
        )
        records = [{"age": i} for i in range(5)]
        report = estimator.estimate(records)
        assert report.prosecutor_risk == 1.0
        assert report.marketer_risk == 1.0
        assert report.equivalence_classes == 5
        assert report.smallest_class_size == 1

    def test_mixed_classes(self) -> None:
        from ml.data.deidentification import ReidentificationRiskEstimator
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age",),
        )
        records = [
            {"age": 30}, {"age": 30}, {"age": 30},  # class of 3
            {"age": 40},                              # singleton
        ]
        report = estimator.estimate(records)
        assert report.prosecutor_risk == 1.0  # singleton
        assert report.marketer_risk == 0.25   # 1 singleton / 4 total
        assert report.equivalence_classes == 2
        assert report.smallest_class_size == 1

    def test_report_fields(self) -> None:
        from ml.data.deidentification import ReidentificationRiskEstimator
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age", "sex"),
        )
        records = [
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
        ]
        report = estimator.estimate(records)
        assert report.records_analysed == 4
        assert report.equivalence_classes == 2
        assert report.smallest_class_size == 2
        assert report.prosecutor_risk == 0.5
        assert report.marketer_risk == 0.0

    def test_class_entropy_all_null_star(self) -> None:
        """Cover _class_entropy return 0.0 when all values are None or star."""
        from ml.data.deidentification import EntropyLDiversityChecker
        checker = EntropyLDiversityChecker(
            quasi_identifiers=("zip",),
            sensitive_attribute="diagnosis",
            min_l=2,
        )
        records: list[dict[str, Any]] = [
            {"zip": "100", "diagnosis": None},
            {"zip": "100", "diagnosis": "*"},
            {"zip": "100", "diagnosis": None},
        ]
        entropies = checker.class_entropies(records)
        assert len(entropies) == 1
        # Dict keyed by QI tuple
        assert entropies[("100",)] == 0.0


# ===========================================================================
# OverallPrivacyRisk Enum Tests
# ===========================================================================


class TestOverallPrivacyRisk:
    """Verify privacy risk classification enum values."""

    def test_negligible_value(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert OverallPrivacyRisk.NEGLIGIBLE.value == "negligible"

    def test_low_value(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert OverallPrivacyRisk.LOW.value == "low"

    def test_moderate_value(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert OverallPrivacyRisk.MODERATE.value == "moderate"

    def test_high_value(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert OverallPrivacyRisk.HIGH.value == "high"

    def test_very_high_value(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert OverallPrivacyRisk.VERY_HIGH.value == "very_high"

    def test_member_count(self) -> None:
        from ml.data.deidentification import OverallPrivacyRisk
        assert len(OverallPrivacyRisk) == 5


# ===========================================================================
# SyntheticUtilityMetric Enum Tests
# ===========================================================================


class TestSyntheticUtilityMetric:
    """Verify synthetic utility metric enum values."""

    def test_jensen_shannon_value(self) -> None:
        from ml.data.deidentification import SyntheticUtilityMetric
        assert SyntheticUtilityMetric.JENSEN_SHANNON.value == "jensen_shannon_divergence"

    def test_correlation_preservation_value(self) -> None:
        from ml.data.deidentification import SyntheticUtilityMetric
        assert SyntheticUtilityMetric.CORRELATION_PRESERVATION.value == "correlation_preservation"

    def test_membership_inference_value(self) -> None:
        from ml.data.deidentification import SyntheticUtilityMetric
        assert SyntheticUtilityMetric.MEMBERSHIP_INFERENCE.value == "membership_inference_proxy"

    def test_member_count(self) -> None:
        from ml.data.deidentification import SyntheticUtilityMetric
        assert len(SyntheticUtilityMetric) == 3


# ===========================================================================
# PrivacyRiskScorecard Dataclass Tests
# ===========================================================================


class TestPrivacyRiskScorecard:
    """Verify frozen dataclass construction and defaults."""

    def test_default_values(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            PrivacyRiskScorecard,
        )
        sc = PrivacyRiskScorecard()
        assert sc.k_anonymity_level == 0
        assert sc.l_diversity_level == 0
        assert sc.epsilon_consumed == 0.0
        assert sc.epsilon_total == 1.0
        assert sc.prosecutor_risk == 0.0
        assert sc.journalist_risk == 0.0
        assert sc.marketer_risk == 0.0
        assert sc.records_evaluated == 0
        assert sc.overall_risk == OverallPrivacyRisk.HIGH

    def test_frozen_immutability(self) -> None:
        from ml.data.deidentification import PrivacyRiskScorecard
        sc = PrivacyRiskScorecard()
        with pytest.raises(AttributeError):
            sc.k_anonymity_level = 99  # type: ignore[misc]


# ===========================================================================
# compute_privacy_scorecard Tests - All 5 Classification Levels
# ===========================================================================


class TestComputePrivacyScorecard:
    """Full coverage of the NIST SP 800-188 classification cascade."""

    def test_negligible_classification(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=15,
            l_level=8,
            epsilon_consumed=0.3,
            epsilon_total=2.0,
        )
        assert sc.overall_risk == OverallPrivacyRisk.NEGLIGIBLE
        assert sc.k_anonymity_level == 15
        assert sc.l_diversity_level == 8
        assert sc.epsilon_consumed == 0.3

    def test_low_classification(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=7,
            l_level=4,
            epsilon_consumed=0.8,
            epsilon_total=2.0,
        )
        assert sc.overall_risk == OverallPrivacyRisk.LOW

    def test_moderate_classification(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=4,
            l_level=2,
            epsilon_consumed=1.5,
            epsilon_total=3.0,
        )
        assert sc.overall_risk == OverallPrivacyRisk.MODERATE

    def test_high_classification_default(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=2,
            l_level=1,
            epsilon_consumed=3.0,
            epsilon_total=5.0,
        )
        assert sc.overall_risk == OverallPrivacyRisk.HIGH

    def test_very_high_low_k(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=1,
            l_level=10,
            epsilon_consumed=0.1,
        )
        assert sc.overall_risk == OverallPrivacyRisk.VERY_HIGH

    def test_very_high_excessive_epsilon(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            compute_privacy_scorecard,
        )
        sc = compute_privacy_scorecard(
            k_level=20,
            l_level=10,
            epsilon_consumed=5.0,
        )
        assert sc.overall_risk == OverallPrivacyRisk.VERY_HIGH

    def test_very_high_prosecutor_risk(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            ReidentificationRiskEstimator,
            compute_privacy_scorecard,
        )
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age",),
        )
        records: list[dict[str, Any]] = [
            {"age": 30},
        ]
        risk_report = estimator.estimate(records)
        sc = compute_privacy_scorecard(
            k_level=20,
            l_level=10,
            epsilon_consumed=0.1,
            risk_report=risk_report,
        )
        assert sc.overall_risk == OverallPrivacyRisk.VERY_HIGH
        assert sc.prosecutor_risk >= 0.5
        assert sc.records_evaluated == 1

    def test_with_risk_report_moderate(self) -> None:
        from ml.data.deidentification import (
            OverallPrivacyRisk,
            ReidentificationRiskEstimator,
            compute_privacy_scorecard,
        )
        estimator = ReidentificationRiskEstimator(
            quasi_identifiers=("age", "sex"),
        )
        records: list[dict[str, Any]] = [
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 30, "sex": "M"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
            {"age": 40, "sex": "F"},
        ]
        risk_report = estimator.estimate(records)
        sc = compute_privacy_scorecard(
            k_level=4,
            l_level=2,
            epsilon_consumed=1.5,
            risk_report=risk_report,
        )
        assert sc.overall_risk == OverallPrivacyRisk.MODERATE
        assert sc.journalist_risk == risk_report.journalist_risk
        assert sc.marketer_risk == risk_report.marketer_risk

    def test_no_risk_report_defaults_zeroes(self) -> None:
        from ml.data.deidentification import compute_privacy_scorecard
        sc = compute_privacy_scorecard(k_level=5, l_level=3, epsilon_consumed=0.5)
        assert sc.prosecutor_risk == 0.0
        assert sc.journalist_risk == 0.0
        assert sc.marketer_risk == 0.0
        assert sc.records_evaluated == 0


# ===========================================================================
# SyntheticEvaluationReport Dataclass Tests
# ===========================================================================


class TestSyntheticEvaluationReport:
    """Verify frozen dataclass defaults."""

    def test_default_values(self) -> None:
        from ml.data.deidentification import SyntheticEvaluationReport
        rpt = SyntheticEvaluationReport()
        assert rpt.column_divergences == {}
        assert rpt.mean_divergence == 0.0
        assert rpt.correlation_score == 0.0
        assert rpt.membership_proxy == 0.0
        assert rpt.overall_utility == 0.0

    def test_frozen_immutability(self) -> None:
        from ml.data.deidentification import SyntheticEvaluationReport
        rpt = SyntheticEvaluationReport()
        with pytest.raises(AttributeError):
            rpt.mean_divergence = 0.5  # type: ignore[misc]


# ===========================================================================
# SyntheticDataEvaluator Tests
# ===========================================================================


class TestSyntheticDataEvaluator:
    """Comprehensive tests for synthetic data evaluation."""

    def test_empty_columns_raises(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        with pytest.raises(ValueError, match="At least one column"):
            SyntheticDataEvaluator(columns=())

    def test_evaluate_identical_datasets(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        data: list[dict[str, Any]] = [
            {"age": "30", "sex": "M"},
            {"age": "40", "sex": "F"},
            {"age": "50", "sex": "M"},
        ]
        ev = SyntheticDataEvaluator(columns=("age", "sex"))
        report = ev.evaluate(data, data)
        assert report.mean_divergence == 0.0
        for jsd in report.column_divergences.values():
            assert jsd == 0.0
        assert report.correlation_score == 1.0
        assert report.membership_proxy == 1.0
        assert report.overall_utility == 1.0

    def test_evaluate_completely_different(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"age": "20", "sex": "M"},
            {"age": "20", "sex": "M"},
            {"age": "20", "sex": "M"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"age": "80", "sex": "F"},
            {"age": "80", "sex": "F"},
            {"age": "80", "sex": "F"},
        ]
        ev = SyntheticDataEvaluator(columns=("age", "sex"))
        report = ev.evaluate(original, synthetic)
        assert report.mean_divergence > 0.0
        assert report.membership_proxy == 0.0

    def test_evaluate_empty_original(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        ev = SyntheticDataEvaluator(columns=("age",))
        report = ev.evaluate([], [{"age": "30"}])
        assert report.mean_divergence == 0.0
        assert report.overall_utility == 0.0

    def test_evaluate_empty_synthetic(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        ev = SyntheticDataEvaluator(columns=("age",))
        report = ev.evaluate([{"age": "30"}], [])
        assert report.mean_divergence == 0.0
        assert report.overall_utility == 0.0

    def test_single_column_correlation_is_one(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"age": "30"},
            {"age": "40"},
            {"age": "50"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"age": "30"},
            {"age": "40"},
            {"age": "50"},
        ]
        ev = SyntheticDataEvaluator(columns=("age",))
        report = ev.evaluate(original, synthetic)
        assert report.correlation_score == 1.0

    def test_partial_overlap_membership(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"age": "30", "sex": "M"},
            {"age": "40", "sex": "F"},
            {"age": "50", "sex": "M"},
            {"age": "60", "sex": "F"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"age": "30", "sex": "M"},
            {"age": "40", "sex": "F"},
            {"age": "70", "sex": "M"},
            {"age": "80", "sex": "F"},
        ]
        ev = SyntheticDataEvaluator(columns=("age", "sex"))
        report = ev.evaluate(original, synthetic)
        assert report.membership_proxy == 0.5

    def test_jsd_symmetry(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"col": "A"}, {"col": "A"}, {"col": "B"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"col": "B"}, {"col": "B"}, {"col": "A"},
        ]
        ev = SyntheticDataEvaluator(columns=("col",))
        report_fwd = ev.evaluate(original, synthetic)
        report_rev = ev.evaluate(synthetic, original)
        assert abs(report_fwd.mean_divergence - report_rev.mean_divergence) < 1e-6

    def test_overall_utility_bounded(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"a": "1", "b": "x"},
            {"a": "2", "b": "y"},
            {"a": "3", "b": "z"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"a": "1", "b": "x"},
            {"a": "4", "b": "w"},
            {"a": "5", "b": "v"},
        ]
        ev = SyntheticDataEvaluator(columns=("a", "b"))
        report = ev.evaluate(original, synthetic)
        assert 0.0 <= report.overall_utility <= 1.0

    def test_column_divergences_keys_match(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"x": "1", "y": "2", "z": "3"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"x": "1", "y": "5", "z": "9"},
        ]
        ev = SyntheticDataEvaluator(columns=("x", "y", "z"))
        report = ev.evaluate(original, synthetic)
        assert set(report.column_divergences.keys()) == {"x", "y", "z"}

    def test_missing_column_treated_as_empty_string(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        original: list[dict[str, Any]] = [
            {"age": "30"},
        ]
        synthetic: list[dict[str, Any]] = [
            {"other": "value"},
        ]
        ev = SyntheticDataEvaluator(columns=("age",))
        report = ev.evaluate(original, synthetic)
        assert report.mean_divergence > 0.0

    def test_jensen_shannon_empty_distribution(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        jsd = SyntheticDataEvaluator._jensen_shannon([], ["a", "b"])
        assert jsd == 1.0
        jsd2 = SyntheticDataEvaluator._jensen_shannon(["a"], [])
        assert jsd2 == 1.0

    def test_correlation_preservation_single_pair_no_diffs(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        ev = SyntheticDataEvaluator(columns=("a", "b"))
        score = ev._correlation_preservation(
            [{"a": "1", "b": "2"}],
            [{"a": "1", "b": "2"}],
        )
        assert score >= 0.0

    def test_membership_proxy_empty_original(self) -> None:
        from ml.data.deidentification import SyntheticDataEvaluator
        result = SyntheticDataEvaluator._membership_proxy(
            [], [{"a": "1"}],
        )
        assert result == 0.0
