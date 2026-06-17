"""Pytest config for Subphase 1.2 PAM vignette generator tests.

Registers the ``subphase_1_2`` marker, ensures the project root is on
``sys.path`` so ``scripts.vignettes.generate_pam_vignettes`` is importable, and
exposes session-scoped fixtures for the 20 generated vignettes so the
generator runs once instead of once per test.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.vignettes.generate_pam_vignettes import (  # noqa: E402
    DAY1_DISTRIBUTION,
    DAY2_DISTRIBUTION,
    PMID_REGISTRY,
    generate_vignette,
    load_pmid_metadata,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "subphase_1_2: tests for the Subphase 1.2 PAM vignette generator",
    )


@pytest.fixture(scope="session")
def distribution() -> list[dict[str, Any]]:
    return DAY1_DISTRIBUTION


@pytest.fixture(scope="session")
def day2_distribution() -> list[dict[str, Any]]:
    return DAY2_DISTRIBUTION


@pytest.fixture(scope="session")
def pmid_registry() -> dict[str, dict[str, Any]]:
    return PMID_REGISTRY


@pytest.fixture(scope="session")
def generated_vignettes() -> list[dict[str, Any]]:
    """Generate all 20 vignettes once per session."""
    out: list[dict[str, Any]] = []
    for spec in DAY1_DISTRIBUTION:
        pmid_meta = load_pmid_metadata(spec["pmid"])
        out.append(generate_vignette(spec, pmid_meta))
    return out
