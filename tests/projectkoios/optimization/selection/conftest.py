"""Shared pytest fixtures for deterministic selection tests."""

from __future__ import annotations

import pytest

from tests.projectkoios.optimization.selection.fixtures import (
    ResolvedHarmonicCase,
    resolved_harmonic_case,
)


@pytest.fixture
def resolved_harmonic() -> ResolvedHarmonicCase:
    """Return the named analytical 3-4-5 harmonic case."""
    return resolved_harmonic_case()
