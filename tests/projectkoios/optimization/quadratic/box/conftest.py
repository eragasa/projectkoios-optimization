"""Shared pytest fixtures for box-quadratic contract tests."""

from __future__ import annotations

import pytest

from tests.projectkoios.optimization.quadratic.box.fixtures import (
    SatisfiedBoxMinimumCase,
    satisfied_box_minimum_case,
)


@pytest.fixture
def satisfied_minimum_case() -> SatisfiedBoxMinimumCase:
    """Return named analytically KKT-satisfied minimum evidence."""
    return satisfied_box_minimum_case()
