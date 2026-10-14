"""Unit tests for Hermitian spectral-subspace tolerances."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceTolerance,
)

pytestmark = pytest.mark.unit


def _values() -> tuple[float, ...]:
    return (1.0e-12, 1.0e-10, 2.0e-12, 2.0e-10, 3.0e-12, 4.0e-12, 4.0e-10)


def test_tolerance_retains_exact_finite_controls_and_is_frozen() -> None:
    tolerance = HermitianSpectralSubspaceTolerance(*_values())

    assert tolerance.hermiticity_absolute == 1.0e-12
    assert tolerance.hermiticity_relative == 1.0e-10
    assert tolerance.boundary_absolute == 2.0e-12
    assert tolerance.boundary_relative == 2.0e-10
    assert tolerance.projector_validation_absolute == 3.0e-12
    assert tolerance.spectral_validation_absolute == 4.0e-12
    assert tolerance.spectral_validation_relative == 4.0e-10
    with pytest.raises(FrozenInstanceError):
        tolerance.boundary_absolute = 1.0  # type: ignore[misc]


def test_tolerance_accepts_zero_for_every_control() -> None:
    tolerance = HermitianSpectralSubspaceTolerance(*(0.0,) * 7)

    assert tolerance == HermitianSpectralSubspaceTolerance(*(0.0,) * 7)


@pytest.mark.parametrize(
    ("index", "value", "error"),
    (
        (0, True, TypeError),
        (0, 1, TypeError),
        (0, -1.0, ValueError),
        (0, float("nan"), ValueError),
        (0, float("inf"), ValueError),
        (1, 1.0, ValueError),
        (3, 1.0, ValueError),
        (6, 1.0, ValueError),
    ),
)
def test_tolerance_rejects_invalid_controls(
    index: int,
    value: object,
    error: type[Exception],
) -> None:
    values: list[object] = list(_values())
    values[index] = value

    with pytest.raises(error):
        HermitianSpectralSubspaceTolerance(
            *cast(tuple[Any, Any, Any, Any, Any, Any, Any], tuple(values))
        )
