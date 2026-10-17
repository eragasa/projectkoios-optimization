"""Unit tests for periodic harmonic definitions."""

from __future__ import annotations

import math
import sys

import pytest

from projectkoios.optimization.selection.harmonic import PeriodicHarmonic

pytestmark = pytest.mark.unit


def test_definition_factory_normalizes_real_coefficients_and_exposes_interval() -> None:
    harmonic = PeriodicHarmonic.from_values(
        offset=1,
        cosine_coefficient=2,
        sine_coefficient=-3,
        harmonic_order=2,
        fundamental_interval_start=-math.pi / 2.0,
    )

    assert harmonic.offset == 1.0
    assert harmonic.cosine_coefficient == 2.0
    assert harmonic.sine_coefficient == -3.0
    assert harmonic.harmonic_order == 2
    assert harmonic.fundamental_interval_start == -math.pi / 2.0
    assert harmonic.fundamental_period == math.pi
    assert harmonic.fundamental_interval_stop == math.pi / 2.0


def test_raw_definition_rejects_noncanonical_coefficients() -> None:
    with pytest.raises(TypeError, match="exact built-in float"):
        PeriodicHarmonic(1, 2.0, -3.0, 2, -math.pi / 2.0)


@pytest.mark.parametrize("value", (True, "1", 1.0 + 0.0j))
def test_definition_rejects_nonreal_or_boolean_coefficients(value: object) -> None:
    with pytest.raises(TypeError):
        PeriodicHarmonic(value, 1.0, 0.0, 2, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PeriodicHarmonic(0.0, value, 0.0, 2, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PeriodicHarmonic(0.0, 1.0, value, 2, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_definition_rejects_nonfinite_coefficients_and_start(value: float) -> None:
    with pytest.raises(ValueError):
        PeriodicHarmonic(value, 1.0, 0.0, 2, 0.0)
    with pytest.raises(ValueError):
        PeriodicHarmonic(0.0, value, 0.0, 2, 0.0)
    with pytest.raises(ValueError):
        PeriodicHarmonic(0.0, 1.0, value, 2, 0.0)
    with pytest.raises(ValueError):
        PeriodicHarmonic(0.0, 1.0, 0.0, 2, value)


@pytest.mark.parametrize("value", (True, 1.0, "2"))
def test_definition_requires_exact_integer_harmonic_order(value: object) -> None:
    with pytest.raises(TypeError):
        PeriodicHarmonic(0.0, 1.0, 0.0, value, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (0, -1, 2**32))
def test_definition_requires_positive_u32_harmonic_order(value: int) -> None:
    with pytest.raises(ValueError):
        PeriodicHarmonic(0.0, 1.0, 0.0, value, 0.0)


def test_definition_rejects_unrepresentable_interval_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        PeriodicHarmonic(0.0, 1.0, 0.0, 1, sys.float_info.max)
