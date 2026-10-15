"""Unit tests for closed-form periodic harmonic maximization."""

from __future__ import annotations

import math
import sys
from typing import cast

import pytest

from projectkoios.optimization.selection.harmonic import (
    DEFAULT_PERIODIC_HARMONIC_ABSOLUTE_DEGENERACY_TOLERANCE,
    DEFAULT_PERIODIC_HARMONIC_RELATIVE_DEGENERACY_TOLERANCE,
    PeriodicHarmonic,
    PeriodicHarmonicMaximizer,
    PeriodicHarmonicMaximumStatus,
)

pytestmark = pytest.mark.unit


def test_default_tolerances_are_explicit() -> None:
    maximizer = PeriodicHarmonicMaximizer()

    assert (
        maximizer.absolute_degeneracy_tolerance
        == DEFAULT_PERIODIC_HARMONIC_ABSOLUTE_DEGENERACY_TOLERANCE
    )
    assert (
        maximizer.relative_degeneracy_tolerance
        == DEFAULT_PERIODIC_HARMONIC_RELATIVE_DEGENERACY_TOLERANCE
    )


@pytest.mark.parametrize("value", (True, "0", 0.0 + 0.0j))
def test_tolerances_reject_nonreal_or_boolean_values(value: object) -> None:
    with pytest.raises(TypeError):
        PeriodicHarmonicMaximizer(absolute_degeneracy_tolerance=cast(float, value))
    with pytest.raises(TypeError):
        PeriodicHarmonicMaximizer(relative_degeneracy_tolerance=cast(float, value))


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0))
def test_tolerances_reject_nonfinite_or_negative_values(value: float) -> None:
    with pytest.raises(ValueError):
        PeriodicHarmonicMaximizer(absolute_degeneracy_tolerance=value)
    with pytest.raises(ValueError):
        PeriodicHarmonicMaximizer(relative_degeneracy_tolerance=value)


def test_both_tolerances_cannot_be_zero() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PeriodicHarmonicMaximizer(
            absolute_degeneracy_tolerance=0.0,
            relative_degeneracy_tolerance=0.0,
        )


def test_rank_two_second_harmonic_returns_continuous_global_maximum() -> None:
    harmonic = PeriodicHarmonic(
        offset=0.25,
        cosine_coefficient=0.6,
        sine_coefficient=0.8,
        harmonic_order=2,
        fundamental_interval_start=-math.pi / 2.0,
    )

    result = PeriodicHarmonicMaximizer().maximize(harmonic)

    assert result.maximum_value == 1.25
    assert result.amplitude == 1.0
    assert result.canonical_maximizing_argument == 0.5 * math.atan2(0.8, 0.6)
    assert result.status is PeriodicHarmonicMaximumStatus.UNIQUE


def test_periodic_boundary_maps_excluded_endpoint_to_included_start() -> None:
    start = -math.pi / 2.0
    result = PeriodicHarmonicMaximizer().maximize(
        PeriodicHarmonic(0.0, -1.0, 0.0, 2, start)
    )

    assert result.canonical_maximizing_argument == start
    assert result.canonical_maximizing_argument != start + math.pi


def test_negative_signed_zero_has_same_canonical_boundary_representative() -> None:
    start = -math.pi / 2.0
    positive_zero = PeriodicHarmonicMaximizer().maximize(
        PeriodicHarmonic(0.0, -1.0, 0.0, 2, start)
    )
    negative_zero = PeriodicHarmonicMaximizer().maximize(
        PeriodicHarmonic(0.0, -1.0, -0.0, 2, start)
    )

    assert positive_zero.canonical_maximizing_argument == start
    assert negative_zero.canonical_maximizing_argument == start


def test_arbitrary_order_and_interval_start_are_canonicalized() -> None:
    start = 0.4
    result = PeriodicHarmonicMaximizer().maximize(
        PeriodicHarmonic(1.0, 0.0, 2.0, 3, start)
    )

    expected = 0.5 * math.pi / 3.0
    assert result.canonical_maximizing_argument == expected
    assert start <= expected < start + math.tau / 3.0


def test_numerically_constant_result_keeps_mathematical_maximum() -> None:
    maximizer = PeriodicHarmonicMaximizer(
        absolute_degeneracy_tolerance=1.0e-6,
        relative_degeneracy_tolerance=0.0,
    )
    result = maximizer.maximize(PeriodicHarmonic(2.0, 3.0e-7, 4.0e-7, 2, 0.0))

    assert result.amplitude == 5.0e-7
    assert result.maximum_value == 2.0000005
    assert result.canonical_maximizing_argument is None
    assert result.status is PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT


def test_relative_degeneracy_uses_declared_problem_scale() -> None:
    result = PeriodicHarmonicMaximizer(
        absolute_degeneracy_tolerance=0.0,
        relative_degeneracy_tolerance=1.0e-10,
    ).maximize(PeriodicHarmonic(1.0e8, 0.005, 0.0, 2, 0.0))

    assert result.applied_degeneracy_cutoff == 0.01
    assert result.status is PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT


def test_result_is_repeatable() -> None:
    harmonic = PeriodicHarmonic(-0.2, -0.4, 0.9, 7, -0.3)
    maximizer = PeriodicHarmonicMaximizer()

    assert maximizer.maximize(harmonic) == maximizer.maximize(harmonic)


def test_nonfinite_derived_arithmetic_fails_closed() -> None:
    maximum_float = sys.float_info.max
    maximizer = PeriodicHarmonicMaximizer()
    with pytest.raises(FloatingPointError, match="amplitude"):
        maximizer.maximize(PeriodicHarmonic(0.0, maximum_float, maximum_float, 1, 0.0))
    with pytest.raises(FloatingPointError, match="maximum"):
        maximizer.maximize(PeriodicHarmonic(maximum_float, maximum_float, 0.0, 1, 0.0))


def test_maximize_requires_exact_harmonic_type() -> None:
    with pytest.raises(TypeError, match="PeriodicHarmonic"):
        PeriodicHarmonicMaximizer().maximize(cast(PeriodicHarmonic, object()))
