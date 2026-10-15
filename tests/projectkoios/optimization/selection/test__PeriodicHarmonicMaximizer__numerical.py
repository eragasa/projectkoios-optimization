"""Bounded numerical checks for periodic harmonic maximization."""

from __future__ import annotations

import math
import random

import pytest

from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximizer,
    PeriodicHarmonicMaximumStatus,
)

pytestmark = pytest.mark.numerical


def _evaluate(harmonic: PeriodicHarmonic, argument: float) -> float:
    phase = float(harmonic.harmonic_order) * argument
    return harmonic.offset + (
        harmonic.cosine_coefficient * math.cos(phase)
        + harmonic.sine_coefficient * math.sin(phase)
    )


def test_seeded_harmonics_match_closed_form_and_dense_grid_bounds() -> None:
    generator = random.Random(20261015)
    maximizer = PeriodicHarmonicMaximizer(
        absolute_degeneracy_tolerance=1.0e-14,
        relative_degeneracy_tolerance=1.0e-14,
    )

    for _ in range(256):
        order = generator.randint(1, 12)
        start = generator.uniform(-4.0, 4.0)
        offset = generator.uniform(-10.0, 10.0)
        amplitude = generator.uniform(0.1, 5.0)
        phase = generator.uniform(-math.pi, math.pi)
        harmonic = PeriodicHarmonic(
            offset,
            amplitude * math.cos(phase),
            amplitude * math.sin(phase),
            order,
            start,
        )
        result = maximizer.maximize(harmonic)
        argument = result.canonical_maximizing_argument

        assert result.status is PeriodicHarmonicMaximumStatus.UNIQUE
        assert argument is not None
        assert start <= argument < harmonic.fundamental_interval_stop
        assert _evaluate(harmonic, argument) == pytest.approx(
            result.maximum_value,
            rel=2.0e-15,
            abs=2.0e-15,
        )

        sample_count = 4096
        step = harmonic.fundamental_period / float(sample_count)
        sampled_maximum = max(
            _evaluate(harmonic, start + index * step) for index in range(sample_count)
        )
        assert sampled_maximum <= result.maximum_value + 2.0e-14
        assert result.maximum_value - sampled_maximum <= (
            0.5 * result.amplitude * (float(order) * step) ** 2 + 2.0e-14
        )


def test_continuous_maximum_bounds_sampled_argument_inventories() -> None:
    harmonic = PeriodicHarmonic(
        offset=0.0,
        cosine_coefficient=0.37,
        sine_coefficient=-0.81,
        harmonic_order=2,
        fundamental_interval_start=-math.pi / 2.0,
    )
    result = PeriodicHarmonicMaximizer().maximize(harmonic)

    assert result.canonical_maximizing_argument is not None
    for count in (17, 31, 63, 127, 255):
        finite_maximum = max(
            _evaluate(
                harmonic,
                harmonic.fundamental_interval_start
                + index * harmonic.fundamental_period / float(count),
            )
            for index in range(count)
        )
        assert finite_maximum <= result.maximum_value


def test_canonical_argument_is_covariant_under_periodic_interval_shift() -> None:
    base = PeriodicHarmonic(0.3, -0.2, 0.7, 5, -0.4)
    shifted = PeriodicHarmonic(
        base.offset,
        base.cosine_coefficient,
        base.sine_coefficient,
        base.harmonic_order,
        base.fundamental_interval_start + 9.0 * base.fundamental_period,
    )
    maximizer = PeriodicHarmonicMaximizer()
    base_result = maximizer.maximize(base)
    shifted_result = maximizer.maximize(shifted)

    assert base_result.maximum_value == shifted_result.maximum_value
    assert base_result.amplitude == shifted_result.amplitude
    assert base_result.canonical_maximizing_argument is not None
    assert shifted_result.canonical_maximizing_argument is not None
    assert shifted_result.canonical_maximizing_argument == pytest.approx(
        base_result.canonical_maximizing_argument + 9.0 * base.fundamental_period,
        rel=0.0,
        abs=2.0e-15,
    )
