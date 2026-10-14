"""Bounded numerical verification for NumPy finite Fourier-series evaluation."""

from __future__ import annotations

import math
import random

import mpmath
import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.finite_series import (
    NumpyFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.numerical


def _high_precision_reference(
    definition: FiniteFourierSeriesEvaluationDefinition,
) -> tuple[complex, ...]:
    with mpmath.workdps(100):
        result: list[complex] = []
        for coordinate in definition.coordinates:
            total = mpmath.mpc(0)
            high_precision_coordinate = tuple(mpmath.mpf(value) for value in coordinate)
            for mode, coefficient in zip(
                definition.integer_modes,
                definition.coefficients,
                strict=True,
            ):
                dot = mpmath.fsum(
                    value * mode_component
                    for value, mode_component in zip(
                        high_precision_coordinate,
                        mode,
                        strict=True,
                    )
                )
                phase = 2 * mpmath.pi * dot
                total += mpmath.mpc(coefficient.real, coefficient.imag) * mpmath.expj(
                    phase
                )
            result.append(complex(float(total.real), float(total.imag)))
        return tuple(result)


def _acceptance_bound(coefficients: tuple[complex, ...], factor: int) -> float:
    epsilon = math.ulp(1.0)
    return factor * epsilon * max(1.0, sum(abs(value) for value in coefficients))


def test_bounded_random_cases_agree_with_100_digit_reference() -> None:
    generator = random.Random(20261014)
    evaluator = NumpyFiniteFourierSeriesEvaluator()

    for dimension in (1, 2, 3):
        coordinates = tuple(
            tuple(generator.uniform(-1.0, 1.0) for _ in range(dimension))
            for _ in range(11)
        )
        modes = tuple(
            tuple(generator.randint(-16, 16) for _ in range(dimension))
            for _ in range(13)
        )
        coefficients = tuple(
            complex(generator.uniform(-2.0, 2.0), generator.uniform(-2.0, 2.0))
            for _ in modes
        )
        definition = FiniteFourierSeriesEvaluationDefinition(
            dimension,
            coordinates,
            modes,
            coefficients,
        )

        actual = evaluator.evaluate(definition).values
        expected = _high_precision_reference(definition)
        bound = _acceptance_bound(coefficients, 512)

        assert all(
            abs(actual_value - expected_value) <= bound
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )


def test_exact_angle_cases_satisfy_analytic_acceptance_criterion() -> None:
    coefficients = (2.0 - 1.0j, -0.5 + 0.25j)
    definition = FiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.0,), (0.25,), (0.5,), (0.75,)),
        ((0,), (1,)),
        coefficients,
    )
    expected = (
        1.5 - 0.75j,
        1.75 - 1.5j,
        2.5 - 1.25j,
        2.25 - 0.5j,
    )

    actual = NumpyFiniteFourierSeriesEvaluator().evaluate(definition).values
    bound = _acceptance_bound(coefficients, 128)

    assert all(
        abs(actual_value - expected_value) <= bound
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def test_integer_coordinate_translation_is_periodic_within_bounded_criterion() -> None:
    coefficients = (1.0 + 0.5j, -0.75 + 0.25j, 0.125 - 0.5j)
    original = ((0.125, -0.375), (0.2, 0.3))
    translated = tuple((x + 3.0, y - 2.0) for x, y in original)
    modes = ((1, 2), (-3, 4), (5, -6))
    evaluator = NumpyFiniteFourierSeriesEvaluator()

    first = evaluator.evaluate(
        FiniteFourierSeriesEvaluationDefinition(
            2,
            original,
            modes,
            coefficients,
        )
    ).values
    second = evaluator.evaluate(
        FiniteFourierSeriesEvaluationDefinition(
            2,
            translated,
            modes,
            coefficients,
        )
    ).values
    bound = _acceptance_bound(coefficients, 1024)

    assert all(
        abs(first_value - second_value) <= bound
        for first_value, second_value in zip(first, second, strict=True)
    )


def test_cancellation_is_checked_on_absolute_coefficient_sum_scale() -> None:
    coefficients = (
        2.5 + 2.5j,
        -2.5 - 2.5j,
        2.0 - 3.0j,
        -2.0 + 3.0j,
    )
    definition = FiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.123456789,),),
        ((7,), (7,), (-11,), (-11,)),
        coefficients,
    )

    actual = NumpyFiniteFourierSeriesEvaluator().evaluate(definition).values[0]
    expected = _high_precision_reference(definition)[0]
    bound = _acceptance_bound(coefficients, 512)

    assert abs(actual - expected) <= bound
