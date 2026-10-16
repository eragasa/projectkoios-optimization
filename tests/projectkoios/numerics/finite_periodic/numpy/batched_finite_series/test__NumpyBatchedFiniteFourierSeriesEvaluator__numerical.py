"""Bounded numerical verification for batched finite Fourier evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.batched_finite_series import (
    NumpyBatchedFiniteFourierSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.finite_series import (
    NumpyFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.numerical


def _scale_aware_bound(
    coefficients: tuple[tuple[complex, ...], ...],
) -> float:
    scale = max(
        1.0,
        max(
            sum(abs(coefficient) for coefficient in channel) for channel in coefficients
        ),
    )
    term_count = len(coefficients[0]) if coefficients else 0
    return 64.0 * np.finfo(np.float64).eps * max(1, term_count) * scale


def _assert_scalar_parity(
    dimension: int,
    coordinates: tuple[tuple[float, ...], ...],
    modes: tuple[tuple[int, ...], ...],
    coefficients: tuple[tuple[complex, ...], ...],
) -> None:
    batched = NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            dimension,
            coordinates,
            modes,
            coefficients,
        )
    )
    scalar = NumpyFiniteFourierSeriesEvaluator()
    expected_columns = tuple(
        scalar.evaluate(
            FiniteFourierSeriesEvaluationDefinition(
                dimension,
                coordinates,
                modes,
                channel,
            )
        ).values
        for channel in coefficients
    )
    expected = np.asarray(expected_columns, dtype=np.complex128).T
    actual = np.asarray(batched.values, dtype=np.complex128)

    assert actual.shape == (len(coordinates), len(coefficients))
    assert float(np.max(np.abs(actual - expected))) <= _scale_aware_bound(coefficients)


def test_real_and_complex_shifted_one_dimensional_channels_match_scalar_backend() -> (
    None
):
    coordinates = tuple((float(value),) for value in np.linspace(-0.47, 0.49, 37))
    modes = tuple((mode,) for mode in range(-12, 13))
    coefficients = (
        tuple(complex(math.cos(mode), 0.0) for (mode,) in modes),
        tuple(complex(math.sin(mode), math.cos(0.5 * mode)) for (mode,) in modes),
        tuple(
            0.0 + 0.0j if abs(mode) > 4 else complex(0.25 * mode, -0.1 * mode)
            for (mode,) in modes
        ),
    )

    _assert_scalar_parity(1, coordinates, modes, coefficients)


def test_multidimensional_seeded_channels_match_scalar_backend() -> None:
    random = np.random.default_rng(240318)
    coordinates_array = random.uniform(-0.75, 0.75, size=(29, 3))
    modes_array = random.integers(-7, 8, size=(19, 3), endpoint=False)
    coefficients_array = random.normal(size=(8, 19)) + 1j * random.normal(size=(8, 19))
    coordinates = tuple(
        tuple(float(value) for value in row) for row in coordinates_array
    )
    modes = tuple(tuple(int(value) for value in row) for row in modes_array)
    coefficients = tuple(
        tuple(complex(value) for value in row) for row in coefficients_array
    )

    _assert_scalar_parity(3, coordinates, modes, coefficients)


def test_zero_masked_range_family_matches_independent_scalar_series() -> None:
    coordinates = tuple((float(value),) for value in np.linspace(-0.5, 0.5, 65))
    modes = tuple((mode,) for mode in range(-16, 17))
    source = tuple(complex(math.exp(-abs(mode) / 4.0), 0.1 * mode) for (mode,) in modes)
    cutoffs = (0, 1, 2, 4, 8, 16)
    coefficients = (source,) + tuple(
        tuple(
            coefficient if abs(mode) <= cutoff else 0.0 + 0.0j
            for (mode,), coefficient in zip(modes, source, strict=True)
        )
        for cutoff in cutoffs
    )

    _assert_scalar_parity(1, coordinates, modes, coefficients)


def test_conjugate_symmetric_channels_remain_real_within_roundoff() -> None:
    coordinates = tuple((float(value),) for value in np.linspace(-0.5, 0.5, 101))
    modes = tuple((mode,) for mode in range(-10, 11))
    coefficients = (
        tuple(complex(math.exp(-abs(mode)), 0.0) for (mode,) in modes),
        tuple(complex(1.0 / (1.0 + mode * mode), 0.0) for (mode,) in modes),
    )
    result = NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            coordinates,
            modes,
            coefficients,
        )
    )

    assert float(np.max(np.abs(np.asarray(result.values).imag))) <= 5.0e-15
