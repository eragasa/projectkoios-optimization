"""Numerical evidence for metric-free reduced-coordinate Fourier phases."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.numpy.matrix_evaluation import (
    NumpyFiniteFourierMatrixEvaluationDefinition,
    NumpyFiniteFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.matrix_series import (
    NumpyFiniteFourierMatrixSeries,
)

pytestmark = pytest.mark.numerical

_OBLIQUE_DIRECT_PRIMITIVES = np.array(
    [[1.0, 0.35], [0.2, 1.1]],
    dtype=np.float64,
)
_FCC_DIRECT_PRIMITIVES = np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    dtype=np.float64,
)
_MATRIX_COEFFICIENTS = np.array(
    [
        [[1.0 + 0.0j, 0.25j], [-0.25j, -0.5 + 0.0j]],
        [[0.2 + 0.1j, -0.4 + 0.0j], [0.3j, 0.75 - 0.2j]],
        [[-0.3 + 0.0j, 0.1 - 0.2j], [0.6 + 0.1j, 0.4 + 0.0j]],
    ],
    dtype=np.complex128,
)


@pytest.mark.parametrize(
    ("direct_primitives", "reduced_coordinates", "integer_modes"),
    [
        (
            _OBLIQUE_DIRECT_PRIMITIVES,
            np.array([[0.1, -0.2], [0.35, 0.4], [-0.45, 0.25]], dtype=np.float64),
            np.array([[0, 0], [1, -2], [-1, 1]], dtype=np.int64),
        ),
        (
            _FCC_DIRECT_PRIMITIVES,
            np.array(
                [[0.1, -0.2, 0.3], [0.35, 0.4, -0.15], [-0.45, 0.25, 0.2]],
                dtype=np.float64,
            ),
            np.array([[0, 0, 0], [1, -2, 1], [-1, 1, 2]], dtype=np.int64),
        ),
    ],
)
def test_reduced_coordinate_evaluation_matches_angular_dual_lattice_oracle(
    direct_primitives: np.ndarray,
    reduced_coordinates: np.ndarray,
    integer_modes: np.ndarray,
) -> None:
    """Oblique and FCC primitive cells use the same q-dot-n numerical contract."""
    dimension = int(reduced_coordinates.shape[1])
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        dimension,
        integer_modes,
        _MATRIX_COEFFICIENTS,
    )

    result = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            series,
            reduced_coordinates,
            maximum_phase_elements=4,
        )
    )

    angular_reciprocal_primitives = 2.0 * np.pi * np.linalg.inv(direct_primitives).T
    physical_wave_vectors = reduced_coordinates @ angular_reciprocal_primitives.T
    physical_translations = integer_modes @ direct_primitives.T
    angular_pairings = physical_wave_vectors @ physical_translations.T
    reduced_pairings = 2.0 * np.pi * (reduced_coordinates @ integer_modes.T)
    assert angular_pairings == pytest.approx(
        reduced_pairings,
        abs=2.0e-15,
        rel=0.0,
    )
    physical_phases = np.exp(1.0j * angular_pairings)
    expected = np.einsum(
        "pr,rij->pij",
        physical_phases,
        _MATRIX_COEFFICIENTS,
        optimize=False,
    )

    assert result.matrix_values == pytest.approx(expected, abs=4.0e-15, rel=0.0)
