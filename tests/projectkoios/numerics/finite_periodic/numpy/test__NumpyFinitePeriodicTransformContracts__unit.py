"""Contract verification for immutable NumPy finite-periodic transform records."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.numpy.matrix_evaluation import (
    NumpyFiniteFourierMatrixEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.matrix_series import (
    NumpyFiniteFourierMatrixSeries,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
)

pytestmark = pytest.mark.unit


def test_transform_definition_defensively_copies_exact_array() -> None:
    mesh = CenteredUniformMesh(2, 2, (0.5, 0.5))
    source = np.zeros((2, 2, 2, 2), dtype=np.complex128)

    definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
        mesh,
        source,
    )
    source[0, 0, 0, 0] = 5.0 + 0.0j

    assert definition.matrix_samples[0, 0, 0, 0] == 0.0 + 0.0j
    assert not definition.matrix_samples.flags.writeable
    with pytest.raises(ValueError):
        definition.matrix_samples.flags.writeable = True


def test_transform_definition_rejects_incomplete_tensor_mesh() -> None:
    mesh = CenteredUniformMesh(2, 4, (0.5, 0.5))
    samples = np.zeros((4, 3, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
            mesh,
            samples,
        )


def test_matrix_series_rejects_non_int64_modes() -> None:
    modes = np.array([[0.0], [1.0]], dtype=np.float64)
    coefficients = np.zeros((2, 1, 1), dtype=np.complex128)

    with pytest.raises(TypeError):
        NumpyFiniteFourierMatrixSeries.from_arrays(
            1,
            modes,  # type: ignore[arg-type]
            coefficients,
        )


def test_matrix_series_rejects_modes_not_exactly_convertible_to_float64() -> None:
    modes = np.array([[2**53 + 1]], dtype=np.int64)
    coefficients = np.zeros((1, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError):
        NumpyFiniteFourierMatrixSeries.from_arrays(1, modes, coefficients)


def test_matrix_series_rejects_rectangular_coefficients() -> None:
    modes = np.array([[0, 0], [1, 0]], dtype=np.int64)
    coefficients = np.zeros((2, 2, 3), dtype=np.complex128)

    with pytest.raises(ValueError):
        NumpyFiniteFourierMatrixSeries.from_arrays(2, modes, coefficients)


def test_evaluation_definition_rejects_coordinate_width_mismatch() -> None:
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        2,
        np.array([[0, 0]], dtype=np.int64),
        np.ones((1, 1, 1), dtype=np.complex128),
    )
    coordinates = np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError):
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(series, coordinates)


def test_evaluation_definition_rejects_nonpositive_phase_bound() -> None:
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        1,
        np.array([[0]], dtype=np.int64),
        np.ones((1, 1, 1), dtype=np.complex128),
    )
    coordinates = np.zeros((1, 1), dtype=np.float64)

    with pytest.raises(ValueError):
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            series,
            coordinates,
            maximum_phase_elements=0,
        )
