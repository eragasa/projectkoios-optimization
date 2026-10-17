"""Unit tests for tensor-product matrix Fourier evaluation definitions."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.numpy.matrix_series import (
    NumpyFiniteFourierMatrixSeries,
)
from projectkoios.numerics.finite_periodic.numpy.tensor_product_evaluation import (
    NumpyTensorProductFourierMatrixEvaluationDefinition,
)


def test_rejects_mode_outside_construction_box() -> None:
    mesh = CenteredUniformMesh(1, 4, (0.5,))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        1,
        np.array([[2]], dtype=np.int64),
        np.ones((1, 1, 1), dtype=np.complex128),
    )

    with pytest.raises(ValueError):
        NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            (np.array([0.0], dtype=np.float64),),
        )


def test_rejects_insufficient_bound_for_coefficient_grid() -> None:
    mesh = CenteredUniformMesh(3, 4, (0.5, 0.5, 0.5))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        3,
        np.array([[0, 0, 0]], dtype=np.int64),
        np.ones((1, 2, 2), dtype=np.complex128),
    )
    axes = (np.zeros(2, dtype=np.float64),) * 3

    with pytest.raises(ValueError):
        NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            axes,
            maximum_working_elements=255,
        )


def test_rejects_insufficient_bound_for_nonleading_phase_matrix() -> None:
    mesh = CenteredUniformMesh(2, 2, (0.5, 0.5))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        2,
        np.array([[0, 0]], dtype=np.int64),
        np.ones((1, 1, 1), dtype=np.complex128),
    )
    axes = (
        np.zeros(1, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
    )

    with pytest.raises(ValueError):
        NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            axes,
            maximum_working_elements=10,
        )


def test_rejects_insufficient_bound_for_one_output_slice() -> None:
    mesh = CenteredUniformMesh(3, 2, (0.5, 0.5, 0.5))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        3,
        np.array([[0, 0, 0]], dtype=np.int64),
        np.ones((1, 2, 2), dtype=np.complex128),
    )
    axes = tuple(np.zeros(length, dtype=np.float64) for length in (2, 3, 4))

    with pytest.raises(ValueError):
        NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            axes,
            maximum_working_elements=40,
        )


def test_one_dimensional_batch_bound_includes_leading_phase_matrix() -> None:
    mesh = CenteredUniformMesh(1, 8, (0.5,))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        1,
        np.array([[0]], dtype=np.int64),
        np.ones((1, 1, 1), dtype=np.complex128),
    )

    definition = NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        series,
        (np.zeros(17, dtype=np.float64),),
        maximum_working_elements=16,
    )

    assert definition.elements_per_leading_axis_point == 8
    assert definition.leading_axis_batch_size == 2
    assert definition.maximum_temporary_elements == 16


def test_reports_exact_working_element_bounds() -> None:
    mesh = CenteredUniformMesh(3, 4, (0.5, 0.5, 0.5))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        3,
        np.array([[0, 0, 0]], dtype=np.int64),
        np.ones((1, 2, 2), dtype=np.complex128),
    )
    axes = tuple(np.zeros(length, dtype=np.float64) for length in (7, 5, 3))

    definition = NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        series,
        axes,
        maximum_working_elements=400,
    )

    assert definition.point_shape == (7, 5, 3)
    assert definition.coefficient_grid_elements == 256
    assert definition.maximum_nonleading_phase_elements == 20
    assert definition.elements_per_leading_axis_point == 80
    assert definition.leading_axis_batch_size == 5
    assert definition.maximum_temporary_elements == 400
