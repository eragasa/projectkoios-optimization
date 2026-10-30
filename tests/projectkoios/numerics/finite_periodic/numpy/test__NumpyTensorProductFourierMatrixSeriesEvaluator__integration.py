"""Integration tests for separable tensor-product matrix-series evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.numpy.matrix_evaluation import (
    NumpyFiniteFourierMatrixEvaluationDefinition,
    NumpyFiniteFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.matrix_series import (
    NumpyFiniteFourierMatrixSeries,
)
from projectkoios.numerics.finite_periodic.numpy.tensor_product_evaluation import (
    NumpyTensorProductFourierMatrixEvaluationDefinition,
    NumpyTensorProductFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformer,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_separable_tensor_evaluation_matches_bounded_selected_mode_evaluation(
    dimension: int,
) -> None:
    mesh = CenteredUniformMesh(dimension, 4, (0.5,) * dimension)
    rng = np.random.default_rng(6100 + dimension)
    shape = (mesh.size,) * dimension + (2, 2)
    samples = (rng.normal(size=shape) + 1.0j * rng.normal(size=shape)).astype(
        np.complex128
    )
    definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
        mesh,
        samples,
    )
    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=definition
    ).result()
    axes = tuple(
        np.linspace(-0.43, 0.41, 5, dtype=np.float64) for _ in range(dimension)
    )
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    bounded = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            transformed.series,
            coordinates,
            maximum_phase_elements=13,
        )
    )
    tensor_definition = NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        transformed.series,
        axes,
        maximum_working_elements=1_000,
    )
    separable = NumpyTensorProductFourierMatrixSeriesEvaluator(
        definition=tensor_definition
    ).result()

    assert tensor_definition.maximum_temporary_elements <= 1_000
    assert separable.matrix_values == pytest.approx(
        bounded.matrix_values,
        abs=2.0e-14,
        rel=0.0,
    )
    assert separable.backend_identifier == "numpy.einsum.tensor_product"
    assert separable.backend_version == np.__version__
    assert not separable.matrix_values.flags.writeable


def test_selected_cartesian_product_inventory_matches_bounded_evaluation() -> None:
    mesh = CenteredUniformMesh(2, 4, (0.5, 0.5))
    modes = np.array(
        [[first, second] for first in (-1, 1) for second in (0, 1)],
        dtype=np.int64,
    )
    coefficients = (
        np.arange(1, 5, dtype=np.float64).astype(np.complex128).reshape((4, 1, 1))
    )
    series = NumpyFiniteFourierMatrixSeries.from_arrays(2, modes, coefficients)
    axes = (
        np.linspace(-0.4, 0.4, 7, dtype=np.float64),
        np.linspace(-0.3, 0.3, 5, dtype=np.float64),
    )
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    bounded = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            series,
            coordinates,
            maximum_phase_elements=9,
        )
    )
    separable = NumpyTensorProductFourierMatrixSeriesEvaluator(
        definition=NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            axes,
            maximum_working_elements=100,
        )
    ).result()

    assert separable.matrix_values == pytest.approx(
        bounded.matrix_values,
        abs=4.0e-15,
        rel=0.0,
    )


def test_duplicate_modes_accumulate_in_declared_order() -> None:
    mesh = CenteredUniformMesh(1, 4, (0.5,))
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        1,
        np.array([[1], [1]], dtype=np.int64),
        np.array([[[1.0 + 2.0j]], [[3.0 - 1.0j]]], dtype=np.complex128),
    )
    axis = np.array([-0.25, 0.0, 0.25], dtype=np.float64)

    evaluated = NumpyTensorProductFourierMatrixSeriesEvaluator(
        definition=NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
            mesh,
            series,
            (axis,),
            maximum_working_elements=16,
        )
    ).result()

    expected = (4.0 + 1.0j) * np.exp(2.0j * np.pi * axis)
    assert evaluated.matrix_values[:, 0, 0] == pytest.approx(
        expected,
        abs=2.0e-15,
        rel=0.0,
    )


def test_separable_tensor_evaluation_covers_retained_three_dimensional_scale() -> None:
    mesh = CenteredUniformMesh(3, 32, (0.5, 0.5, 0.5))
    constant_matrix = np.array(
        [[2.0 + 0.0j, 0.25 - 0.5j], [0.25 + 0.5j, -1.0 + 0.0j]],
        dtype=np.complex128,
    )
    samples = np.broadcast_to(
        constant_matrix,
        (mesh.size, mesh.size, mesh.size, 2, 2),
    ).copy()
    definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
        mesh,
        samples,
    )
    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=definition
    ).result()
    evaluation_axis = -0.5 + (np.arange(65, dtype=np.float64) + 0.25) / 65.0
    tensor_definition = NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        transformed.series,
        (evaluation_axis, evaluation_axis, evaluation_axis),
        maximum_working_elements=1_000_000,
    )

    evaluated = NumpyTensorProductFourierMatrixSeriesEvaluator(
        definition=tensor_definition
    ).result()

    assert tensor_definition.coefficient_grid_elements == 131_072
    assert tensor_definition.maximum_temporary_elements <= 1_000_000
    assert evaluated.matrix_values.shape == (65, 65, 65, 2, 2)
    assert float(np.max(np.abs(evaluated.matrix_values - constant_matrix))) <= 2.0e-15
