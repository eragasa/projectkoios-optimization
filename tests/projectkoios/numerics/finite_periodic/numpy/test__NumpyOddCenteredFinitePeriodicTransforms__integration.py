"""Integration verification for odd centered finite-periodic matrix transforms."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.numpy.matrix_series import (
    NumpyFiniteFourierMatrixSeries,
)
from projectkoios.numerics.finite_periodic.numpy.tensor_product_evaluation import (
    NumpyTensorProductFourierMatrixEvaluationDefinition,
    NumpyTensorProductFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.transform_diagnostics import (
    NumpyCenteredUniformMeshHermiticityAnalyzer,
    NumpyFinitePeriodicParsevalAnalyzer,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformer,
)

pytestmark = pytest.mark.integration

_COEFFICIENT = np.array(
    [[1.0 + 0.0j, 0.25 - 0.5j], [-0.125 + 0.75j, -2.0 + 0.25j]],
    dtype=np.complex128,
)


def _coordinate_axis(size: int, offset: float) -> np.ndarray:
    return -0.5 + (np.arange(size, dtype=np.float64) + offset) / float(size)


@pytest.mark.parametrize("size", [513, 1025, 2049])
def test_odd_transform_recovers_positive_centered_endpoint_mode(size: int) -> None:
    mesh = CenteredUniformMesh(1, size, (0.5,))
    coordinates = _coordinate_axis(size, mesh.offsets[0])
    mode = size // 2
    samples = (
        np.exp(2.0j * np.pi * coordinates * float(mode))[:, None, None] * _COEFFICIENT
    )

    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
            mesh,
            np.asarray(samples, dtype=np.complex128),
        )
    ).result()

    modes = transformed.series.integer_modes[:, 0]
    coefficients = transformed.series.matrix_coefficients
    selected = modes == mode
    assert modes.shape == (size,)
    assert modes[0] == -(size // 2)
    assert modes[-1] == size // 2
    assert np.count_nonzero(selected) == 1
    assert coefficients[selected][0] == pytest.approx(
        _COEFFICIENT,
        abs=2.0e-12,
        rel=0.0,
    )
    assert coefficients[~selected] == pytest.approx(
        np.zeros_like(coefficients[~selected]),
        abs=2.0e-12,
        rel=0.0,
    )


@pytest.mark.parametrize(
    ("dimension", "mode"),
    [
        (2, (1, -1)),
        (3, (1, -1, 0)),
    ],
)
def test_multidimensional_odd_transform_recovers_declared_mode(
    dimension: int,
    mode: tuple[int, ...],
) -> None:
    mesh = CenteredUniformMesh(dimension, 3, (0.5,) * dimension)
    axis = _coordinate_axis(mesh.size, 0.5)
    coordinates = np.stack(
        np.meshgrid(*(axis,) * dimension, indexing="ij"),
        axis=-1,
    )
    mode_array = np.asarray(mode, dtype=np.int64)
    samples = (
        np.exp(2.0j * np.pi * np.einsum("...d,d->...", coordinates, mode_array))[
            ..., None, None
        ]
        * _COEFFICIENT
    )

    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
            mesh,
            np.asarray(samples, dtype=np.complex128),
        )
    ).result()

    selected = np.all(transformed.series.integer_modes == mode_array, axis=1)
    assert np.count_nonzero(selected) == 1
    assert transformed.series.matrix_coefficients[selected][0] == pytest.approx(
        _COEFFICIENT,
        abs=2.0e-15,
        rel=0.0,
    )
    assert transformed.series.matrix_coefficients[~selected] == pytest.approx(
        np.zeros_like(transformed.series.matrix_coefficients[~selected]),
        abs=2.0e-15,
        rel=0.0,
    )


def test_odd_transform_matches_independent_negative_sign_normalized_sum() -> None:
    size = 7
    offset = 0.375
    mesh = CenteredUniformMesh(1, size, (offset,))
    coordinates = _coordinate_axis(size, offset)
    samples = np.empty((size, 2, 2), dtype=np.complex128)
    for index, coordinate in enumerate(coordinates):
        samples[index] = np.array(
            [
                [1.0 + coordinate + 0.25j, -0.5j * coordinate],
                [0.75 * coordinate, -0.25 + 0.5j * coordinate],
            ],
            dtype=np.complex128,
        )
    modes = np.arange(-(size // 2), size // 2 + 1, dtype=np.int64)
    expected = np.empty((size, 2, 2), dtype=np.complex128)
    for mode_index, mode in enumerate(modes):
        phase = np.exp(-2.0j * np.pi * coordinates * float(mode))
        expected[mode_index] = np.sum(
            phase[:, None, None] * samples,
            axis=0,
        ) / float(size)

    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
            mesh,
            samples,
        )
    ).result()

    np.testing.assert_array_equal(
        transformed.series.integer_modes[:, 0],
        modes,
    )
    assert transformed.series.matrix_coefficients == pytest.approx(
        expected,
        abs=8.0e-16,
        rel=0.0,
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_odd_parseval_and_ordinary_hermiticity_cover_complete_inventory(
    dimension: int,
) -> None:
    mesh = CenteredUniformMesh(dimension, 3, (0.5,) * dimension)
    random = np.random.default_rng(20261018 + dimension)
    shape = (mesh.size,) * dimension + (2, 2)
    raw = np.asarray(
        random.normal(size=shape) + 1.0j * random.normal(size=shape),
        dtype=np.complex128,
    )
    samples = raw + np.swapaxes(raw.conjugate(), -1, -2)
    definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
        mesh,
        samples,
    )
    transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=definition
    ).result()

    parseval = NumpyFinitePeriodicParsevalAnalyzer(
        definition=definition,
        transform_result=transformed,
    ).result()
    hermiticity = NumpyCenteredUniformMeshHermiticityAnalyzer(
        transform_result=transformed
    ).result()

    assert parseval.absolute_defect <= (
        2.0e-15 * parseval.sample_mean_squared_frobenius_norm
    )
    assert hermiticity.twisted_cyclic_maximum_frobenius_defect <= 3.0e-15
    assert (
        hermiticity.ordinary_maximum_frobenius_defect
        == hermiticity.twisted_cyclic_maximum_frobenius_defect
    )


def test_odd_tensor_evaluation_covers_required_shifted_axis_scale() -> None:
    size = 2049
    evaluation_size = 8191
    mesh = CenteredUniformMesh(1, size, (0.5,))
    mode = size // 2
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        1,
        np.array([[mode]], dtype=np.int64),
        _COEFFICIENT[None, ...],
    )
    evaluation_axis = _coordinate_axis(evaluation_size, 0.75)
    definition = NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        series,
        (evaluation_axis,),
        maximum_working_elements=1_000_000,
    )

    evaluated = NumpyTensorProductFourierMatrixSeriesEvaluator(
        definition=definition
    ).result()

    expected = (
        np.exp(2.0j * np.pi * evaluation_axis * float(mode))[:, None, None]
        * _COEFFICIENT
    )
    assert definition.leading_axis_batch_size == 488
    assert definition.maximum_temporary_elements == 999_912
    assert evaluated.matrix_values.shape == (evaluation_size, 2, 2)
    assert evaluated.matrix_values == pytest.approx(
        expected,
        abs=3.0e-13,
        rel=0.0,
    )
