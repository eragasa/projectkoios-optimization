"""Integration verification for NumPy centered finite-periodic transforms."""

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
from projectkoios.numerics.finite_periodic.numpy.transform_diagnostics import (
    NumpyCenteredUniformMeshHermiticityAnalyzer,
    NumpyFinitePeriodicParsevalAnalyzer,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformer,
    NumpyCenteredUniformMeshMatrixFourierTransformResult,
)

pytestmark = pytest.mark.integration

_MODE_COEFFICIENT = np.array(
    [[1.0 + 0.0j, 0.25 + 0.5j], [-0.75 + 0.125j, 2.0 - 0.25j]],
    dtype=np.complex128,
)


def _reduced_coordinates(mesh: CenteredUniformMesh) -> np.ndarray:
    axes = tuple(
        -0.5 + (np.arange(mesh.size, dtype=np.float64) + offset) / float(mesh.size)
        for offset in mesh.offsets
    )
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


@pytest.mark.parametrize(
    ("dimension", "offsets", "mode"),
    [
        (1, (0.25,), (1,)),
        (2, (0.25, 0.625), (1, -1)),
        (3, (0.25, 0.625, 0.875), (1, -1, 0)),
    ],
)
def test_forward_transform_recovers_one_declared_integer_mode(
    dimension: int,
    offsets: tuple[float, ...],
    mode: tuple[int, ...],
) -> None:
    mesh = CenteredUniformMesh(dimension, 4, offsets)
    coordinates = _reduced_coordinates(mesh)
    mode_array = np.asarray(mode, dtype=np.int64)
    phase = np.exp(2.0j * np.pi * np.einsum("...d,d->...", coordinates, mode_array))
    samples = phase[..., None, None] * _MODE_COEFFICIENT

    definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
        mesh,
        samples,
    )
    result = NumpyCenteredUniformMeshMatrixFourierTransformer(
        definition=definition
    ).result()

    selected = np.all(result.series.integer_modes == mode_array, axis=1)
    assert np.count_nonzero(selected) == 1
    assert result.series.matrix_coefficients[selected][0] == pytest.approx(
        _MODE_COEFFICIENT,
        abs=8.0e-16,
        rel=0.0,
    )
    assert result.series.matrix_coefficients[~selected] == pytest.approx(
        np.zeros_like(result.series.matrix_coefficients[~selected]),
        abs=8.0e-16,
        rel=0.0,
    )
    assert result.backend_identifier == "numpy.fft.fftn"
    assert result.backend_version == np.__version__


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_complete_transform_reconstructs_its_source_mesh(dimension: int) -> None:
    mesh = CenteredUniformMesh(dimension, 4, (0.5,) * dimension)
    coordinates = _reduced_coordinates(mesh)
    rng = np.random.default_rng(4100 + dimension)
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

    evaluated = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            transformed.series,
            coordinates,
            maximum_phase_elements=11,
        )
    )

    assert evaluated.point_shape == coordinates.shape[:-1]
    assert evaluated.term_count == mesh.point_count
    assert evaluated.matrix_values == pytest.approx(samples, abs=8.0e-15, rel=0.0)
    assert not evaluated.matrix_values.flags.writeable


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_parseval_and_origin_aware_hermiticity_hold_for_hermitian_samples(
    dimension: int,
) -> None:
    mesh = CenteredUniformMesh(dimension, 4, (0.5,) * dimension)
    rng = np.random.default_rng(8200 + dimension)
    shape = (mesh.size,) * dimension + (2, 2)
    raw = (rng.normal(size=shape) + 1.0j * rng.normal(size=shape)).astype(np.complex128)
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

    scale = parseval.sample_mean_squared_frobenius_norm
    assert parseval.absolute_defect <= 2.0e-15 * scale
    assert hermiticity.twisted_cyclic_maximum_frobenius_defect <= 2.0e-14
    assert hermiticity.ordinary_maximum_frobenius_defect <= 2.0e-14


def test_hermiticity_analyzer_reports_independently_declared_defect() -> None:
    mesh = CenteredUniformMesh(1, 4, (0.5,))
    modes = np.array([[-2], [-1], [0], [1]], dtype=np.int64)
    coefficients = np.array(
        [
            [[0.0j, 0.0j], [0.0j, 0.0j]],
            [[0.0j, 0.0j], [0.0j, 0.0j]],
            [[0.0j, 0.0j], [0.0j, 0.0j]],
            [[0.0j, 0.25j], [0.0j, 0.0j]],
        ],
        dtype=np.complex128,
    )
    series = NumpyFiniteFourierMatrixSeries.from_arrays(1, modes, coefficients)
    result = NumpyCenteredUniformMeshMatrixFourierTransformResult(
        mesh,
        series,
        "test.independently_declared_hermiticity_defect",
        "1",
    )

    diagnostics = NumpyCenteredUniformMeshHermiticityAnalyzer(
        transform_result=result
    ).result()

    assert diagnostics.twisted_cyclic_maximum_frobenius_defect == pytest.approx(
        0.25,
        abs=2.0e-16,
        rel=0.0,
    )
    assert diagnostics.ordinary_maximum_frobenius_defect == pytest.approx(
        0.25,
        abs=2.0e-16,
        rel=0.0,
    )


def test_evaluator_bounds_every_temporary_point_by_mode_phase_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes = np.array([[-2], [-1], [0], [1], [2]], dtype=np.int64)
    coefficients = (
        np.arange(20, dtype=np.float64).reshape((5, 2, 2)).astype(np.complex128)
    )
    series = NumpyFiniteFourierMatrixSeries.from_arrays(1, modes, coefficients)
    coordinates = np.linspace(-0.4, 0.4, 3, dtype=np.float64).reshape((-1, 1))
    original_exp = np.exp
    phase_shapes: list[tuple[int, ...]] = []

    def recording_exp(values: np.ndarray) -> np.ndarray:
        phase_shapes.append(values.shape)
        return original_exp(values)

    monkeypatch.setattr(np, "exp", recording_exp)

    result = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
            series,
            coordinates,
            maximum_phase_elements=3,
        )
    )

    assert result.matrix_values.shape == (3, 2, 2)
    assert phase_shapes == [(1, 3), (1, 2)] * 3
    assert all(np.prod(shape) <= 3 for shape in phase_shapes)


def test_empty_selected_series_evaluates_to_exact_zero_matrices() -> None:
    series = NumpyFiniteFourierMatrixSeries.from_arrays(
        2,
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 3, 3), dtype=np.complex128),
    )
    coordinates = np.array([[0.0, 0.0], [0.25, -0.5]], dtype=np.float64)

    result = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
        NumpyFiniteFourierMatrixEvaluationDefinition.from_array(series, coordinates)
    )

    assert np.array_equal(result.matrix_values, np.zeros((2, 3, 3)))
