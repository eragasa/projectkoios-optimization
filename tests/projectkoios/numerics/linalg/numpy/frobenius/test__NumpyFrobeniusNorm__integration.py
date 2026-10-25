"""Integration tests for dense NumPy Frobenius values."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.linalg.frobenius import ComplexMatrixFrobenius
from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.numpy.frobenius import (
    NumpyFrobeniusNorm,
    numpy_matrix_frobenius,
)
from projectkoios.numerics.numpy.typing import DenseMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected_power"),
    [
        (np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64), 25.0),
        (
            np.asarray(
                (
                    (3.0 + 4.0j, 0.0 + 0.0j),
                    (1.0 - 1.0j, 0.0 + 1.0j),
                ),
                dtype=np.complex128,
            ),
            28.0,
        ),
    ],
)
def test_detailed_operation_evaluates_supported_dense_storage(
    matrix: DenseMatrix,
    expected_power: float,
) -> None:
    result = numpy_matrix_frobenius(matrix)

    assert result == ComplexMatrixFrobenius(
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        expected_power,
        expected_power**0.5,
    )


def test_evaluator_reports_semantics_and_returns_scalar_norm() -> None:
    evaluator = NumpyFrobeniusNorm()
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    result = evaluator(matrix)

    assert evaluator.matrix_norm is MatrixNormEnum.frobenius
    assert result == 5.0
    assert type(result) is float


def test_logical_row_major_reduction_accepts_storage_layouts_and_views() -> None:
    fortran_matrix: DenseMatrix = np.asfortranarray(
        np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    )
    source = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    noncontiguous_view: DenseMatrix = source[:, ::2]

    fortran_result = numpy_matrix_frobenius(fortran_matrix)
    view_result = numpy_matrix_frobenius(noncontiguous_view)

    assert fortran_result.power == 91.0
    assert view_result.power == 220.0


def test_operation_does_not_mutate_writable_or_read_only_input() -> None:
    writable: DenseMatrix = np.asarray(((1.0 + 2.0j, 3.0 + 4.0j),), dtype=np.complex128)
    writable_before = writable.copy()
    read_only: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)
    read_only.setflags(write=False)

    writable_result = numpy_matrix_frobenius(writable)
    read_only_result = numpy_matrix_frobenius(read_only)

    assert np.array_equal(writable, writable_before)
    assert writable.flags.writeable
    assert not read_only.flags.writeable
    assert writable_result.power == 30.0
    assert read_only_result.norm == 5.0


def test_zero_matrix_has_exact_zero_power_and_norm() -> None:
    matrix: DenseMatrix = np.zeros((2, 3), dtype=np.float64)

    result = numpy_matrix_frobenius(matrix)

    assert result.power == 0.0
    assert result.norm == 0.0


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray((1.0, 2.0), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.complex128),
    ],
)
def test_rejects_non_matrix_or_empty_shape(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="two-dimensional|nonempty"):
        numpy_matrix_frobenius(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((float("nan"),),), dtype=np.float64),
        np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128),
    ],
)
def test_rejects_nonfinite_entries(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="finite"):
        numpy_matrix_frobenius(matrix)


def test_finite_entry_power_overflow_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(((1.0e308,),), dtype=np.float64)

    with pytest.raises(FloatingPointError, match="power term became nonfinite"):
        numpy_matrix_frobenius(matrix)


def test_finite_power_sum_overflow_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(((1.0e154, 1.0e154),), dtype=np.float64)

    with pytest.raises(FloatingPointError, match="power became nonfinite"):
        numpy_matrix_frobenius(matrix)
