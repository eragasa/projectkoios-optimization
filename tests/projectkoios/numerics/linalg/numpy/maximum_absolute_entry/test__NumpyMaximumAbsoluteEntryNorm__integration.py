"""Integration tests for dense NumPy maximum-absolute-entry norms."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.numpy.maximum_absolute_entry import (
    NumpyMaximumAbsoluteEntryNorm,
)
from projectkoios.numerics.numpy.typing import DenseMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (np.asarray(((-2.0, 4.0, -3.0),), dtype=np.float64), 4.0),
        (
            np.asarray(
                ((3.0 + 4.0j, -4.0 + 0.0j), (1.0 - 1.0j, 0.0 + 0.0j)),
                dtype=np.complex128,
            ),
            5.0,
        ),
    ],
)
def test_evaluation_accepts_supported_dense_storage(
    matrix: DenseMatrix,
    expected: float,
) -> None:
    result = NumpyMaximumAbsoluteEntryNorm(matrix).result()

    assert result == expected
    assert type(result) is float


def test_evaluation_reports_maximum_absolute_entry_semantics() -> None:
    matrix: DenseMatrix = np.asarray(((3.0 + 4.0j, 4.0j),), dtype=np.complex128)
    evaluation = NumpyMaximumAbsoluteEntryNorm(matrix)

    result = evaluation.result()

    assert evaluation.matrix_norm is MatrixNormEnum.maximum_absolute_entry
    assert result == 5.0


def test_accepts_storage_layouts_and_noncontiguous_views() -> None:
    fortran_matrix: DenseMatrix = np.asfortranarray(
        np.asarray(((-1.0, 2.0, -7.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    )
    source = np.arange(-6.0, 6.0, dtype=np.float64).reshape(3, 4)
    noncontiguous_view: DenseMatrix = source[:, ::2]

    assert NumpyMaximumAbsoluteEntryNorm(fortran_matrix).result() == 7.0
    assert NumpyMaximumAbsoluteEntryNorm(noncontiguous_view).result() == 6.0


def test_operation_does_not_mutate_writable_or_read_only_input() -> None:
    writable: DenseMatrix = np.asarray(((1.0 + 2.0j, 3.0 + 4.0j),), dtype=np.complex128)
    writable_before = writable.copy()
    read_only: DenseMatrix = np.asarray(((-3.0, 2.0),), dtype=np.float64)
    read_only.setflags(write=False)

    writable_result = NumpyMaximumAbsoluteEntryNorm(writable).result()
    read_only_result = NumpyMaximumAbsoluteEntryNorm(read_only).result()

    assert np.array_equal(writable, writable_before)
    assert writable.flags.writeable
    assert not read_only.flags.writeable
    assert writable_result == 5.0
    assert read_only_result == 3.0


def test_zero_matrix_has_exact_zero_norm() -> None:
    matrix: DenseMatrix = np.zeros((2, 3), dtype=np.complex128)

    result = NumpyMaximumAbsoluteEntryNorm(matrix).result()

    assert result == 0.0


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
        NumpyMaximumAbsoluteEntryNorm(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((float("nan"),),), dtype=np.float64),
        np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128),
    ],
)
def test_rejects_nonfinite_entries(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumpyMaximumAbsoluteEntryNorm(matrix).result()


def test_finite_complex_entry_magnitude_overflow_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(
        ((complex(1.3e308, 1.3e308),),), dtype=np.complex128
    )

    with pytest.raises(FloatingPointError, match="became nonfinite"):
        NumpyMaximumAbsoluteEntryNorm(matrix).result()
