"""Integration tests for dense NumPy Frobenius powers."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.linalg.numpy.frobenius import NumpyFrobeniusPower
from projectkoios.numerics.numpy.typing import DenseMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
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
def test_evaluation_returns_scalar_power(
    matrix: DenseMatrix,
    expected: float,
) -> None:
    result = NumpyFrobeniusPower(matrix).result()

    assert result == expected
    assert type(result) is float


def test_logical_row_major_reduction_accepts_storage_layouts_and_views() -> None:
    fortran_matrix: DenseMatrix = np.asfortranarray(
        np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    )
    source = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    noncontiguous_view: DenseMatrix = source[:, ::2]

    assert NumpyFrobeniusPower(fortran_matrix).result() == 91.0
    assert NumpyFrobeniusPower(noncontiguous_view).result() == 220.0


def test_evaluation_does_not_mutate_writable_or_read_only_input() -> None:
    writable: DenseMatrix = np.asarray(((1.0 + 2.0j, 3.0 + 4.0j),), dtype=np.complex128)
    writable_before = writable.copy()
    read_only: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)
    read_only.setflags(write=False)

    writable_result = NumpyFrobeniusPower(writable).result()
    read_only_result = NumpyFrobeniusPower(read_only).result()

    assert np.array_equal(writable, writable_before)
    assert writable.flags.writeable
    assert not read_only.flags.writeable
    assert writable_result == 30.0
    assert read_only_result == 25.0


def test_zero_matrix_has_exact_zero_power() -> None:
    matrix: DenseMatrix = np.zeros((2, 3), dtype=np.float64)

    result = NumpyFrobeniusPower(matrix).result()

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
        NumpyFrobeniusPower(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((1.0,),), dtype=np.float32),
        np.asarray(((1,),), dtype=np.int64),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: DenseMatrix) -> None:
    with pytest.raises(TypeError, match="dtype"):
        NumpyFrobeniusPower(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((float("nan"),),), dtype=np.float64),
        np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128),
    ],
)
def test_rejects_nonfinite_entries(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumpyFrobeniusPower(matrix).result()


def test_finite_entry_power_overflow_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(((1.0e308,),), dtype=np.float64)

    with pytest.raises(FloatingPointError, match="power term became nonfinite"):
        NumpyFrobeniusPower(matrix).result()


def test_finite_power_sum_overflow_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(((1.0e154, 1.0e154),), dtype=np.float64)

    with pytest.raises(FloatingPointError, match="power became nonfinite"):
        NumpyFrobeniusPower(matrix).result()
