"""Integration tests for SciPy CSR Frobenius values."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from projectkoios.numerics.linalg.frobenius import ComplexMatrixFrobenius
from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.scipy.frobenius import (
    ScipyCsrFrobeniusNorm,
    scipy_csr_matrix_frobenius,
)
from projectkoios.numerics.scipy.typing import CsrMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected_power"),
    [
        (
            sp.csr_matrix(np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64)),
            25.0,
        ),
        (
            sp.csr_array(
                np.asarray(
                    (
                        (3.0 + 4.0j, 0.0 + 0.0j),
                        (1.0 - 1.0j, 0.0 + 1.0j),
                    ),
                    dtype=np.complex128,
                )
            ),
            28.0,
        ),
    ],
)
def test_detailed_operation_evaluates_both_csr_representations(
    matrix: CsrMatrix,
    expected_power: float,
) -> None:
    result = scipy_csr_matrix_frobenius(matrix)

    assert result == ComplexMatrixFrobenius(
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        expected_power,
        expected_power**0.5,
    )


def test_evaluator_reports_semantics_and_returns_scalar_norm() -> None:
    matrix: CsrMatrix = sp.csr_array(np.asarray(((3.0, 4.0),), dtype=np.float64))
    evaluation = ScipyCsrFrobeniusNorm(matrix)

    result = evaluation.result()

    assert evaluation.matrix_norm is MatrixNormEnum.frobenius
    assert result == 5.0
    assert type(result) is float


def test_private_canonicalization_reduces_represented_values_without_mutation() -> None:
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((2.0, 3.0, 0.0, 4.0), dtype=np.float64),
            np.asarray((1, 1, 0, 0), dtype=np.int64),
            np.asarray((0, 3, 4), dtype=np.int64),
        ),
        shape=(2, 2),
    )
    data_before = matrix.data.copy()
    indices_before = matrix.indices.copy()
    indptr_before = matrix.indptr.copy()
    canonical_before = matrix.has_canonical_format
    sorted_before = matrix.has_sorted_indices

    result = scipy_csr_matrix_frobenius(matrix)

    assert result.power == 41.0
    assert np.array_equal(matrix.data, data_before)
    assert np.array_equal(matrix.indices, indices_before)
    assert np.array_equal(matrix.indptr, indptr_before)
    assert matrix.has_canonical_format is canonical_before
    assert matrix.has_sorted_indices is sorted_before


def test_zero_matrix_has_exact_zero_power_and_norm() -> None:
    matrix: CsrMatrix = sp.csr_matrix((2, 3), dtype=np.complex128)

    result = scipy_csr_matrix_frobenius(matrix)

    assert result.power == 0.0
    assert result.norm == 0.0


def test_large_implicit_zero_dimension_is_not_densified() -> None:
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((2.0,), dtype=np.float64),
            np.asarray((999_999_999,), dtype=np.int64),
            np.asarray((0, 1), dtype=np.int64),
        ),
        shape=(1, 1_000_000_000),
    )

    result = scipy_csr_matrix_frobenius(matrix)

    assert result.column_count == 1_000_000_000
    assert result.power == 4.0
    assert result.norm == 2.0


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray((1.0, 2.0), dtype=np.float64)),
        sp.csr_array((0, 2), dtype=np.float64),
        sp.csr_matrix((2, 0), dtype=np.complex128),
    ],
)
def test_rejects_non_matrix_or_empty_shape(matrix: CsrMatrix) -> None:
    with pytest.raises(ValueError, match="two-dimensional|nonempty"):
        scipy_csr_matrix_frobenius(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray(((1.0,),), dtype=np.float32)),
        sp.csr_matrix(np.asarray(((1,),), dtype=np.int64)),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: CsrMatrix) -> None:
    with pytest.raises(TypeError, match="dtype"):
        scipy_csr_matrix_frobenius(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray(((float("nan"),),), dtype=np.float64)),
        sp.csr_matrix(
            np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128)
        ),
    ],
)
def test_rejects_nonfinite_source_entries(matrix: CsrMatrix) -> None:
    with pytest.raises(ValueError, match="finite"):
        scipy_csr_matrix_frobenius(matrix)


def test_duplicate_consolidation_overflow_fails_closed() -> None:
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((1.0e308, 1.0e308), dtype=np.float64),
            np.asarray((0, 0), dtype=np.int64),
            np.asarray((0, 2), dtype=np.int64),
        ),
        shape=(1, 1),
    )

    with pytest.raises(FloatingPointError, match="consolidation became nonfinite"):
        scipy_csr_matrix_frobenius(matrix)


def test_finite_entry_power_overflow_fails_closed() -> None:
    matrix: CsrMatrix = sp.csr_array(np.asarray(((1.0e308,),), dtype=np.float64))

    with pytest.raises(FloatingPointError, match="power term became nonfinite"):
        scipy_csr_matrix_frobenius(matrix)


def test_finite_power_sum_overflow_fails_closed() -> None:
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(((1.0e154, 1.0e154),), dtype=np.float64)
    )

    with pytest.raises(FloatingPointError, match="power became nonfinite"):
        scipy_csr_matrix_frobenius(matrix)
