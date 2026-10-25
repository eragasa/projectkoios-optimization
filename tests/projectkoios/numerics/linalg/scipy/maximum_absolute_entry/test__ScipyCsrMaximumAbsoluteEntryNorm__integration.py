"""Integration tests for SciPy CSR maximum-absolute-entry norms."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.scipy.maximum_absolute_entry import (
    ScipyCsrMaximumAbsoluteEntryNorm,
)
from projectkoios.numerics.scipy.typing import CsrMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (
            sp.csr_matrix(np.asarray(((-2.0, 4.0, -3.0),), dtype=np.float64)),
            4.0,
        ),
        (
            sp.csr_array(
                np.asarray(
                    ((3.0 + 4.0j, -4.0 + 0.0j), (0.0j, 1.0 - 1.0j)),
                    dtype=np.complex128,
                )
            ),
            5.0,
        ),
    ],
)
def test_evaluator_accepts_both_csr_representations(
    matrix: CsrMatrix,
    expected: float,
) -> None:
    result = ScipyCsrMaximumAbsoluteEntryNorm()(matrix)

    assert result == expected
    assert type(result) is float


def test_evaluator_reports_maximum_absolute_entry_semantics() -> None:
    evaluator = ScipyCsrMaximumAbsoluteEntryNorm()
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(((3.0 + 4.0j, 4.0j),), dtype=np.complex128)
    )

    result = evaluator(matrix)

    assert evaluator.matrix_norm is MatrixNormEnum.maximum_absolute_entry
    assert result == 5.0


def test_private_canonicalization_uses_represented_values_without_mutation() -> None:
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((2.0, 3.0, 0.0, -4.0), dtype=np.float64),
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

    result = ScipyCsrMaximumAbsoluteEntryNorm()(matrix)

    assert result == 5.0
    assert np.array_equal(matrix.data, data_before)
    assert np.array_equal(matrix.indices, indices_before)
    assert np.array_equal(matrix.indptr, indptr_before)
    assert matrix.has_canonical_format is canonical_before
    assert matrix.has_sorted_indices is sorted_before


def test_duplicate_cancellation_and_all_zero_storage_return_exact_zero() -> None:
    evaluator = ScipyCsrMaximumAbsoluteEntryNorm()
    cancellation: CsrMatrix = sp.csr_array(
        (
            np.asarray((5.0, -5.0), dtype=np.float64),
            np.asarray((0, 0), dtype=np.int64),
            np.asarray((0, 2), dtype=np.int64),
        ),
        shape=(1, 1),
    )
    empty: CsrMatrix = sp.csr_matrix((2, 3), dtype=np.complex128)

    assert evaluator(cancellation) == 0.0
    assert evaluator(empty) == 0.0


def test_large_implicit_zero_dimension_is_not_densified() -> None:
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((-2.0,), dtype=np.float64),
            np.asarray((999_999_999,), dtype=np.int64),
            np.asarray((0, 1), dtype=np.int64),
        ),
        shape=(1, 1_000_000_000),
    )

    result = ScipyCsrMaximumAbsoluteEntryNorm()(matrix)

    assert result == 2.0


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
        ScipyCsrMaximumAbsoluteEntryNorm()(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray(((1.0,),), dtype=np.float32)),
        sp.csr_matrix(np.asarray(((1,),), dtype=np.int64)),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: CsrMatrix) -> None:
    with pytest.raises(TypeError, match="dtype"):
        ScipyCsrMaximumAbsoluteEntryNorm()(matrix)


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
        ScipyCsrMaximumAbsoluteEntryNorm()(matrix)


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
        ScipyCsrMaximumAbsoluteEntryNorm()(matrix)


def test_finite_complex_entry_magnitude_overflow_fails_closed() -> None:
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(((complex(1.3e308, 1.3e308),),), dtype=np.complex128)
    )

    with pytest.raises(FloatingPointError, match="became nonfinite"):
        ScipyCsrMaximumAbsoluteEntryNorm()(matrix)
