"""Unit tests for canonical sparse Hermitian-spectrum records."""

from __future__ import annotations

from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.sparse_hermitian_spectrum import (
    SparseComplexCsrMatrix,
    SparseHermitianPartialSpectrumDefinition,
    SparseHermitianSpectrumSelection,
)

pytestmark = pytest.mark.unit


def _matrix() -> SparseComplexCsrMatrix:
    return SparseComplexCsrMatrix(
        4,
        (0, 2, 4, 6, 8),
        (0, 1, 0, 1, 2, 3, 2, 3),
        (
            2.0 + 0.0j,
            1.0j,
            -1.0j,
            3.0 + 0.0j,
            4.0 + 0.0j,
            0.5 + 0.0j,
            0.5 + 0.0j,
            5.0 + 0.0j,
        ),
    )


def test_csr_record_retains_canonical_sparse_storage() -> None:
    indptr = (0, 1, 2)
    indices = (0, 1)
    data = (2.0 + 0.0j, 3.0 + 0.0j)

    matrix = SparseComplexCsrMatrix(2, indptr, indices, data)

    assert matrix.dimension == 2
    assert matrix.nonzero_count == 2
    assert matrix.indptr is indptr
    assert matrix.indices is indices
    assert matrix.data is data


@pytest.mark.parametrize(
    "args, match",
    [
        ((0, (0,), (), ()), "dimension"),
        ((2, (0, 1), (0,), (1.0 + 0.0j,)), "indptr length"),
        ((2, (1, 1, 1), (), ()), "start at zero"),
        ((2, (0, 2, 1), (0,), (1.0 + 0.0j,)), "nondecreasing"),
        ((2, (0, 1, 1), (2,), (1.0 + 0.0j,)), "outside"),
        ((2, (0, 2, 2), (0, 0), (1.0 + 0.0j, 2.0 + 0.0j)), "strictly"),
        ((2, (0, 1, 1), (0,), (0.0 + 0.0j,)), "explicit zeros"),
        ((2, (0, 1, 1), (0,), (1.0,)), "exact built-in complex"),
    ],
)
def test_csr_record_rejects_noncanonical_storage(
    args: tuple[object, object, object, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        SparseComplexCsrMatrix(*cast(Any, args))


def test_definition_retains_explicit_sparse_selection_controls() -> None:
    definition = SparseHermitianPartialSpectrumDefinition(
        matrix=_matrix(),
        eigenvalue_count=2,
        selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
        hermiticity_tolerance=BatchedHermitianSpectrumTolerance(1.0e-12, 1.0e-12),
        eigensolver_tolerance=1.0e-10,
        maximum_iterations=500,
        krylov_subspace_dimension=4,
        random_seed=17,
    )

    assert definition.eigenvalue_count == 2
    assert definition.maximum_iterations == 500
    assert definition.krylov_subspace_dimension == 4
    assert definition.random_seed == 17


def test_definition_requires_a_strict_partial_spectrum() -> None:
    with pytest.raises(ValueError):
        SparseHermitianPartialSpectrumDefinition(
            matrix=_matrix(),
            eigenvalue_count=4,
            selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            hermiticity_tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
        )


def test_definition_requires_krylov_space_larger_than_requested_spectrum() -> None:
    with pytest.raises(ValueError):
        SparseHermitianPartialSpectrumDefinition(
            matrix=_matrix(),
            eigenvalue_count=2,
            selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            hermiticity_tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
            krylov_subspace_dimension=2,
        )
