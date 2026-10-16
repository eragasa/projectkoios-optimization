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
        (2.0, 1.0j, -1.0j, 3.0, 4.0, 0.5, 0.5, 5.0),
    )


def test_csr_record_retains_canonical_sparse_storage() -> None:
    matrix = _matrix()

    assert matrix.dimension == 4
    assert matrix.nonzero_count == 8
    assert matrix.indptr == (0, 2, 4, 6, 8)
    assert matrix.data[0] == 2.0 + 0.0j


@pytest.mark.parametrize(
    "args, match",
    [
        ((0, (0,), (), ()), "dimension"),
        ((2, (0, 1), (0,), (1.0,)), "indptr length"),
        ((2, (1, 1, 1), (), ()), "start at zero"),
        ((2, (0, 2, 1), (0,), (1.0,)), "nondecreasing"),
        ((2, (0, 1, 1), (2,), (1.0,)), "outside"),
        ((2, (0, 2, 2), (0, 0), (1.0, 2.0)), "strictly"),
        ((2, (0, 1, 1), (0,), (0.0,)), "explicit zeros"),
    ],
)
def test_csr_record_rejects_noncanonical_storage(
    args: tuple[object, object, object, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        SparseComplexCsrMatrix(*args)


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


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"eigenvalue_count": 4}, "smaller"),
        ({"eigenvalue_count": 0}, "positive"),
        ({"selection": "smallest-algebraic"}, "selection"),
        ({"eigensolver_tolerance": -1.0}, "nonnegative"),
        ({"maximum_iterations": 0}, "positive"),
        ({"krylov_subspace_dimension": 2}, "must exceed"),
        ({"random_seed": -1}, "nonnegative"),
    ],
)
def test_definition_rejects_invalid_sparse_controls(
    overrides: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "matrix": _matrix(),
        "eigenvalue_count": 2,
        "selection": SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
        "hermiticity_tolerance": BatchedHermitianSpectrumTolerance(0.0, 0.0),
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        SparseHermitianPartialSpectrumDefinition(**cast(Any, values))
