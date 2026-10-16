"""Integration tests for SciPy sparse partial Hermitian spectra."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.scipy.sparse_hermitian_spectrum import (
    ScipySparseHermitianPartialSpectrumSolver,
    scipy_sparse_hermitian_csr_matrix,
)
from projectkoios.numerics.linalg.sparse_hermitian_spectrum import (
    SparseHermitianPartialSpectrumDefinition,
    SparseHermitianSpectrumSelection,
)

pytestmark = pytest.mark.integration


def _definition(
    matrix: object,
    count: int,
    selection: SparseHermitianSpectrumSelection,
) -> SparseHermitianPartialSpectrumDefinition:
    return SparseHermitianPartialSpectrumDefinition(
        matrix=scipy_sparse_hermitian_csr_matrix(matrix),
        eigenvalue_count=count,
        selection=selection,
        hermiticity_tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
        eigensolver_tolerance=1.0e-12,
        random_seed=19,
    )


def test_scipy_adapter_canonicalizes_duplicates_and_explicit_zeros() -> None:
    matrix = scipy.sparse.coo_array(
        (
            np.asarray([1.0, 2.0, -2.0, 3.0]),
            (np.asarray([0, 0, 0, 2]), np.asarray([0, 1, 1, 2])),
        ),
        shape=(3, 3),
    )

    record = scipy_sparse_hermitian_csr_matrix(matrix)

    assert record.indptr == (0, 1, 1, 2)
    assert record.indices == (0, 2)
    assert record.data == (1.0 + 0.0j, 3.0 + 0.0j)


@pytest.mark.parametrize(
    "selection, expected",
    [
        (SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC, (1.0, 2.0)),
        (SparseHermitianSpectrumSelection.LARGEST_ALGEBRAIC, (5.0, 8.0)),
        (SparseHermitianSpectrumSelection.SMALLEST_MAGNITUDE, (1.0, 2.0)),
        (SparseHermitianSpectrumSelection.LARGEST_MAGNITUDE, (5.0, 8.0)),
    ],
)
def test_scipy_sparse_selection_rules_return_ascending_values(
    selection: SparseHermitianSpectrumSelection,
    expected: tuple[float, float],
) -> None:
    matrix = scipy.sparse.diags_array([1.0, 2.0, 3.0, 5.0, 8.0], format="csr")

    result = ScipySparseHermitianPartialSpectrumSolver().solve(
        _definition(matrix, 2, selection)
    )

    np.testing.assert_allclose(result.eigenvalues, expected, rtol=0.0, atol=1.0e-11)
    assert result.residual_maximum_absolute_norm <= 1.0e-10
    assert result.backend_identifier == "scipy.sparse.linalg.eigsh"


def test_scipy_sparse_complex_problem_is_supported() -> None:
    dense = np.asarray(
        [
            [3.0, 1.0j, 0.0, 0.0],
            [-1.0j, 4.0, 0.5, 0.0],
            [0.0, 0.5, 6.0, 1.0 - 0.5j],
            [0.0, 0.0, 1.0 + 0.5j, 9.0],
        ],
        dtype=np.complex128,
    )
    result = ScipySparseHermitianPartialSpectrumSolver().solve(
        _definition(
            scipy.sparse.csr_array(dense),
            2,
            SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
        )
    )

    expected = np.linalg.eigvalsh(dense)[:2]
    np.testing.assert_allclose(result.eigenvalues, expected, rtol=0.0, atol=1.0e-10)
    assert result.residual_maximum_absolute_norm <= 1.0e-9


def test_sparse_backend_translates_arpack_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonconvergent(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise scipy.sparse.linalg.ArpackNoConvergence(
            "failed",
            np.asarray([], dtype=np.float64),
            np.empty((5, 0), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "eigsh", nonconvergent)
    matrix = scipy.sparse.diags_array([1.0, 2.0, 3.0, 4.0, 5.0], format="csr")

    with pytest.raises(RuntimeError):
        ScipySparseHermitianPartialSpectrumSolver().solve(
            _definition(
                matrix,
                2,
                SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            )
        )


def test_sparse_backend_rejects_nonfinite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        return (
            np.asarray([1.0, np.nan], dtype=np.float64),
            np.ones((5, 2), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "eigsh", nonfinite)
    matrix = scipy.sparse.diags_array([1.0, 2.0, 3.0, 4.0, 5.0], format="csr")

    with pytest.raises(FloatingPointError):
        ScipySparseHermitianPartialSpectrumSolver().solve(
            _definition(
                matrix,
                2,
                SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            )
        )


def test_complex_count_constraint_is_explicit() -> None:
    matrix = scipy.sparse.csr_array(
        np.asarray(
            [[1.0, 1.0j, 0.0], [-1.0j, 2.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=np.complex128,
        )
    )

    with pytest.raises(ValueError, match="dimension - 1"):
        ScipySparseHermitianPartialSpectrumSolver().solve(
            _definition(
                matrix,
                2,
                SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            )
        )
