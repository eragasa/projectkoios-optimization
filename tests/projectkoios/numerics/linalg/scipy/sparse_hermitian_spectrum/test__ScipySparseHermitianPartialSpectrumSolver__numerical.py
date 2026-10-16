"""Bounded numerical verification for SciPy sparse Hermitian spectra."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

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

pytestmark = pytest.mark.numerical


def test_seeded_sparse_cases_match_dense_reference_and_residual_bounds() -> None:
    rng = np.random.default_rng(20261018)
    solver = ScipySparseHermitianPartialSpectrumSolver()

    for case in range(32):
        dimension = 6 + case % 5
        raw = np.zeros((dimension, dimension), dtype=np.complex128)
        mask = rng.random((dimension, dimension)) < 0.2
        values = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
            size=(dimension, dimension)
        )
        raw[mask] = values[mask]
        matrix = (raw + raw.conj().T) * 0.5
        matrix += np.diag(rng.normal(size=dimension))
        count = min(3, dimension - 2)
        definition = SparseHermitianPartialSpectrumDefinition(
            matrix=scipy_sparse_hermitian_csr_matrix(scipy.sparse.csr_array(matrix)),
            eigenvalue_count=count,
            selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            hermiticity_tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
            eigensolver_tolerance=1.0e-12,
            maximum_iterations=5000,
            random_seed=case,
        )

        result = solver.solve(definition)
        expected = np.linalg.eigvalsh(matrix)[:count]
        scale = max(1.0, float(np.max(np.abs(expected))))

        np.testing.assert_allclose(
            result.eigenvalues,
            expected,
            rtol=0.0,
            atol=2.0e-10 * scale,
        )
        assert result.residual_maximum_absolute_norm <= 2.0e-10 * scale


def test_fixed_seed_repeats_sparse_values_and_residual_evidence() -> None:
    matrix = scipy.sparse.diags_array(
        (
            np.full(11, -1.0),
            np.arange(1.0, 13.0),
            np.full(11, -1.0),
        ),
        offsets=(-1, 0, 1),
        shape=(12, 12),
        format="csr",
    )
    definition = SparseHermitianPartialSpectrumDefinition(
        matrix=scipy_sparse_hermitian_csr_matrix(matrix),
        eigenvalue_count=3,
        selection=SparseHermitianSpectrumSelection.LARGEST_ALGEBRAIC,
        hermiticity_tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
        eigensolver_tolerance=1.0e-13,
        random_seed=73,
    )
    solver = ScipySparseHermitianPartialSpectrumSolver()

    first = solver.solve(definition)
    second = solver.solve(definition)

    assert first.eigenvalues == second.eigenvalues
    assert first.residual_maximum_absolute_norm == second.residual_maximum_absolute_norm
