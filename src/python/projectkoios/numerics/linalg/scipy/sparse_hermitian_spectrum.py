"""SciPy backend for sparse partial Hermitian spectra."""

from __future__ import annotations

import math
from typing import final

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from ..sparse_hermitian_spectrum import (
    SparseComplexCsrMatrix,
    SparseHermitianPartialSpectrumDefinition,
    SparseHermitianPartialSpectrumSolver,
    SparseHermitianSpectrumSelection,
)

_SELECTION_TO_SCIPY = {
    SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC: "SA",
    SparseHermitianSpectrumSelection.LARGEST_ALGEBRAIC: "LA",
    SparseHermitianSpectrumSelection.SMALLEST_MAGNITUDE: "SM",
    SparseHermitianSpectrumSelection.LARGEST_MAGNITUDE: "LM",
}


def scipy_sparse_hermitian_csr_matrix(matrix: object, /) -> SparseComplexCsrMatrix:
    """Normalizes one SciPy-compatible sparse input to the owner CSR record.

    Parameters
    ----------
    matrix
        SciPy sparse matrix, sparse array, or two-dimensional array-like input.

    Returns
    -------
    SparseComplexCsrMatrix
        Canonical finite complex CSR record.

    Raises
    ------
    TypeError
        If SciPy cannot interpret the input as a two-dimensional numeric matrix.
    ValueError
        If the matrix is empty, nonsquare, or contains nonfinite values.
    """
    try:
        normalized = scipy.sparse.csr_array(matrix, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "matrix must be SciPy sparse-compatible numeric input"
        ) from error
    if normalized.ndim != 2 or normalized.shape[0] != normalized.shape[1]:
        raise ValueError("matrix must be square")
    if normalized.shape[0] == 0:
        raise ValueError("matrix must be nonempty")
    # SciPy inputs may contain duplicates, unsorted columns, or explicit zeros; owner
    # records remove those representation ambiguities before qualification.
    normalized.sum_duplicates()
    normalized.sort_indices()
    normalized.eliminate_zeros()
    if not np.all(np.isfinite(normalized.data)):
        raise ValueError("matrix entries must be finite")
    return SparseComplexCsrMatrix(
        int(normalized.shape[0]),
        tuple(int(value) for value in normalized.indptr),
        tuple(int(value) for value in normalized.indices),
        tuple(complex(value) for value in normalized.data),
    )


def _scipy_matrix(matrix: SparseComplexCsrMatrix, /) -> scipy.sparse.csr_array:
    """Constructs one SciPy CSR array from the owner record.

    Parameters
    ----------
    matrix
        Canonical owner CSR record.

    Returns
    -------
    scipy.sparse.csr_array
        SciPy sparse array with real or complex binary64 data.
    """
    # Preserving a real dtype lets SciPy use its Hermitian-specialized real ARPACK path,
    # which supports one more requested eigenvalue than the complex wrapper.
    is_real = all(value.imag == 0.0 for value in matrix.data)
    dtype = np.float64 if is_real else np.complex128
    data = (
        np.asarray(tuple(value.real for value in matrix.data), dtype=np.float64)
        if is_real
        else np.asarray(matrix.data, dtype=np.complex128)
    )
    return scipy.sparse.csr_array(
        (
            data,
            np.asarray(matrix.indices, dtype=np.int64),
            np.asarray(matrix.indptr, dtype=np.int64),
        ),
        shape=(matrix.dimension, matrix.dimension),
        dtype=dtype,
    )


@final
class ScipySparseHermitianPartialSpectrumSolver(SparseHermitianPartialSpectrumSolver):
    """Computes selected sparse Hermitian eigenvalues with SciPy ARPACK.

    Notes
    -----
    The backend uses ``scipy.sparse.linalg.eigsh`` and returns no public eigenvectors.
    A deterministic seeded start vector makes repeated owner calls reproducible for a
    fixed SciPy/LAPACK/ARPACK environment.
    """

    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        """Returns the API-level SciPy sparse operation identity.

        Returns
        -------
        str
            Stable backend identifier.
        """
        return "scipy.sparse.linalg.eigsh"

    @property
    def _backend_version(self) -> str:
        """Returns the runtime SciPy package version.

        Returns
        -------
        str
            SciPy version string.
        """
        return str(scipy.__version__)

    def _eigsh_validated(
        self,
        representative: SparseComplexCsrMatrix,
        definition: SparseHermitianPartialSpectrumDefinition,
        /,
    ) -> tuple[tuple[float, ...], float]:
        """Computes selected eigenvalues and maximum eigenpair residual.

        Parameters
        ----------
        representative
            Canonical represented-Hermitian sparse matrix.
        definition
            Validated selection and convergence controls.

        Returns
        -------
        tuple[tuple[float, ...], float]
            Ascending selected eigenvalues and maximum Euclidean residual norm.

        Raises
        ------
        ValueError
            If SciPy cannot request the selected count for a complex matrix.
        RuntimeError
            If ARPACK does not converge.
        FloatingPointError
            If eigenvalues, eigenvectors, or residual evidence become nonfinite.
        """
        matrix = _scipy_matrix(representative)
        if np.issubdtype(matrix.dtype, np.complexfloating) and (
            definition.eigenvalue_count >= definition.matrix.dimension - 1
        ):
            raise ValueError(
                "complex SciPy sparse problems require eigenvalue_count < dimension - 1"
            )
        # An explicit seed avoids ARPACK's implicit process-global random start while a
        # random-looking vector avoids predictable orthogonality to structured modes.
        rng = np.random.default_rng(definition.random_seed)
        start = rng.standard_normal(definition.matrix.dimension)
        if np.issubdtype(matrix.dtype, np.complexfloating):
            start = start + 1.0j * rng.standard_normal(definition.matrix.dimension)
        start_norm = float(np.linalg.norm(start))
        if not math.isfinite(start_norm) or start_norm == 0.0:
            raise FloatingPointError("sparse eigensolver start vector became invalid")
        start = start / start_norm
        try:
            # ARPACK returns eigenvectors internally so the owner can report residual
            # evidence without exposing basis-dependent vectors in the public result.
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                matrix,
                k=definition.eigenvalue_count,
                which=_SELECTION_TO_SCIPY[definition.selection],
                tol=definition.eigensolver_tolerance,
                maxiter=definition.maximum_iterations,
                ncv=definition.krylov_subspace_dimension,
                v0=start,
                return_eigenvectors=True,
            )
        except scipy.sparse.linalg.ArpackNoConvergence as error:
            # Partial ARPACK output is not silently presented as satisfaction of the
            # requested eigenvalue count.
            raise RuntimeError(
                "SciPy sparse Hermitian eigensolver did not converge"
            ) from error
        if not np.all(np.isfinite(eigenvalues)) or not np.all(
            np.isfinite(eigenvectors)
        ):
            raise FloatingPointError(
                "SciPy sparse Hermitian eigensolver returned nonfinite values"
            )
        # SciPy ordering follows ``which`` and is not a uniform public convention;
        # algebraic sorting makes downstream interpretation independent of selection.
        order = np.argsort(eigenvalues)
        ordered_values = np.asarray(eigenvalues[order], dtype=np.float64)
        ordered_vectors = eigenvectors[:, order]
        # Residuals use the same represented-Hermitian matrix supplied to ARPACK, which
        # keeps the reported evidence correlated with the actual numerical problem.
        residuals = (
            matrix @ ordered_vectors - ordered_vectors * ordered_values[np.newaxis, :]
        )
        residual_norms = np.linalg.norm(residuals, axis=0)
        residual = float(np.max(residual_norms))
        if not math.isfinite(residual):
            raise FloatingPointError("sparse eigenpair residual became nonfinite")
        return tuple(float(value) for value in ordered_values), residual


__all__ = [
    "ScipySparseHermitianPartialSpectrumSolver",
    "scipy_sparse_hermitian_csr_matrix",
]
