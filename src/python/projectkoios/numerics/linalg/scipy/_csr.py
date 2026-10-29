"""Private canonicalization for exact binary64 SciPy CSR matrices."""

from __future__ import annotations

import numpy as np
import scipy.sparse

from projectkoios.numerics.scipy.typing import CsrMatrix


def _canonical_binary64_csr(matrix: CsrMatrix, /) -> CsrMatrix:
    """Validate one CSR matrix and return an independent canonical copy.

    Parameters
    ----------
    matrix
        SciPy CSR matrix or array with exact float64 or complex128 storage.

    Returns
    -------
    CsrMatrix
        Independent CSR value with duplicates consolidated, explicit zeros removed, and
        column indices sorted.

    Raises
    ------
    TypeError
        If the value is not a SciPy CSR matrix or array or has an unsupported exact
        dtype.
    ValueError
        If the matrix is not nonempty, two-dimensional, and finite before duplicate
        consolidation.
    FloatingPointError
        If duplicate consolidation produces a nonfinite represented entry.

    Notes
    -----
    Validation precedes copying so unsupported inputs fail before SciPy interprets them.
    Every mutation applies only to the private copy.
    """
    if not isinstance(matrix, scipy.sparse.csr_matrix | scipy.sparse.csr_array):
        raise TypeError("matrix must be a scipy.sparse.csr_matrix or csr_array")
    if matrix.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("matrix dtype must be exactly float64 or complex128")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    row_count, column_count = matrix.shape
    if row_count == 0 or column_count == 0:
        raise ValueError("matrix must be nonempty")
    if not bool(np.isfinite(matrix.data).all()):
        raise ValueError("matrix entries must be finite")

    # Duplicate coordinates, explicit zeros, and unsorted indices are valid CSR storage
    # states. Normalize them on a private copy before any represented-value algorithm.
    canonical = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        canonical.sum_duplicates()
    if not bool(np.isfinite(canonical.data).all()):
        raise FloatingPointError("CSR duplicate consolidation became nonfinite")
    canonical.eliminate_zeros()
    canonical.sort_indices()
    return canonical
