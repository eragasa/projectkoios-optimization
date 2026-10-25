"""SciPy CSR Frobenius power and norm for binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np
import scipy.sparse

from projectkoios.numerics.scipy.typing import CsrMatrix

from ..frobenius import ComplexMatrixFrobenius, _frobenius_power
from ..matrix_norm import MatrixNormABC, MatrixNormEnum


def scipy_csr_matrix_frobenius(matrix: CsrMatrix, /) -> ComplexMatrixFrobenius:
    """Compute Frobenius power and norm from a SciPy CSR matrix or array.

    Parameters
    ----------
    matrix
        Nonempty finite two-dimensional SciPy CSR matrix or array with exact float64 or
        complex128 storage. The operation copies before canonicalization and never
        densifies or mutates the caller's object.

    Returns
    -------
    ComplexMatrixFrobenius
        Matrix dimensions, raw Frobenius power, and its binary64 square root.

    Raises
    ------
    TypeError
        If the value is not a SciPy CSR matrix or array or has an unsupported exact
        dtype.
    ValueError
        If the CSR value is not a nonempty finite matrix.
    FloatingPointError
        If duplicate consolidation, a squared magnitude, or the accumulated raw power
        becomes nonfinite.
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

    # CSR permits duplicate coordinates, unsorted column indices, and explicit zeros.
    # Canonicalize a private copy before reducing the represented matrix values.
    canonical = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        canonical.sum_duplicates()
    if not bool(np.isfinite(canonical.data).all()):
        raise FloatingPointError("CSR duplicate consolidation became nonfinite")
    canonical.eliminate_zeros()
    canonical.sort_indices()

    power = _frobenius_power(complex(entry) for entry in canonical.data)
    return ComplexMatrixFrobenius(
        int(row_count),
        int(column_count),
        power,
        math.sqrt(power),
    )


@final
class ScipyCsrFrobeniusNorm(MatrixNormABC[CsrMatrix]):
    """Evaluate the Frobenius norm of one SciPy float64 or complex128 CSR value."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the Frobenius semantic identity."""
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: CsrMatrix, /) -> float:
        """Return the norm from the detailed SciPy CSR Frobenius operation."""
        return scipy_csr_matrix_frobenius(matrix).norm


__all__ = ["ScipyCsrFrobeniusNorm", "scipy_csr_matrix_frobenius"]
