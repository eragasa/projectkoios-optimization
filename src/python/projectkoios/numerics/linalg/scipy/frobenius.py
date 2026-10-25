"""SciPy CSR Frobenius power and norm for binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np
import scipy.sparse

from projectkoios.numerics.scipy.typing import CsrMatrix

from ..frobenius import _frobenius_power
from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class ScipyCsrFrobeniusPower:
    """Evaluate the squared Frobenius norm of one binary64 SciPy CSR value."""

    __slots__ = ("_matrix",)

    def __init__(self, matrix: CsrMatrix, /) -> None:
        """Retain the CSR matrix or array evaluated by :meth:`result`."""
        self._matrix = matrix

    def result(self) -> float:
        """Validate the retained CSR value and return its Frobenius power.

        Returns
        -------
        float
            Exact built-in finite nonnegative sum of squared represented magnitudes.

        Raises
        ------
        TypeError
            If the value is not a SciPy CSR matrix or array or has an unsupported exact
            dtype.
        ValueError
            If the CSR value is not a nonempty finite two-dimensional matrix.
        FloatingPointError
            If duplicate consolidation, a squared magnitude, or the accumulated power
            becomes nonfinite.
        """
        matrix = self._matrix
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

        # CSR permits duplicate coordinates, unsorted column indices, and explicit
        # zeros. Canonicalize a private copy before reducing represented values.
        canonical = matrix.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            canonical.sum_duplicates()
        if not bool(np.isfinite(canonical.data).all()):
            raise FloatingPointError("CSR duplicate consolidation became nonfinite")
        canonical.eliminate_zeros()
        canonical.sort_indices()

        return _frobenius_power(complex(entry) for entry in canonical.data)


@final
class ScipyCsrFrobeniusNorm(MatrixNormABC[CsrMatrix]):
    """Evaluate the Frobenius norm of one binary64 SciPy CSR value."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the Frobenius semantic identity."""
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: CsrMatrix, /) -> float:
        """Return the square root of the matrix's Frobenius power."""
        return math.sqrt(ScipyCsrFrobeniusPower(matrix).result())


__all__ = ["ScipyCsrFrobeniusNorm", "ScipyCsrFrobeniusPower"]
