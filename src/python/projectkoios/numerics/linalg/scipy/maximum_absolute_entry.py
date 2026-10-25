"""SciPy CSR maximum-absolute-entry norms for binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np
import scipy.sparse

from projectkoios.numerics.scipy.typing import CsrMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class ScipyCsrMaximumAbsoluteEntryNorm(MatrixNormABC[CsrMatrix]):
    """Evaluate the maximum absolute entry of one binary64 SciPy CSR value."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the maximum-absolute-entry semantic identity."""
        return MatrixNormEnum.maximum_absolute_entry

    def _evaluate(self, matrix: CsrMatrix, /) -> float:
        """Validate one CSR value and return its maximum represented magnitude."""
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

        # Duplicate coordinates denote one represented value and therefore must be
        # summed before taking a maximum. A private copy preserves caller ownership.
        canonical = matrix.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            canonical.sum_duplicates()
        if not bool(np.isfinite(canonical.data).all()):
            raise FloatingPointError("CSR duplicate consolidation became nonfinite")
        canonical.eliminate_zeros()
        canonical.sort_indices()

        # Initial zero includes implicit sparse zeros and defines the all-zero case.
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.max(np.abs(canonical.data), initial=0.0))
        if not math.isfinite(result):
            raise FloatingPointError("maximum absolute entry became nonfinite")
        return result


__all__ = ["ScipyCsrMaximumAbsoluteEntryNorm"]
