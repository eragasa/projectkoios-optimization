"""NumPy maximum-absolute-entry norms for dense binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class NumpyMaximumAbsoluteEntryNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the maximum absolute entry of one dense binary64 matrix."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the maximum-absolute-entry semantic identity."""
        return MatrixNormEnum.maximum_absolute_entry

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Validate one dense matrix and return its maximum entry magnitude."""
        if type(matrix) is not np.ndarray:
            raise TypeError("matrix must be an exact numpy.ndarray")
        if matrix.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise TypeError("matrix dtype must be exactly float64 or complex128")
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        row_count, column_count = matrix.shape
        if row_count == 0 or column_count == 0:
            raise ValueError("matrix must be nonempty")
        if not bool(np.isfinite(matrix).all()):
            raise ValueError("matrix entries must be finite")

        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.max(np.abs(matrix)))
        if not math.isfinite(result):
            raise FloatingPointError("maximum absolute entry became nonfinite")
        return result


__all__ = ["NumpyMaximumAbsoluteEntryNorm"]
