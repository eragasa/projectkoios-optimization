"""NumPy maximum-absolute-entry norms for dense binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum


def numpy_matrix_maximum_absolute_entry(matrix: DenseMatrix, /) -> float:
    """Return the maximum absolute entry of a dense NumPy matrix.

    Parameters
    ----------
    matrix
        Nonempty finite two-dimensional exact NumPy array with exact float64 or
        complex128 storage. The operation reads but does not mutate the array.

    Returns
    -------
    float
        Exact built-in finite nonnegative maximum entry magnitude.

    Raises
    ------
    TypeError
        If the value is not an exact NumPy array or has an unsupported exact dtype.
    ValueError
        If the array is not a nonempty finite matrix.
    FloatingPointError
        If magnitude evaluation becomes nonfinite.
    """
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


@final
class NumpyMaximumAbsoluteEntryNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the maximum absolute entry of one dense binary64 matrix."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the maximum-absolute-entry semantic identity."""
        return MatrixNormEnum.maximum_absolute_entry

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Return the detailed NumPy maximum-absolute-entry operation."""
        return numpy_matrix_maximum_absolute_entry(matrix)


__all__ = [
    "NumpyMaximumAbsoluteEntryNorm",
    "numpy_matrix_maximum_absolute_entry",
]
