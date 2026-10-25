"""NumPy Frobenius power and norm for dense binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..frobenius import ComplexMatrixFrobenius, _frobenius_power
from ..matrix_norm import MatrixNormABC, MatrixNormEnum


def numpy_matrix_frobenius(matrix: DenseMatrix, /) -> ComplexMatrixFrobenius:
    """Compute Frobenius power and norm from a dense NumPy matrix.

    Parameters
    ----------
    matrix
        Nonempty finite two-dimensional exact NumPy array with exact float64 or
        complex128 storage. The operation reads but does not mutate the array.

    Returns
    -------
    ComplexMatrixFrobenius
        Matrix dimensions, raw Frobenius power, and its binary64 square root.

    Raises
    ------
    TypeError
        If the value is not an exact NumPy array or has an unsupported exact dtype.
    ValueError
        If the array is not a nonempty finite matrix.
    FloatingPointError
        If a squared magnitude or the accumulated raw power becomes nonfinite.
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

    # ``flat`` uses logical C-index order for contiguous arrays and views alike. Scalar
    # conversion provides the same built-in complex arithmetic used by the dependency-
    # free representation without copying or normalizing the caller's matrix.
    power = _frobenius_power(complex(entry) for entry in matrix.flat)
    return ComplexMatrixFrobenius(
        int(row_count),
        int(column_count),
        power,
        math.sqrt(power),
    )


@final
class NumpyFrobeniusNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the Frobenius norm of one dense float64 or complex128 matrix."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the Frobenius semantic identity."""
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Return the norm from the detailed NumPy Frobenius operation."""
        return numpy_matrix_frobenius(matrix).norm


__all__ = ["NumpyFrobeniusNorm", "numpy_matrix_frobenius"]
