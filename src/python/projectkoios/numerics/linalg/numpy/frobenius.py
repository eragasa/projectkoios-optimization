"""NumPy Frobenius power and norm for dense binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..frobenius import _frobenius_power
from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class NumpyFrobeniusPower:
    """Evaluate the squared Frobenius norm of one dense binary64 matrix."""

    __slots__ = ("_matrix",)

    def __init__(self, matrix: DenseMatrix, /) -> None:
        """Retain the dense matrix evaluated by :meth:`result`."""
        self._matrix = matrix

    def result(self) -> float:
        """Validate the retained matrix and return its Frobenius power.

        Returns
        -------
        float
            Exact built-in finite nonnegative sum of squared entry magnitudes.

        Raises
        ------
        TypeError
            If the value is not an exact NumPy array or has an unsupported exact dtype.
        ValueError
            If the array is not a nonempty finite two-dimensional matrix.
        FloatingPointError
            If a squared magnitude or the accumulated power becomes nonfinite.
        """
        matrix = self._matrix
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

        # ``flat`` uses logical C-index order for contiguous arrays and views alike.
        # Scalar conversion provides the same built-in complex arithmetic used by the
        # dependency-free representation without copying or normalizing the matrix.
        return _frobenius_power(complex(entry) for entry in matrix.flat)


@final
class NumpyFrobeniusNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the Frobenius norm of one dense binary64 matrix."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the Frobenius semantic identity."""
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Return the square root of the matrix's Frobenius power."""
        return math.sqrt(NumpyFrobeniusPower(matrix).result())


__all__ = ["NumpyFrobeniusNorm", "NumpyFrobeniusPower"]
