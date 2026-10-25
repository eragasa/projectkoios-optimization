"""NumPy spectral norms for dense binary64 matrices."""

from __future__ import annotations

from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class NumpySpectralNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the largest singular value of one dense binary64 matrix."""

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the spectral-norm semantic identity."""
        return MatrixNormEnum.spectral

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Validate the retained matrix and return its largest singular value.

        Raises
        ------
        TypeError
            If the value is not an exact NumPy array or has an unsupported exact dtype.
        ValueError
            If the array is not a nonempty finite two-dimensional matrix.
        RuntimeError
            If the NumPy singular-value decomposition does not converge.
        FloatingPointError
            If NumPy returns a nonfinite singular value.
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

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                singular_values = np.linalg.svdvals(matrix)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "NumPy singular-value decomposition did not converge"
            ) from error
        if not bool(np.isfinite(singular_values).all()):
            raise FloatingPointError(
                "NumPy singular-value decomposition returned nonfinite values"
            )

        return float(singular_values[0])


__all__ = ["NumpySpectralNorm"]
