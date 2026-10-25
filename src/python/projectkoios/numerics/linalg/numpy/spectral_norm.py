"""Evaluate spectral norms of dense NumPy binary64 matrices.

The spectral norm is the largest singular value of a general rectangular matrix. This
module owns the exact NumPy representation boundary and the complete dense
``numpy.linalg.svdvals`` call; it does not own units, sparse iteration, scaling policy,
or scientific interpretation.
"""

from __future__ import annotations

from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import DenseMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum


@final
class NumpySpectralNorm(MatrixNormABC[DenseMatrix]):
    """Evaluate the largest singular value of one dense binary64 matrix.

    Parameters
    ----------
    matrix
        Exact nonempty finite two-dimensional NumPy ndarray with exact ``float64`` or
        ``complex128`` storage. The evaluation retains and reads the array without
        mutation or dtype conversion.

    Notes
    -----
    The operation uses a complete dense singular-value decomposition. It does not apply
    an iterative tolerance, scaling fallback, or eigenvalue-magnitude shortcut.
    """

    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the spectral-norm semantic identity.

        Returns
        -------
        MatrixNormEnum
            ``MatrixNormEnum.spectral``.
        """
        return MatrixNormEnum.spectral

    def _evaluate(self, matrix: DenseMatrix, /) -> float:
        """Validate one dense matrix and return its largest singular value.

        Parameters
        ----------
        matrix
            Retained dense matrix supplied at construction.

        Returns
        -------
        float
            Exact built-in finite nonnegative largest singular value.

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
        # Static NumPy annotations do not encode exact runtime class, dimensionality,
        # emptiness, or finiteness. Validate those public preconditions before allowing
        # ``svdvals`` to interpret higher dimensions as a matrix stack or coerce dtype.
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
            # Finite source validation precedes the backend call. Suppress only floating
            # warnings from a finite problem whose mathematical result is not
            # representable; the explicit output check below classifies that failure.
            with np.errstate(over="ignore", invalid="ignore"):
                singular_values = np.linalg.svdvals(matrix)
        except np.linalg.LinAlgError as error:
            # Translate exactly the backend's dense decomposition nonconvergence while
            # allowing unrelated programming or representation failures to propagate.
            raise RuntimeError(
                "NumPy singular-value decomposition did not converge"
            ) from error
        if not bool(np.isfinite(singular_values).all()):
            raise FloatingPointError(
                "NumPy singular-value decomposition returned nonfinite values"
            )

        # NumPy returns complete singular values in descending order, so index zero is
        # the induced Euclidean matrix norm for square and rectangular inputs alike.
        return float(singular_values[0])


__all__ = ["NumpySpectralNorm"]
