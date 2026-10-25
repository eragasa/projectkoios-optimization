"""SciPy CSR Frobenius power and norm for binary64 matrices."""

from __future__ import annotations

import math
from typing import final

from projectkoios.numerics.scipy.typing import CsrMatrix

from ..frobenius import _frobenius_power
from ..matrix_norm import MatrixNormABC, MatrixNormEnum
from ._csr import _canonical_binary64_csr


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
        canonical = _canonical_binary64_csr(self._matrix)
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
