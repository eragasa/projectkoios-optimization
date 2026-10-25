"""SciPy CSR maximum-absolute-entry norms for binary64 matrices."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.scipy.typing import CsrMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum
from ._csr import _canonical_binary64_csr


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
        canonical = _canonical_binary64_csr(matrix)

        # Initial zero includes implicit sparse zeros and defines the all-zero case.
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.max(np.abs(canonical.data), initial=0.0))
        if not math.isfinite(result):
            raise FloatingPointError("maximum absolute entry became nonfinite")
        return result


__all__ = ["ScipyCsrMaximumAbsoluteEntryNorm"]
