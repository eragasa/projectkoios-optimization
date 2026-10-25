"""Shared precise interfaces for optional SciPy capability modules.

SciPy does not currently expose complete static types for the objects used here. These
interfaces describe only the operations that Project Koios consumes; they are typing
boundaries for third-party return values, not solver or backend capability contracts.
Importing this module imports NumPy and SciPy. Importing
``projectkoios.numerics.scipy`` alone remains dependency-free.
"""

from __future__ import annotations

from typing import Protocol, overload

import numpy as np
import scipy.sparse

from ..numpy.typing import Float64Array

type CsrMatrix = scipy.sparse.csr_matrix | scipy.sparse.csr_array
"""Concrete SciPy compressed-sparse-row matrix and array representations."""


class Float64SparseMatrix(Protocol):
    """Operations consumed from a two-dimensional SciPy float64 sparse array."""

    shape: tuple[int, int]
    dtype: np.dtype[np.float64]
    data: Float64Array
    nnz: int

    def copy(self) -> Float64SparseMatrix:
        """Return an independent sparse copy."""
        ...

    def sum_duplicates(self) -> None:
        """Combine duplicate represented entries in place."""
        ...

    def eliminate_zeros(self) -> None:
        """Remove explicitly represented zeros in place."""
        ...

    def sort_indices(self) -> None:
        """Sort sparse indices into canonical order in place."""
        ...

    def transpose(self) -> Float64SparseMatrix:
        """Return the sparse transpose."""
        ...

    def __sub__(self, other: Float64SparseMatrix, /) -> Float64SparseMatrix: ...

    def __matmul__(self, other: Float64Array, /) -> Float64Array: ...

    @overload
    def __getitem__(self, key: tuple[int, int], /) -> np.float64: ...

    @overload
    def __getitem__(
        self,
        key: tuple[slice, slice],
        /,
    ) -> Float64SparseMatrix: ...


class GaussianKde(Protocol):
    """SciPy Gaussian-KDE surface consumed by optimization samplers."""

    factor: float

    def __call__(self, points: Float64Array, /) -> Float64Array: ...

    def integrate_kde(self, other: GaussianKde, /) -> float: ...

    def logpdf(self, points: Float64Array, /) -> Float64Array: ...

    def resample(
        self,
        size: int | None = None,
        seed: np.random.RandomState | None = None,
    ) -> Float64Array: ...


class OptimizeResult(Protocol):
    """Fields consumed from a successful SciPy optimization result."""

    x: Float64Array
    fun: float
    success: bool
    message: str


__all__ = ["CsrMatrix", "Float64SparseMatrix", "GaussianKde", "OptimizeResult"]
