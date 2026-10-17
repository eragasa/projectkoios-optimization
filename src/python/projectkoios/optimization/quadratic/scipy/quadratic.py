"""SciPy-backed sparse positive-definite quadratic loss."""

from __future__ import annotations

import math
from typing import cast, final

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from projectkoios.numerics import validation as numerical_validation

from ....numerics.scipy.typing import Float64SparseMatrix
from .. import validation as quadratic_validation
from ..quadratic import DenseQuadraticInput, _QuadraticLossBackend


@final
class ScipySparsePositiveDefiniteQuadraticLoss(_QuadraticLossBackend):
    """Evaluate a completed-square loss with sparse SciPy linear algebra.

    Parameters
    ----------
    center
        Nonempty finite coordinate center.
    matrix
        Finite symmetric tuple matrix or SciPy sparse object.
    minimum_squared_loss
        Finite nonnegative loss at ``center``.

    Notes
    -----
    Positive-definiteness acceptance uses the smallest algebraic eigenvalue returned by
    ARPACK through :func:`scipy.sparse.linalg.eigsh`. It is numerical acceptance, not an
    exact mathematical certificate.
    """

    __slots__ = ("_center", "_matrix", "_minimum_squared_loss")

    def __init__(  # noqa: C901
        self,
        center: tuple[float, ...],
        matrix: DenseQuadraticInput | Float64SparseMatrix,
        minimum_squared_loss: float,
    ) -> None:
        normalized_center = quadratic_validation.finite_vector(center, "center")
        dimension = len(normalized_center)
        normalized_minimum = numerical_validation.finite_float(
            minimum_squared_loss,
            "minimum_squared_loss",
        )
        if normalized_minimum < 0.0:
            raise ValueError("minimum_squared_loss must be nonnegative")

        if sp.issparse(matrix):
            source = cast(Float64SparseMatrix, matrix)
            if source.shape != (dimension, dimension):
                raise ValueError("matrix shape must match center dimension")
            if source.dtype.kind in "bOcSUV":
                raise TypeError("matrix entries must be real numbers excluding bool")
            try:
                sparse_matrix = sp.csr_array(source, dtype=np.float64, copy=True)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "matrix entries must be representable as floats"
                ) from error
        else:
            normalized_matrix = quadratic_validation.finite_square_matrix(
                cast(DenseQuadraticInput, matrix),
                dimension,
            )
            sparse_matrix = sp.csr_array(
                np.asarray(normalized_matrix, dtype=np.float64)
            )
        sparse_matrix = cast(Float64SparseMatrix, sparse_matrix)

        # Canonical CSR removes representation-dependent duplicate and zero entries.
        sparse_matrix.sum_duplicates()
        sparse_matrix.eliminate_zeros()
        sparse_matrix.sort_indices()
        if not np.isfinite(sparse_matrix.data).all():
            raise ValueError("matrix entries must be finite")
        # Compare represented values exactly after canonicalization.
        difference = sparse_matrix - sparse_matrix.transpose()
        difference.eliminate_zeros()
        if difference.nnz != 0:
            raise ValueError("matrix must be exactly symmetric")

        if dimension == 1:
            if float(sparse_matrix[0, 0]) <= 0.0:
                raise ValueError("matrix must be positive definite")
        else:
            # Request only the eigenvalue needed for sparse backend acceptance.
            try:
                eigenvalues = spla.eigsh(
                    sparse_matrix,
                    k=1,
                    which="SA",
                    return_eigenvectors=False,
                    tol=0.0,
                )
            except (spla.ArpackError, ValueError) as error:
                raise ValueError(
                    "sparse positive-definiteness check did not converge"
                ) from error
            smallest = float(eigenvalues[0])
            if not math.isfinite(smallest) or smallest <= 0.0:
                raise ValueError("matrix must be positive definite")

        self._center = normalized_center
        self._matrix = sparse_matrix
        self._minimum_squared_loss = normalized_minimum

    @property
    def center(self) -> tuple[float, ...]:
        """Return the immutable normalized center.

        Returns
        -------
        tuple[float, ...]
            Built-in binary64 coordinates.
        """
        return self._center

    @property
    def matrix(self) -> Float64SparseMatrix:
        """Return the canonical sparse matrix.

        Returns
        -------
        scipy.sparse.csr_array
            Defensive copy of private CSR storage.
        """
        return self._matrix.copy()

    @property
    def minimum_squared_loss(self) -> float:
        """Return the completed-square minimum.

        Returns
        -------
        float
            Finite nonnegative loss at the center.
        """
        return self._minimum_squared_loss

    @property
    def dimension(self) -> int:
        """Return the coordinate dimension.

        Returns
        -------
        int
            Positive matrix dimension.
        """
        return len(self.center)

    def squared_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the completed-square loss.

        Parameters
        ----------
        coordinates
            Finite coordinates matching :attr:`dimension`.

        Returns
        -------
        float
            Finite nonnegative squared loss.

        Raises
        ------
        TypeError
            If coordinates are not an exact tuple of real values.
        ValueError
            If dimensions differ or backend arithmetic is invalid or nonfinite.
        """
        normalized = quadratic_validation.finite_vector(
            coordinates,
            "coordinates",
        )
        if len(normalized) != self.dimension:
            raise ValueError("coordinates must match the quadratic dimension")
        delta = np.asarray(normalized, dtype=np.float64) - np.asarray(
            self.center,
            dtype=np.float64,
        )
        # Sparse matrix-vector action avoids materializing a dense coefficient matrix.
        quadratic = float(delta @ (self._matrix @ delta))
        if not math.isfinite(quadratic):
            raise ValueError("quadratic evaluation must be finite")
        if quadratic < 0.0:
            raise ValueError("quadratic evaluation must be nonnegative")
        result = self.minimum_squared_loss + quadratic
        if not math.isfinite(result):
            raise ValueError("quadratic loss must be finite")
        return result

    def root_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the root loss.

        Parameters
        ----------
        coordinates
            Finite coordinates matching :attr:`dimension`.

        Returns
        -------
        float
            Nonnegative square root of :meth:`squared_loss`.
        """
        return math.sqrt(self.squared_loss(coordinates))


__all__ = ["ScipySparsePositiveDefiniteQuadraticLoss"]
