"""NumPy-backed dense positive-definite quadratic loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import numpy as np

from projectkoios.numerics import validation as numerical_validation

from .. import validation as quadratic_validation
from ..quadratic import _QuadraticLossBackend


@final
@dataclass(frozen=True, slots=True)
class NumpyPositiveDefiniteQuadraticLoss(_QuadraticLossBackend):
    """Evaluate a completed-square loss with dense NumPy linear algebra.

    Parameters
    ----------
    center
        Nonempty finite coordinate center.
    matrix
        Finite exactly symmetric positive-definite tuple matrix.
    minimum_squared_loss
        Finite nonnegative loss at ``center``.
    """

    center: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]
    minimum_squared_loss: float

    def __post_init__(self) -> None:
        numerical_validation.validate_finite_float_tuple(self.center, "center")
        if type(self.matrix) is not tuple:
            raise TypeError("matrix must be an exact tuple")
        if len(self.matrix) != len(self.center):
            raise ValueError("matrix row count must match center dimension")
        for row_index, row in enumerate(self.matrix):
            numerical_validation.validate_finite_float_tuple(
                row,
                f"matrix[{row_index}]",
            )
            if len(row) != len(self.center):
                raise ValueError("matrix must be square with center dimension")
        if not quadratic_validation.symmetric(self.matrix):
            raise ValueError("matrix must be exactly symmetric")
        numerical_validation.exact_nonnegative_float(
            self.minimum_squared_loss,
            "minimum_squared_loss",
        )

        matrix_array = np.asarray(self.matrix, dtype=np.float64)
        # NumPy owns dense positive-definiteness acceptance for this adapter.
        try:
            np.linalg.cholesky(matrix_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("matrix must be positive definite") from error

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
        normalized = quadratic_validation.finite_vector(coordinates, "coordinates")
        if len(normalized) != self.dimension:
            raise ValueError("coordinates must match the quadratic dimension")
        delta = np.asarray(normalized, dtype=np.float64) - np.asarray(
            self.center,
            dtype=np.float64,
        )
        # Keep the numerical action in the backend selected by this concrete class.
        quadratic = float(delta @ np.asarray(self.matrix, dtype=np.float64) @ delta)
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


__all__ = ["NumpyPositiveDefiniteQuadraticLoss"]
