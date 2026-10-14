"""NumPy-backed dense positive-definite quadratic loss."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import final

import numpy as np
import numpy.typing as npt

from ...validation import validation
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
    _center_array: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _matrix_array: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        center = quadratic_validation.finite_vector(self.center, "center")
        matrix = quadratic_validation.finite_square_matrix(self.matrix, len(center))
        if not quadratic_validation.symmetric(matrix):
            raise ValueError("matrix must be exactly symmetric")
        minimum = validation.finite_float(
            self.minimum_squared_loss,
            "minimum_squared_loss",
        )
        if minimum < 0.0:
            raise ValueError("minimum_squared_loss must be nonnegative")

        center_array = np.asarray(center, dtype=np.float64)
        matrix_array = np.asarray(matrix, dtype=np.float64)
        # NumPy owns dense positive-definiteness acceptance for this adapter.
        try:
            np.linalg.cholesky(matrix_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("matrix must be positive definite") from error
        # Prevent backend arrays from becoming a mutation path around frozen state.
        center_array.flags.writeable = False
        matrix_array.flags.writeable = False

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "minimum_squared_loss", minimum)
        object.__setattr__(self, "_center_array", center_array)
        object.__setattr__(self, "_matrix_array", matrix_array)

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
        delta = np.asarray(normalized, dtype=np.float64) - self._center_array
        # Keep the numerical action in the backend selected by this concrete class.
        quadratic = float(delta @ self._matrix_array @ delta)
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
