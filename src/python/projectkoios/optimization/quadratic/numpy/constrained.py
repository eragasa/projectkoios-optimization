"""NumPy implementations of constrained positive-quadratic optimization."""

from __future__ import annotations

import math
from typing import cast, final

import numpy as np

from ...spaces.parameters import ParameterSpace
from ..constrained import (
    QuadraticSublevelSupportResult,
    QuadraticSublevelSupportSolver,
    RectangleQuadraticMinimizer2D,
    RectangleQuadraticMinimum,
    _rectangle_minimum_from_entries,
)
from ..quadratic import PositiveDefiniteQuadraticLoss


@final
class NumpyRectangleQuadraticMinimizer2D(RectangleQuadraticMinimizer2D):
    """Minimize a dense 2-D positive quadratic over a closed rectangle."""

    __slots__ = ()

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        if loss.sparse:
            raise ValueError("NumPy rectangle minimizer requires a dense loss")
        matrix_object = loss.matrix
        if type(matrix_object) is not tuple:
            raise TypeError("dense quadratic matrix must be a tuple of tuples")
        matrix = cast(tuple[tuple[float, float], tuple[float, float]], matrix_object)
        return _rectangle_minimum_from_entries(loss, bounds, matrix)


@final
class NumpyQuadraticSublevelSupportSolver(QuadraticSublevelSupportSolver):
    """Calculate full-ellipsoid support using a dense NumPy linear solve."""

    __slots__ = ()

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        if loss.sparse:
            raise ValueError("NumPy support solver requires a dense loss")
        matrix_object = loss.matrix
        if type(matrix_object) is not tuple:
            raise TypeError("dense quadratic matrix must be a tuple of tuples")

        matrix = np.asarray(matrix_object, dtype=np.float64)
        direction_array = np.asarray(direction, dtype=np.float64)
        try:
            solved = np.linalg.solve(matrix, direction_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("dense quadratic support solve failed") from error
        if not np.isfinite(solved).all():
            raise ValueError("dense quadratic support solve must be finite")

        denominator = float(direction_array @ solved)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "quadratic support denominator must be positive and finite"
            )
        budget = squared_loss_ceiling - loss.minimum_squared_loss
        ratio = budget / denominator
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("quadratic support scale must be finite and nonnegative")
        scale = math.sqrt(ratio)
        point_array = np.asarray(loss.center, dtype=np.float64) + scale * solved
        if not np.isfinite(point_array).all():
            raise ValueError("quadratic support point must be finite")
        point = tuple(float(value) for value in point_array)
        support_value = float(direction_array @ point_array)
        if not math.isfinite(support_value):
            raise ValueError("quadratic support value must be finite")
        squared_loss = loss.squared_loss(point)
        return QuadraticSublevelSupportResult(
            point,
            direction,
            support_value,
            squared_loss,
        )


__all__ = [
    "NumpyQuadraticSublevelSupportSolver",
    "NumpyRectangleQuadraticMinimizer2D",
]
