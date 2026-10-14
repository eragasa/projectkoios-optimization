"""SciPy sparse implementations of constrained positive-quadratic optimization."""

from __future__ import annotations

import math
import warnings
from typing import Any, final

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
class ScipySparseRectangleQuadraticMinimizer2D(RectangleQuadraticMinimizer2D):
    """Minimize a sparse 2-D positive quadratic over a closed rectangle."""

    __slots__ = ()

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        if not loss.sparse:
            raise ValueError("SciPy sparse rectangle minimizer requires a sparse loss")
        matrix: Any = loss.matrix
        if not sp.issparse(matrix):
            raise TypeError("sparse quadratic matrix must use a SciPy sparse format")
        entries = (
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        )
        return _rectangle_minimum_from_entries(loss, bounds, entries)


@final
class ScipySparseQuadraticSublevelSupportSolver(QuadraticSublevelSupportSolver):
    """Calculate full-ellipsoid support using a SciPy sparse linear solve."""

    __slots__ = ()

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        if not loss.sparse:
            raise ValueError("SciPy sparse support solver requires a sparse loss")
        matrix: Any = loss.matrix
        if not sp.issparse(matrix):
            raise TypeError("sparse quadratic matrix must use a SciPy sparse format")

        direction_array = np.asarray(direction, dtype=np.float64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                solved = np.asarray(
                    spla.spsolve(matrix, direction_array),
                    dtype=np.float64,
                )
        except (spla.MatrixRankWarning, RuntimeError, ValueError) as error:
            raise ValueError("sparse quadratic support solve failed") from error
        if solved.shape != direction_array.shape or not np.isfinite(solved).all():
            raise ValueError("sparse quadratic support solve must be finite")

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
    "ScipySparseQuadraticSublevelSupportSolver",
    "ScipySparseRectangleQuadraticMinimizer2D",
]
