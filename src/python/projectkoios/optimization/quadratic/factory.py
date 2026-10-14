"""Closed lazy factories for constrained-quadratic solvers."""

from __future__ import annotations

from enum import StrEnum
from typing import final

from .constrained import (
    DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE,
    ConstrainedQuadraticTolerance,
    QuadraticSublevelSupportSolver,
    RectangleQuadraticMinimizer2D,
)


class RectangleQuadraticMinimizerMethod(StrEnum):
    """Supported 2-D rectangle-minimizer implementations."""

    NUMPY_DENSE = "numpy-dense"
    SCIPY_SPARSE = "scipy-sparse"


class QuadraticSublevelSupportMethod(StrEnum):
    """Supported full-ellipsoid support implementations."""

    NUMPY_DENSE = "numpy-dense"
    SCIPY_SPARSE = "scipy-sparse"


@final
class RectangleQuadraticMinimizerFactory:
    """Construct supported nominal rectangle-minimizer implementations."""

    __slots__ = ()

    @staticmethod
    def create(
        method: RectangleQuadraticMinimizerMethod,
        /,
        *,
        tolerance: ConstrainedQuadraticTolerance = (
            DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE
        ),
    ) -> RectangleQuadraticMinimizer2D:
        """Construct the exact requested dense or sparse 2-D implementation."""
        if type(method) is not RectangleQuadraticMinimizerMethod:
            raise TypeError("method must be RectangleQuadraticMinimizerMethod")
        if type(tolerance) is not ConstrainedQuadraticTolerance:
            raise TypeError("tolerance must be ConstrainedQuadraticTolerance")
        if method is RectangleQuadraticMinimizerMethod.NUMPY_DENSE:
            from .numpy.constrained import NumpyRectangleQuadraticMinimizer2D

            return NumpyRectangleQuadraticMinimizer2D(tolerance=tolerance)
        from .scipy.constrained import ScipySparseRectangleQuadraticMinimizer2D

        return ScipySparseRectangleQuadraticMinimizer2D(tolerance=tolerance)


@final
class QuadraticSublevelSupportSolverFactory:
    """Construct supported nominal ellipsoid-support implementations."""

    __slots__ = ()

    @staticmethod
    def create(
        method: QuadraticSublevelSupportMethod,
        /,
        *,
        tolerance: ConstrainedQuadraticTolerance = (
            DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE
        ),
    ) -> QuadraticSublevelSupportSolver:
        """Construct the exact requested dense or sparse support implementation."""
        if type(method) is not QuadraticSublevelSupportMethod:
            raise TypeError("method must be QuadraticSublevelSupportMethod")
        if type(tolerance) is not ConstrainedQuadraticTolerance:
            raise TypeError("tolerance must be ConstrainedQuadraticTolerance")
        if method is QuadraticSublevelSupportMethod.NUMPY_DENSE:
            from .numpy.constrained import NumpyQuadraticSublevelSupportSolver

            return NumpyQuadraticSublevelSupportSolver(tolerance=tolerance)
        from .scipy.constrained import ScipySparseQuadraticSublevelSupportSolver

        return ScipySparseQuadraticSublevelSupportSolver(tolerance=tolerance)


__all__ = [
    "QuadraticSublevelSupportMethod",
    "QuadraticSublevelSupportSolverFactory",
    "RectangleQuadraticMinimizerFactory",
    "RectangleQuadraticMinimizerMethod",
]
