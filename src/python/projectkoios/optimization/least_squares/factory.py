"""Closed lazy factory for linear least-squares solvers."""

from __future__ import annotations

from enum import StrEnum
from typing import final

from .linear import (
    DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE,
    LinearLeastSquaresSolver,
    LinearLeastSquaresTolerance,
)


class LinearLeastSquaresMethod(StrEnum):
    """Supported linear least-squares implementations."""

    NUMPY_DENSE = "numpy-dense"


@final
class LinearLeastSquaresSolverFactory:
    """Construct a supported nominal linear least-squares implementation."""

    __slots__ = ()

    @staticmethod
    def create(
        method: LinearLeastSquaresMethod,
        /,
        *,
        tolerance: LinearLeastSquaresTolerance = (
            DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE
        ),
    ) -> LinearLeastSquaresSolver:
        """Construct the exact requested dense implementation lazily."""
        if type(method) is not LinearLeastSquaresMethod:
            raise TypeError("method must be LinearLeastSquaresMethod")
        if type(tolerance) is not LinearLeastSquaresTolerance:
            raise TypeError("tolerance must be LinearLeastSquaresTolerance")
        from .numpy.linear import NumpyLinearLeastSquaresSolver

        return NumpyLinearLeastSquaresSolver(tolerance=tolerance)


__all__ = [
    "LinearLeastSquaresMethod",
    "LinearLeastSquaresSolverFactory",
]
