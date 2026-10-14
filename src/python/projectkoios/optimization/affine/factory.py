"""Closed lazy factories for affine quadratic constructors."""

from __future__ import annotations

from enum import StrEnum
from typing import final

from .complex_quadratic import (
    DEFAULT_COMPLEX_AFFINE_RESIDUAL_QUADRATIC_TOLERANCE,
    ComplexAffineResidualQuadraticConstructor,
    ComplexAffineResidualQuadraticTolerance,
)


class ComplexAffineResidualQuadraticMethod(StrEnum):
    """Supported complex-observation affine quadratic implementations."""

    NUMPY_DENSE = "numpy-dense"


@final
class ComplexAffineResidualQuadraticConstructorFactory:
    """Construct supported nominal complex affine quadratic implementations."""

    __slots__ = ()

    @staticmethod
    def create(
        method: ComplexAffineResidualQuadraticMethod,
        /,
        *,
        tolerance: ComplexAffineResidualQuadraticTolerance = (
            DEFAULT_COMPLEX_AFFINE_RESIDUAL_QUADRATIC_TOLERANCE
        ),
    ) -> ComplexAffineResidualQuadraticConstructor:
        """Construct the exact requested dense implementation lazily."""
        if type(method) is not ComplexAffineResidualQuadraticMethod:
            raise TypeError("method must be ComplexAffineResidualQuadraticMethod")
        if type(tolerance) is not ComplexAffineResidualQuadraticTolerance:
            raise TypeError("tolerance must be ComplexAffineResidualQuadraticTolerance")
        from .numpy.complex_quadratic import (
            NumpyComplexAffineResidualQuadraticConstructor,
        )

        return NumpyComplexAffineResidualQuadraticConstructor(tolerance=tolerance)


__all__ = [
    "ComplexAffineResidualQuadraticConstructorFactory",
    "ComplexAffineResidualQuadraticMethod",
]
