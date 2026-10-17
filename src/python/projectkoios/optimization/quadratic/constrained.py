"""Dependency-free contracts for constrained positive quadratics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from projectkoios.numerics.validation import (
    exact_finite_float,
    exact_nonnegative_float,
    finite_float,
    validate_finite_float_tuple,
)

from ..spaces.parameters import ParameterSpace
from .quadratic import PositiveDefiniteQuadraticLoss
from .validation import finite_vector


@final
@dataclass(frozen=True, slots=True)
class ConstrainedQuadraticTolerance:
    """Numerical acceptance tolerance for constrained-quadratic backend results."""

    absolute: float = 1.0e-12
    relative: float = 1.0e-10

    def __post_init__(self) -> None:
        exact_nonnegative_float(self.absolute, "absolute")
        exact_nonnegative_float(self.relative, "relative")
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("at least one tolerance must be positive")

    def accepts(self, left: float, right: float, /) -> bool:
        """Return whether two finite scalars satisfy the scale-aware tolerance."""
        normalized_left = finite_float(left, "left")
        normalized_right = finite_float(right, "right")
        difference = abs(normalized_left - normalized_right)
        bound = self.absolute + self.relative * max(
            abs(normalized_left),
            abs(normalized_right),
        )
        return (
            math.isfinite(difference) and math.isfinite(bound) and difference <= bound
        )


DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE = ConstrainedQuadraticTolerance()


class RectangleQuadraticCandidate(StrEnum):
    """Closed deterministic candidate inventory for the exact 2-D algorithm."""

    CLIPPED_CENTER = "clipped-center"
    COORDINATE_0_LOWER = "coordinate-0-lower"
    COORDINATE_0_UPPER = "coordinate-0-upper"
    COORDINATE_1_LOWER = "coordinate-1-lower"
    COORDINATE_1_UPPER = "coordinate-1-upper"


@final
@dataclass(frozen=True, slots=True)
class RectangleQuadraticMinimum:
    """Global minimum of a positive quadratic over one closed 2-D rectangle."""

    coordinates: tuple[float, float]
    squared_loss: float
    candidate: RectangleQuadraticCandidate

    def __post_init__(self) -> None:
        if type(self.coordinates) is not tuple:
            raise TypeError("coordinates must be an exact tuple")
        if len(self.coordinates) != 2:
            raise ValueError("coordinates must contain exactly two values")
        exact_finite_float(self.coordinates[0], "coordinates[0]")
        exact_finite_float(self.coordinates[1], "coordinates[1]")
        exact_nonnegative_float(self.squared_loss, "squared_loss")
        if type(self.candidate) is not RectangleQuadraticCandidate:
            raise TypeError("candidate must be RectangleQuadraticCandidate")


@final
@dataclass(frozen=True, slots=True)
class QuadraticSublevelSupportResult:
    """Positive-direction support point of one complete quadratic sublevel ellipsoid."""

    point: tuple[float, ...]
    direction: tuple[float, ...]
    support_value: float
    squared_loss: float

    def __post_init__(self) -> None:
        validate_finite_float_tuple(self.point, "point")
        validate_finite_float_tuple(self.direction, "direction")
        if len(self.point) != len(self.direction):
            raise ValueError("point and direction dimensions must match")
        if not any(value != 0.0 for value in self.direction):
            raise ValueError("direction must be nonzero")
        exact_finite_float(self.support_value, "support_value")
        exact_nonnegative_float(self.squared_loss, "squared_loss")


class RectangleQuadraticMinimizer2D(ABC):
    """Nominal SDK boundary for exact 2-D rectangle minimization."""

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        *,
        tolerance: ConstrainedQuadraticTolerance = (
            DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE
        ),
    ) -> None:
        if type(tolerance) is not ConstrainedQuadraticTolerance:
            raise TypeError("tolerance must be ConstrainedQuadraticTolerance")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> ConstrainedQuadraticTolerance:
        """Return the capability-specific backend-result tolerance."""
        return self._tolerance

    @final
    def solve(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        """Return the deterministic global minimum over a closed 2-D rectangle."""
        if type(loss) is not PositiveDefiniteQuadraticLoss:
            raise TypeError("loss must be PositiveDefiniteQuadraticLoss")
        if type(bounds) is not ParameterSpace:
            raise TypeError("bounds must be ParameterSpace")
        if loss.dimension != 2:
            raise ValueError("loss must have dimension two")
        if bounds.dimension != 2:
            raise ValueError("bounds must have dimension two")

        result = self._solve_validated(loss, bounds)
        if type(result) is not RectangleQuadraticMinimum:
            raise TypeError("backend must return RectangleQuadraticMinimum")
        if not bounds.admits(result.coordinates):
            raise ValueError("backend minimum must lie inside bounds")
        evaluated = loss.squared_loss(result.coordinates)
        if not self.tolerance.accepts(result.squared_loss, evaluated):
            raise ValueError("backend minimum squared_loss is inconsistent")
        return result

    @abstractmethod
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        """Solve validated inputs with one concrete backend."""
        raise NotImplementedError


class QuadraticSublevelSupportSolver(ABC):
    """Nominal SDK boundary for full-ellipsoid support optimization."""

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        *,
        tolerance: ConstrainedQuadraticTolerance = (
            DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE
        ),
    ) -> None:
        if type(tolerance) is not ConstrainedQuadraticTolerance:
            raise TypeError("tolerance must be ConstrainedQuadraticTolerance")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> ConstrainedQuadraticTolerance:
        """Return the capability-specific backend-result tolerance."""
        return self._tolerance

    @final
    def solve(  # noqa: C901
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        """Maximize a direction over one complete quadratic sublevel ellipsoid."""
        if type(loss) is not PositiveDefiniteQuadraticLoss:
            raise TypeError("loss must be PositiveDefiniteQuadraticLoss")
        ceiling = finite_float(squared_loss_ceiling, "squared_loss_ceiling")
        if ceiling < 0.0:
            raise ValueError("squared_loss_ceiling must be nonnegative")
        normalized_direction = finite_vector(direction, "direction")
        if len(normalized_direction) != loss.dimension:
            raise ValueError("direction must match the quadratic dimension")
        if not any(value != 0.0 for value in normalized_direction):
            raise ValueError("direction must be nonzero")
        if ceiling < loss.minimum_squared_loss:
            raise ValueError("requested quadratic sublevel set is empty")

        result = self._solve_validated(loss, ceiling, normalized_direction)
        if type(result) is not QuadraticSublevelSupportResult:
            raise TypeError("backend must return QuadraticSublevelSupportResult")
        if result.direction != normalized_direction:
            raise ValueError("backend result must retain the normalized direction")
        if len(result.point) != loss.dimension:
            raise ValueError("backend support point has the wrong dimension")
        if ceiling == loss.minimum_squared_loss and result.point != loss.center:
            raise ValueError("zero-budget support point must equal the center")

        expected_support = _finite_dot(normalized_direction, result.point)
        if not self.tolerance.accepts(result.support_value, expected_support):
            raise ValueError("backend support_value is inconsistent")
        evaluated = loss.squared_loss(result.point)
        if not self.tolerance.accepts(result.squared_loss, evaluated):
            raise ValueError("backend support squared_loss is inconsistent")
        if not self.tolerance.accepts(result.squared_loss, ceiling):
            raise ValueError("backend support point is not on the requested boundary")
        return result

    @abstractmethod
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        """Solve validated inputs with one concrete backend."""
        raise NotImplementedError


def _finite_dot(left: tuple[float, ...], right: tuple[float, ...], /) -> float:
    products: list[float] = []
    for left_value, right_value in zip(left, right, strict=True):
        product = left_value * right_value
        if not math.isfinite(product):
            raise ValueError("dot-product terms must be finite")
        products.append(product)
    try:
        result = math.fsum(products)
    except OverflowError as error:
        raise ValueError("dot product must be finite") from error
    if not math.isfinite(result):
        raise ValueError("dot product must be finite")
    return result


def _rectangle_minimum_from_entries(
    loss: PositiveDefiniteQuadraticLoss,
    bounds: ParameterSpace,
    matrix: tuple[tuple[float, float], tuple[float, float]],
    /,
) -> RectangleQuadraticMinimum:
    center = loss.center
    lower = (bounds.bounds[0].lower, bounds.bounds[1].lower)
    upper = (bounds.bounds[0].upper, bounds.bounds[1].upper)

    candidates: list[tuple[RectangleQuadraticCandidate, tuple[float, float]]] = [
        (
            RectangleQuadraticCandidate.CLIPPED_CENTER,
            (
                min(max(center[0], lower[0]), upper[0]),
                min(max(center[1], lower[1]), upper[1]),
            ),
        )
    ]
    labels = (
        (
            RectangleQuadraticCandidate.COORDINATE_0_LOWER,
            RectangleQuadraticCandidate.COORDINATE_0_UPPER,
        ),
        (
            RectangleQuadraticCandidate.COORDINATE_1_LOWER,
            RectangleQuadraticCandidate.COORDINATE_1_UPPER,
        ),
    )
    for axis in (0, 1):
        other = 1 - axis
        for label, bound in zip(labels[axis], (lower[axis], upper[axis]), strict=True):
            candidate_point = [center[0], center[1]]
            candidate_point[axis] = bound
            candidate_point[other] = center[other] - (
                matrix[other][axis] / matrix[other][other] * (bound - center[axis])
            )
            candidate_point[other] = min(
                max(candidate_point[other], lower[other]), upper[other]
            )
            candidates.append(
                (
                    label,
                    (float(candidate_point[0]), float(candidate_point[1])),
                )
            )

    first_label, first_point = candidates[0]
    best = RectangleQuadraticMinimum(
        first_point,
        loss.squared_loss(first_point),
        first_label,
    )
    for label, point in candidates[1:]:
        squared_loss = loss.squared_loss(point)
        if squared_loss < best.squared_loss:
            best = RectangleQuadraticMinimum(point, squared_loss, label)
    return best


__all__ = [
    "ConstrainedQuadraticTolerance",
    "DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE",
    "QuadraticSublevelSupportResult",
    "QuadraticSublevelSupportSolver",
    "RectangleQuadraticCandidate",
    "RectangleQuadraticMinimum",
    "RectangleQuadraticMinimizer2D",
]
