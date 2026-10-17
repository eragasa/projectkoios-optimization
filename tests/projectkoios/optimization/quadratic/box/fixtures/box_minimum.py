"""Named analytical fixtures for box-quadratic result tests."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimum,
)


@dataclass(frozen=True, slots=True)
class SatisfiedBoxMinimumCase:
    """Retain one analytically KKT-satisfied case spanning all bound statuses."""

    result: BoxQuadraticMinimum
    tolerance_bound: float


def satisfied_box_minimum_case() -> SatisfiedBoxMinimumCase:
    """Construct the shared all-status KKT fixture from named relations."""
    statuses = (
        BoxQuadraticBoundStatus.LOWER,
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.FREE,
        BoxQuadraticBoundStatus.FIXED,
    )
    dimension = len(statuses)
    coordinates = tuple(float(index) for index in range(dimension))
    squared_loss = sum(coordinate**2 for coordinate in coordinates)
    lower_feasible_gradient = 1.0
    upper_feasible_gradient = -1.0
    stationary_gradient = 0.0
    fixed_coordinate_gradient = 2.0
    gradient = (
        lower_feasible_gradient,
        upper_feasible_gradient,
        stationary_gradient,
        fixed_coordinate_gradient,
    )
    tolerance_bound = math.sqrt(sys.float_info.epsilon)
    zero_violation = 0.0
    iteration_count = dimension
    active_set_change_count = sum(
        status in {BoxQuadraticBoundStatus.LOWER, BoxQuadraticBoundStatus.UPPER}
        for status in statuses
    )
    linear_solve_count = statuses.count(BoxQuadraticBoundStatus.FREE)
    result = BoxQuadraticMinimum(
        coordinates=coordinates,
        squared_loss=squared_loss,
        bound_statuses=statuses,
        gradient=gradient,
        kkt_tolerance_bounds=(tolerance_bound,) * dimension,
        maximum_kkt_violation=zero_violation,
        maximum_kkt_ratio=zero_violation,
        iteration_count=iteration_count,
        active_set_change_count=active_set_change_count,
        linear_solve_count=linear_solve_count,
        maximum_iterations=dimension**2,
        backend_identifier="test.active-set",
        backend_version="test-version",
    )
    return SatisfiedBoxMinimumCase(
        result=result,
        tolerance_bound=tolerance_bound,
    )


__all__ = ["SatisfiedBoxMinimumCase", "satisfied_box_minimum_case"]
