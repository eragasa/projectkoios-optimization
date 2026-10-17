"""Unit tests for immutable box-quadratic minimum evidence."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimum,
)
from tests.projectkoios.optimization.quadratic.box.fixtures import (
    SatisfiedBoxMinimumCase,
)

pytestmark = pytest.mark.unit


def _with_entry(
    values: tuple[float, ...],
    index: int,
    replacement: float,
    /,
) -> tuple[float, ...]:
    """Return immutable test evidence with one named coordinate replaced."""
    return (*values[:index], replacement, *values[index + 1 :])


def test_minimum_factory_normalizes_and_retains_complete_evidence(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    expected = satisfied_minimum_case.result
    integral_coordinates = tuple(int(value) for value in expected.coordinates)
    integral_squared_loss = int(expected.squared_loss)

    result = BoxQuadraticMinimum.from_values(
        coordinates=integral_coordinates,
        squared_loss=integral_squared_loss,
        bound_statuses=expected.bound_statuses,
        gradient=expected.gradient,
        kkt_tolerance_bounds=expected.kkt_tolerance_bounds,
        maximum_kkt_violation=expected.maximum_kkt_violation,
        maximum_kkt_ratio=expected.maximum_kkt_ratio,
        iteration_count=expected.iteration_count,
        active_set_change_count=expected.active_set_change_count,
        linear_solve_count=expected.linear_solve_count,
        maximum_iterations=expected.maximum_iterations,
        backend_identifier=expected.backend_identifier,
        backend_version=expected.backend_version,
    )

    assert result == expected


def test_raw_minimum_constructor_rejects_noncanonical_scalars(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    expected = satisfied_minimum_case.result
    integral_coordinates = tuple(int(value) for value in expected.coordinates)

    with pytest.raises(TypeError, match="exact built-in float"):
        replace(
            expected,
            coordinates=cast(tuple[float, ...], integral_coordinates),
        )


def test_minimum_rejects_inconsistent_or_malformed_evidence(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    result = satisfied_minimum_case.result
    dimension = len(result.coordinates)
    free_index = result.bound_statuses.index(BoxQuadraticBoundStatus.FREE)
    lower_index = result.bound_statuses.index(BoxQuadraticBoundStatus.LOWER)
    invalid_statuses = cast(
        tuple[BoxQuadraticBoundStatus, ...],
        ("lower",) * dimension,
    )
    short_gradient = result.gradient[:-1]
    nonpositive_bounds = (0.0,) * dimension
    lower_sign_violation = _with_entry(
        result.gradient,
        lower_index,
        -abs(result.gradient[lower_index]),
    )
    inconsistent_ratio = math.nextafter(result.maximum_kkt_ratio, math.inf)
    rejected_ratio = 2.0
    rejected_violation = rejected_ratio * satisfied_minimum_case.tolerance_bound
    free_stationarity_violation = _with_entry(
        result.gradient,
        free_index,
        rejected_violation,
    )
    excessive_solve_count = result.iteration_count + 1
    excessive_iteration_count = result.maximum_iterations + 1

    with pytest.raises(TypeError, match="bound_statuses"):
        replace(result, bound_statuses=invalid_statuses)
    with pytest.raises(ValueError, match="gradient"):
        replace(result, gradient=short_gradient)
    with pytest.raises(ValueError, match="positive"):
        replace(result, kkt_tolerance_bounds=nonpositive_bounds)
    with pytest.raises(ValueError, match="maximum_kkt_violation"):
        replace(result, gradient=lower_sign_violation)
    with pytest.raises(ValueError, match="maximum_kkt_ratio"):
        replace(result, maximum_kkt_ratio=inconsistent_ratio)
    with pytest.raises(ValueError, match="must not exceed"):
        replace(
            result,
            gradient=free_stationarity_violation,
            maximum_kkt_violation=rejected_violation,
            maximum_kkt_ratio=rejected_ratio,
        )
    with pytest.raises(TypeError, match="built-in integer"):
        replace(result, iteration_count=True)
    with pytest.raises(ValueError, match="must not exceed"):
        replace(result, linear_solve_count=excessive_solve_count)
    with pytest.raises(ValueError, match="must not exceed"):
        replace(result, iteration_count=excessive_iteration_count)
    with pytest.raises(ValueError, match="backend_identifier"):
        replace(result, backend_identifier="")
