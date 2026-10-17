"""Direct tests for box-minimum validation responsibility owners."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import pytest

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimum,
    _BoxMinimumExecutionEvidence,
    _BoxMinimumKktEvidence,
    _BoxMinimumVectors,
    _validate_box_minimum_execution,
    _validate_box_minimum_kkt,
    _validate_box_minimum_vectors,
)
from tests.projectkoios.optimization.quadratic.box.fixtures import (
    SatisfiedBoxMinimumCase,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _BoxMinimumProbe:
    """Mirror result attributes while permitting one deliberately invalid field."""

    coordinates: tuple[float, ...]
    squared_loss: float | int
    bound_statuses: tuple[BoxQuadraticBoundStatus, ...]
    gradient: tuple[float, ...]
    kkt_tolerance_bounds: tuple[float, ...]
    maximum_kkt_violation: float
    maximum_kkt_ratio: float
    iteration_count: int
    active_set_change_count: int
    linear_solve_count: int
    maximum_iterations: int
    backend_identifier: str
    backend_version: str

    @classmethod
    def from_result(cls, result: BoxQuadraticMinimum, /) -> _BoxMinimumProbe:
        """Copy one valid result into an immutable validation probe."""
        return cls(
            coordinates=result.coordinates,
            squared_loss=result.squared_loss,
            bound_statuses=result.bound_statuses,
            gradient=result.gradient,
            kkt_tolerance_bounds=result.kkt_tolerance_bounds,
            maximum_kkt_violation=result.maximum_kkt_violation,
            maximum_kkt_ratio=result.maximum_kkt_ratio,
            iteration_count=result.iteration_count,
            active_set_change_count=result.active_set_change_count,
            linear_solve_count=result.linear_solve_count,
            maximum_iterations=result.maximum_iterations,
            backend_identifier=result.backend_identifier,
            backend_version=result.backend_version,
        )

    def as_result(self) -> BoxQuadraticMinimum:
        """Expose the structural probe to a single validation owner."""
        return cast(BoxQuadraticMinimum, self)


def test_vector_validation_preserves_canonical_values_and_dimensions(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    result = satisfied_minimum_case.result
    probe = _BoxMinimumProbe.from_result(result)
    expected = _BoxMinimumVectors(
        coordinates=result.coordinates,
        squared_loss=result.squared_loss,
        gradient=result.gradient,
        tolerance_bounds=result.kkt_tolerance_bounds,
    )

    assert _validate_box_minimum_vectors(result) == expected
    with pytest.raises(TypeError, match="exact built-in float"):
        _validate_box_minimum_vectors(
            replace(probe, squared_loss=int(result.squared_loss)).as_result()
        )
    with pytest.raises(ValueError, match="gradient"):
        _validate_box_minimum_vectors(
            replace(probe, gradient=result.gradient[:-1]).as_result()
        )
    with pytest.raises(ValueError, match="positive"):
        invalid_bounds = (
            -satisfied_minimum_case.tolerance_bound,
            *result.kkt_tolerance_bounds[1:],
        )
        _validate_box_minimum_vectors(
            replace(probe, kkt_tolerance_bounds=invalid_bounds).as_result()
        )


def test_kkt_validation_reconstructs_componentwise_evidence(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    result = satisfied_minimum_case.result
    probe = _BoxMinimumProbe.from_result(result)
    vectors = _validate_box_minimum_vectors(result)
    expected = _BoxMinimumKktEvidence(
        maximum_violation=result.maximum_kkt_violation,
        maximum_ratio=result.maximum_kkt_ratio,
    )

    assert _validate_box_minimum_kkt(result, vectors) == expected
    inconsistent_violation = math.ulp(satisfied_minimum_case.tolerance_bound)
    with pytest.raises(ValueError, match="maximum_kkt_violation"):
        _validate_box_minimum_kkt(
            replace(
                probe,
                maximum_kkt_violation=inconsistent_violation,
            ).as_result(),
            vectors,
        )
    with pytest.raises(ValueError, match="maximum_kkt_ratio"):
        _validate_box_minimum_kkt(
            replace(
                probe,
                maximum_kkt_ratio=inconsistent_violation,
            ).as_result(),
            vectors,
        )


def test_execution_validation_separates_counts_and_backend_identity(
    satisfied_minimum_case: SatisfiedBoxMinimumCase,
) -> None:
    result = satisfied_minimum_case.result
    probe = _BoxMinimumProbe.from_result(result)
    expected = _BoxMinimumExecutionEvidence(
        iterations=result.iteration_count,
        changes=result.active_set_change_count,
        solves=result.linear_solve_count,
        maximum_iterations=result.maximum_iterations,
        identifier=result.backend_identifier,
        version=result.backend_version,
    )

    assert _validate_box_minimum_execution(result) == expected
    with pytest.raises(ValueError, match="linear_solve_count"):
        _validate_box_minimum_execution(
            replace(
                probe,
                linear_solve_count=result.iteration_count + 1,
            ).as_result()
        )
    with pytest.raises(ValueError, match="backend_identifier"):
        _validate_box_minimum_execution(
            replace(probe, backend_identifier=" padded ").as_result()
        )
