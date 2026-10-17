"""Unit tests for immutable box-quadratic minimum evidence."""

from __future__ import annotations

from typing import cast

import pytest

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimum,
)

pytestmark = pytest.mark.unit


def test_zero_centered_free_minimum_retains_exact_evidence() -> None:
    result = BoxQuadraticMinimum(
        coordinates=(0.0,),
        squared_loss=0.0,
        bound_statuses=(BoxQuadraticBoundStatus.FREE,),
        gradient=(0.0,),
        kkt_tolerance_bounds=(1.0e-12,),
        maximum_kkt_violation=0.0,
        maximum_kkt_ratio=0.0,
        iteration_count=1,
        active_set_change_count=0,
        linear_solve_count=0,
        maximum_iterations=10,
        backend_identifier="test.active-set",
        backend_version="1",
    )

    assert result.coordinates == (0.0,)
    assert result.maximum_kkt_violation == 0.0
    assert result.maximum_kkt_ratio == 0.0


def test_constructor_rejects_noncanonical_coordinate_storage() -> None:
    with pytest.raises(TypeError):
        BoxQuadraticMinimum(
            coordinates=cast(tuple[float, ...], (0,)),
            squared_loss=0.0,
            bound_statuses=(BoxQuadraticBoundStatus.FREE,),
            gradient=(0.0,),
            kkt_tolerance_bounds=(1.0e-12,),
            maximum_kkt_violation=0.0,
            maximum_kkt_ratio=0.0,
            iteration_count=1,
            active_set_change_count=0,
            linear_solve_count=0,
            maximum_iterations=10,
            backend_identifier="test.active-set",
            backend_version="1",
        )


def test_constructor_rejects_incomplete_coordinate_evidence() -> None:
    with pytest.raises(ValueError):
        BoxQuadraticMinimum(
            coordinates=(0.0,),
            squared_loss=0.0,
            bound_statuses=(BoxQuadraticBoundStatus.FREE,),
            gradient=(),
            kkt_tolerance_bounds=(1.0e-12,),
            maximum_kkt_violation=0.0,
            maximum_kkt_ratio=0.0,
            iteration_count=1,
            active_set_change_count=0,
            linear_solve_count=0,
            maximum_iterations=10,
            backend_identifier="test.active-set",
            backend_version="1",
        )


def test_constructor_reconstructs_free_coordinate_kkt_evidence() -> None:
    with pytest.raises(ValueError):
        BoxQuadraticMinimum(
            coordinates=(0.0,),
            squared_loss=0.0,
            bound_statuses=(BoxQuadraticBoundStatus.FREE,),
            gradient=(1.0e-12,),
            kkt_tolerance_bounds=(1.0e-12,),
            maximum_kkt_violation=0.0,
            maximum_kkt_ratio=1.0,
            iteration_count=1,
            active_set_change_count=0,
            linear_solve_count=0,
            maximum_iterations=10,
            backend_identifier="test.active-set",
            backend_version="1",
        )


def test_constructor_rejects_failed_kkt_acceptance() -> None:
    with pytest.raises(ValueError):
        BoxQuadraticMinimum(
            coordinates=(0.0,),
            squared_loss=0.0,
            bound_statuses=(BoxQuadraticBoundStatus.FREE,),
            gradient=(2.0e-12,),
            kkt_tolerance_bounds=(1.0e-12,),
            maximum_kkt_violation=2.0e-12,
            maximum_kkt_ratio=2.0,
            iteration_count=1,
            active_set_change_count=0,
            linear_solve_count=0,
            maximum_iterations=10,
            backend_identifier="test.active-set",
            backend_version="1",
        )


def test_constructor_rejects_execution_counts_outside_declared_bounds() -> None:
    with pytest.raises(ValueError):
        BoxQuadraticMinimum(
            coordinates=(0.0,),
            squared_loss=0.0,
            bound_statuses=(BoxQuadraticBoundStatus.FREE,),
            gradient=(0.0,),
            kkt_tolerance_bounds=(1.0e-12,),
            maximum_kkt_violation=0.0,
            maximum_kkt_ratio=0.0,
            iteration_count=1,
            active_set_change_count=0,
            linear_solve_count=2,
            maximum_iterations=10,
            backend_identifier="test.active-set",
            backend_version="1",
        )
