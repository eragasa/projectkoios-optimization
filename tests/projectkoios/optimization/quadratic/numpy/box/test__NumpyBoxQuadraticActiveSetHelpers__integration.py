"""Direct integration tests for named dense active-set operations."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.optimization.quadratic.box import BoxQuadraticBoundStatus
from projectkoios.optimization.quadratic.constrained import (
    ConstrainedQuadraticTolerance,
)
from projectkoios.optimization.quadratic.numpy.box import (
    _active_set_kkt_evidence,
    _advance_free_coordinates,
    _blocking_step,
    _solve_free_direction,
)

pytestmark = pytest.mark.integration


def test_active_set_kkt_evidence_preserves_coordinate_order() -> None:
    matrix = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    center = (0.0, 0.0, 0.0)
    point = np.asarray((-1.0, 1.0, 0.25), dtype=np.float64)
    statuses = [
        BoxQuadraticBoundStatus.LOWER,
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.FREE,
    ]

    evidence = _active_set_kkt_evidence(
        matrix,
        center,
        point,
        statuses,
        ConstrainedQuadraticTolerance(),
    )

    assert evidence.coordinates == (-1.0, 1.0, 0.25)
    assert evidence.gradient == (-2.0, 2.0, 0.5)
    assert evidence.violations == (2.0, 2.0, 0.5)
    assert evidence.free_indices == (2,)
    assert evidence.free_violations == (2,)
    assert evidence.active_violations == (0, 1)
    assert all(bound > 0.0 for bound in evidence.tolerance_bounds)


def test_free_direction_solves_the_positive_definite_principal_system() -> None:
    matrix = np.asarray(
        (
            (2.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=np.float64,
    )
    gradient = (4.0, 10.0, -6.0)
    free_indices = (0, 2)

    direction = _solve_free_direction(matrix, gradient, free_indices)

    assert direction.dtype == np.dtype(np.float64)
    assert np.array_equal(direction, np.asarray((-1.0, 1.0), dtype=np.float64))


def test_blocking_step_selects_smallest_ratio_in_coordinate_scan() -> None:
    point = np.asarray((0.0, 0.0), dtype=np.float64)
    direction = np.asarray((4.0, -8.0), dtype=np.float64)
    lower = np.asarray((-1.0, -1.0), dtype=np.float64)
    upper = np.asarray((1.0, 1.0), dtype=np.float64)

    step = _blocking_step(point, direction, (0, 1), lower, upper)

    assert step == 0.125


def test_advance_updates_only_free_coordinates_and_counts_activations() -> None:
    point = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    statuses = [
        BoxQuadraticBoundStatus.FREE,
        BoxQuadraticBoundStatus.FREE,
        BoxQuadraticBoundStatus.FIXED,
    ]
    direction = np.asarray((0.5, 1.0), dtype=np.float64)
    lower = np.asarray((-1.0, -1.0, -1.0), dtype=np.float64)
    upper = np.asarray((1.0, 0.5, -1.0), dtype=np.float64)

    activation_count = _advance_free_coordinates(
        point,
        statuses,
        direction,
        (0, 1),
        lower,
        upper,
        0.5,
        ConstrainedQuadraticTolerance(),
    )

    assert np.array_equal(
        point,
        np.asarray((0.25, 0.5, -1.0), dtype=np.float64),
    )
    assert statuses == [
        BoxQuadraticBoundStatus.FREE,
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.FIXED,
    ]
    assert activation_count == 1
