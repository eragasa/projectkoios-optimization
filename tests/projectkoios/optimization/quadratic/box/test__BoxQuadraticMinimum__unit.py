"""Unit tests for immutable box-quadratic minimum evidence."""

import pytest

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimum,
)

pytestmark = pytest.mark.unit


def _minimum(**overrides: object) -> BoxQuadraticMinimum:
    values: dict[str, object] = {
        "coordinates": (0.0, 1.0, 2.0, 3.0),
        "squared_loss": 2.0,
        "bound_statuses": (
            BoxQuadraticBoundStatus.LOWER,
            BoxQuadraticBoundStatus.UPPER,
            BoxQuadraticBoundStatus.FREE,
            BoxQuadraticBoundStatus.FIXED,
        ),
        "gradient": (1.0, -2.0, 0.0, 99.0),
        "kkt_tolerance_bounds": (1.0e-6,) * 4,
        "maximum_kkt_violation": 0.0,
        "maximum_kkt_ratio": 0.0,
        "iteration_count": 4,
        "active_set_change_count": 2,
        "linear_solve_count": 3,
        "maximum_iterations": 100,
        "backend_identifier": "projectkoios.active-set.numpy",
        "backend_version": "1/numpy-2.5.4",
    }
    values.update(overrides)
    return BoxQuadraticMinimum(**values)  # type: ignore[arg-type]


def test_minimum_normalizes_and_retains_complete_evidence() -> None:
    result = _minimum(coordinates=(0, 1, 2, 3), squared_loss=2)

    assert result.coordinates == (0.0, 1.0, 2.0, 3.0)
    assert result.maximum_kkt_ratio == 0.0
    assert result.iteration_count == 4
    assert result.active_set_change_count == 2
    assert result.linear_solve_count == 3
    assert result.maximum_iterations == 100


def test_minimum_rejects_inconsistent_or_malformed_evidence() -> None:
    with pytest.raises(TypeError, match="bound_statuses"):
        _minimum(bound_statuses=("lower",) * 4)
    with pytest.raises(ValueError, match="gradient"):
        _minimum(gradient=(1.0,))
    with pytest.raises(ValueError, match="positive"):
        _minimum(kkt_tolerance_bounds=(0.0,) * 4)
    with pytest.raises(ValueError, match="maximum_kkt_violation"):
        _minimum(
            gradient=(-1.0, -2.0, 0.0, 99.0),
            maximum_kkt_violation=0.0,
        )
    with pytest.raises(ValueError, match="maximum_kkt_ratio"):
        _minimum(maximum_kkt_ratio=0.5)
    with pytest.raises(ValueError, match="must not exceed"):
        _minimum(
            gradient=(-2.0e-6, -2.0, 0.0, 99.0),
            maximum_kkt_violation=2.0e-6,
            maximum_kkt_ratio=2.0,
        )
    with pytest.raises(TypeError, match="built-in integer"):
        _minimum(iteration_count=True)
    with pytest.raises(ValueError, match="must not exceed"):
        _minimum(iteration_count=4, linear_solve_count=5)
    with pytest.raises(ValueError, match="must not exceed"):
        _minimum(iteration_count=101, maximum_iterations=100)
    with pytest.raises(ValueError, match="backend_identifier"):
        _minimum(backend_identifier="")
