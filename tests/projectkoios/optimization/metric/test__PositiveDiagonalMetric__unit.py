import math

import pytest

pytestmark = pytest.mark.unit

from projectkoios.optimization.metric.metric import (  # noqa: E402
    PositiveDiagonalMetric,
)


def test_metric_evaluates_hand_derived_primal_and_dual_values() -> None:
    metric = PositiveDiagonalMetric((4.0, 9.0))

    assert metric.dimension == 2
    assert metric.squared_distance((1.0, 2.0), (4.0, 6.0)) == 180.0
    assert metric.distance((1.0, 2.0), (4.0, 6.0)) == pytest.approx(
        math.sqrt(180.0), rel=0.0, abs=2.0e-15
    )
    assert metric.squared_dual_norm((2.0, 3.0)) == 2.0
    assert metric.dual_norm((2.0, 3.0)) == pytest.approx(
        math.sqrt(2.0), rel=0.0, abs=2.0e-16
    )


def test_metric_preserves_immutable_ordered_diagonal() -> None:
    metric = PositiveDiagonalMetric((1.0, 2.0))

    assert metric.diagonal == (1.0, 2.0)
    assert metric.distance((0.0, 0.0), (0.0, 0.0)) == 0.0
    assert metric.dual_norm((0.0, 0.0)) == 0.0
    with pytest.raises(AttributeError):
        metric.diagonal = (3.0, 4.0)  # type: ignore[misc]


def test_metric_rejects_invalid_state_and_coordinates() -> None:
    with pytest.raises(ValueError):
        PositiveDiagonalMetric(())
    with pytest.raises(ValueError):
        PositiveDiagonalMetric((0.0,))
    with pytest.raises(ValueError):
        PositiveDiagonalMetric((-1.0,))
    with pytest.raises(TypeError):
        PositiveDiagonalMetric((True,))  # type: ignore[arg-type]

    metric = PositiveDiagonalMetric((1.0,))
    with pytest.raises(ValueError):
        metric.distance((0.0,), (0.0, 1.0))
    with pytest.raises(TypeError):
        metric.dual_norm([1.0])  # type: ignore[arg-type]
