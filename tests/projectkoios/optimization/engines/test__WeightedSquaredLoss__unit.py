import pytest

from projectkoios.optimization.engines.loss import (
    WeightedLossTerm,
    WeightedSquaredLoss,
)
from projectkoios.optimization.engines.records import Measurement, RawEvaluation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_weighted_loss_rejects_nonpositive_weights(weight: float) -> None:
    with pytest.raises(ValueError):
        WeightedLossTerm("q", 0.0, weight)


def test_weighted_loss_preserves_declared_objective_order() -> None:
    loss = WeightedSquaredLoss(
        (
            WeightedLossTerm("x", 1.0, 1.0),
            WeightedLossTerm("y", -2.0, 2.0),
        )
    )

    evaluated = loss.evaluate(
        RawEvaluation("c", (Measurement("y", 0.0), Measurement("x", 3.0)))
    )

    assert evaluated.value == 12.0
    assert tuple(item.name for item in evaluated.objectives) == ("x", "y")
    assert tuple(item.value for item in evaluated.objectives) == (2.0, 2.0)

    with pytest.raises(ValueError, match="missing"):
        loss.evaluate(RawEvaluation("c", (Measurement("x", 1.0),)))
