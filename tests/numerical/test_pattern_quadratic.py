import json
from pathlib import Path

import pytest

from projectkoios.optimization import (
    BoundedCoordinatePatternOptimizer,
    ObjectiveFeedback,
    ParameterBound,
    ParameterSpace,
    PatternSearchSettings,
)

pytestmark = pytest.mark.numerical


def objective(values: tuple[float, ...]) -> float:
    x, y = values
    return (x - 1.0) ** 2 + 2.0 * (y + 2.0) ** 2


def run(optimizer: BoundedCoordinatePatternOptimizer) -> None:
    while optimizer.termination is None:
        candidate = optimizer.ask()
        optimizer.tell(
            ObjectiveFeedback(candidate.candidate_id, objective(candidate.values))
        )


def make() -> BoundedCoordinatePatternOptimizer:
    return BoundedCoordinatePatternOptimizer(
        ParameterSpace(
            (ParameterBound("x", -4.0, 4.0), ParameterBound("y", -4.0, 4.0))
        ),
        (0.0, 0.0),
        PatternSearchSettings(100, 30, 0.25, 0.5, 0.001),
        run_id="quadratic",
    )


def test_pattern_search_reaches_exact_bounded_quadratic_minimum() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/synthetic-quadratic.json").read_text()
    )
    assert fixture["description"].startswith("Sanitized dimensionless")
    optimizer = make()
    run(optimizer)
    assert optimizer.termination == "step_tolerance"
    assert optimizer.best_candidate is not None
    assert optimizer.best_candidate.values == (1.0, -2.0)
    assert optimizer.best_loss == pytest.approx(0.0, abs=0.0)
    assert len(optimizer.observations) <= 100
    assert optimizer.iterations <= 30
