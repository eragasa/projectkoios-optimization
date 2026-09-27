import pytest

from projectkoios.optimization import (
    BoundedCoordinatePatternOptimizer,
    BoundedUniformOptimizer,
    ObjectiveFeedback,
    OptimizerCheckpoint,
    ParameterBound,
    ParameterSpace,
    PatternSearchSettings,
)

pytestmark = pytest.mark.property


def test_uniform_resumed_and_uninterrupted_are_exactly_equivalent() -> None:
    space = ParameterSpace((ParameterBound("x", -2.0, 3.0),))
    original = BoundedUniformOptimizer(space, seed=7, max_evaluations=8, run_id="u")
    candidates = [original.ask() for _ in range(5)]
    for index in (3, 0):
        original.tell(ObjectiveFeedback(candidates[index].candidate_id, float(index)))
    resumed = BoundedUniformOptimizer.from_checkpoint(
        OptimizerCheckpoint.from_json(original.checkpoint().to_json())
    )
    assert resumed.checkpoint() == original.checkpoint()
    assert resumed.ask() == original.ask()


def test_pattern_resumed_and_uninterrupted_are_exactly_equivalent() -> None:
    original = BoundedCoordinatePatternOptimizer(
        ParameterSpace((ParameterBound("x", -4.0, 4.0),)),
        (0.0,),
        PatternSearchSettings(40, 20, 0.25, 0.5, 0.001),
        run_id="p",
    )
    for _ in range(6):
        candidate = original.ask()
        original.tell(
            ObjectiveFeedback(candidate.candidate_id, (candidate.values[0] - 1) ** 2)
        )
    resumed = BoundedCoordinatePatternOptimizer.from_checkpoint(
        OptimizerCheckpoint.from_json(original.checkpoint().to_json())
    )
    while original.termination is None:
        left = original.ask()
        right = resumed.ask()
        assert left == right
        feedback = ObjectiveFeedback(left.candidate_id, (left.values[0] - 1) ** 2)
        assert original.tell(feedback) == resumed.tell(feedback)
    assert resumed.termination == original.termination
    assert resumed.observations == original.observations
    assert resumed.checkpoint() == original.checkpoint()
