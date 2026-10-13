import pytest

from projectkoios.optimization.algorithms.coordinate import (
    CoordinateSearch,
    CoordinateSearchSettings,
)
from projectkoios.optimization.engines.records import ObjectiveFeedback
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.property


def test_pattern_resumed_and_uninterrupted_are_exactly_equivalent() -> None:
    original = CoordinateSearch(
        ParameterSpace((ParameterBound("x", -4.0, 4.0),)),
        (0.0,),
        CoordinateSearchSettings(40, 20, 0.25, 0.5, 0.001),
        run_id="p",
    )
    for _ in range(6):
        candidate = original.ask()
        original.tell(
            ObjectiveFeedback(candidate.candidate_id, (candidate.values[0] - 1) ** 2)
        )
    resumed = CoordinateSearch.from_state(original.state())
    while original.termination is None:
        left = original.ask()
        right = resumed.ask()
        assert left == right
        feedback = ObjectiveFeedback(left.candidate_id, (left.values[0] - 1) ** 2)
        assert original.tell(feedback) == resumed.tell(feedback)
    assert resumed.termination == original.termination
    assert resumed.observations == original.observations
    assert resumed.state() == original.state()
