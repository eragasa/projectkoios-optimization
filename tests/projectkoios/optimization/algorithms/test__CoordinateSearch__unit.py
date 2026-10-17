import pytest

pytestmark = pytest.mark.unit

from projectkoios.optimization.algorithms.coordinate import (  # noqa: E402
    CoordinateSearch,
    CoordinateSearchSettings,
    CoordinateSearchState,
)
from projectkoios.optimization.engines.records import ObjectiveFeedback  # noqa: E402
from projectkoios.optimization.exceptions import (  # noqa: E402
    DuplicateObservationError,
    OptimizerExhaustedError,
    UnknownCandidateError,
)
from projectkoios.optimization.spaces.parameters import (  # noqa: E402
    ParameterBound,
    ParameterSpace,
)


def settings(
    max_evaluations: int = 20, max_iterations: int = 10
) -> CoordinateSearchSettings:
    return CoordinateSearchSettings(max_evaluations, max_iterations, 0.25, 0.5, 0.01)


def test_settings_require_canonical_frozen_construction() -> None:
    settings_record = CoordinateSearchSettings(20, 10, 1.0, 0.5, 0.01)

    assert settings_record.initial_step_fraction == 1.0
    assert type(settings_record.initial_step_fraction) is float
    with pytest.raises(TypeError, match="exact built-in float"):
        CoordinateSearchSettings(20, 10, 1, 0.5, 0.01)


def test_state_requires_canonical_frozen_construction() -> None:
    space = ParameterSpace((ParameterBound("x", -1.0, 1.0),))
    state = CoordinateSearchState(
        space=space,
        initial_values=(0.0,),
        settings=settings(),
        run_id="pattern.state",
        feedback=(),
        outstanding=None,
    )

    assert state.initial_values == (0.0,)
    assert type(state.initial_values[0]) is float
    with pytest.raises(TypeError, match="exact built-in float"):
        CoordinateSearchState(
            space=space,
            initial_values=(0,),
            settings=settings(),
            run_id="pattern.state",
            feedback=(),
            outstanding=None,
        )


def test_repeated_ask_returns_same_candidate_and_wrong_tell_fails() -> None:
    optimizer = CoordinateSearch(
        ParameterSpace((ParameterBound("x", -4.0, 4.0),)),
        (0.0,),
        settings(),
        run_id="pattern.repeat",
    )
    first = optimizer.ask()
    assert optimizer.ask() is first
    with pytest.raises(UnknownCandidateError):
        optimizer.tell(ObjectiveFeedback("unknown", 1.0))
    optimizer.tell(ObjectiveFeedback(first.candidate_id, 1.0))
    with pytest.raises(DuplicateObservationError):
        optimizer.tell(ObjectiveFeedback(first.candidate_id, 1.0))


def test_budget_exhaustion_is_exact() -> None:
    optimizer = CoordinateSearch(
        ParameterSpace((ParameterBound("x", -1.0, 1.0),)),
        (0.0,),
        settings(max_evaluations=1),
        run_id="pattern.budget",
    )
    candidate = optimizer.ask()
    update = optimizer.tell(ObjectiveFeedback(candidate.candidate_id, 1.0))
    assert update.termination == "evaluation_limit"
    assert len(optimizer.observations) == 1
    with pytest.raises(OptimizerExhaustedError):
        optimizer.ask()


def test_zero_dimensions_and_all_fixed_coordinates_converge_after_initial() -> None:
    for space, initial in (
        (ParameterSpace(()), ()),
        (ParameterSpace((ParameterBound("x", 2.0, 2.0),)), (2.0,)),
    ):
        optimizer = CoordinateSearch(
            space, initial, settings(), run_id=f"fixed.{space.dimension}"
        )
        candidate = optimizer.ask()
        update = optimizer.tell(ObjectiveFeedback(candidate.candidate_id, 0.0))
        assert update.termination == "step_tolerance"
        assert optimizer.iterations == 0


def test_ties_retain_incumbent_and_iteration_limit_is_exact() -> None:
    optimizer = CoordinateSearch(
        ParameterSpace((ParameterBound("x", -1.0, 1.0),)),
        (0.0,),
        settings(max_evaluations=10, max_iterations=1),
        run_id="pattern.tie",
    )
    initial = optimizer.ask()
    optimizer.tell(ObjectiveFeedback(initial.candidate_id, 1.0))
    while optimizer.termination is None:
        candidate = optimizer.ask()
        update = optimizer.tell(ObjectiveFeedback(candidate.candidate_id, 1.0))
    assert update.termination == "iteration_limit"
    assert optimizer.best_candidate == initial
    assert optimizer.iterations == 1


def test_state_with_pending_candidate_restores_exactly() -> None:
    optimizer = CoordinateSearch(
        ParameterSpace((ParameterBound("x", -4.0, 4.0),)),
        (0.0,),
        settings(),
        run_id="pattern.resume",
    )
    first = optimizer.ask()
    optimizer.tell(ObjectiveFeedback(first.candidate_id, 4.0))
    pending = optimizer.ask()
    restored = CoordinateSearch.from_state(optimizer.state())
    assert restored.ask() == pending
    assert restored.state() == optimizer.state()
