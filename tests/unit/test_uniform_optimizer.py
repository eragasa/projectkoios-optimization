import pytest

from projectkoios.optimization import (
    BoundedUniformOptimizer,
    DuplicateObservationError,
    ObjectiveFeedback,
    OptimizerCheckpoint,
    OptimizerExhaustedError,
    ParameterBound,
    ParameterSpace,
    UnknownCandidateError,
)


def engine() -> BoundedUniformOptimizer:
    return BoundedUniformOptimizer(
        ParameterSpace((ParameterBound("x", -1.0, 1.0),)),
        seed=17,
        max_evaluations=3,
        run_id="uniform.test",
    )


def test_multiple_asks_allow_out_of_order_tells_and_deterministic_ties() -> None:
    optimizer = engine()
    first, second, third = optimizer.ask(), optimizer.ask(), optimizer.ask()
    assert tuple(item.sequence for item in optimizer.outstanding) == (0, 1, 2)

    optimizer.tell(ObjectiveFeedback(third.candidate_id, 1.0))
    optimizer.tell(ObjectiveFeedback(second.candidate_id, 1.0))
    final = optimizer.tell(ObjectiveFeedback(first.candidate_id, 1.0))

    assert final.best_candidate_id == first.candidate_id
    assert final.termination == "evaluation_limit"
    assert optimizer.exhausted
    with pytest.raises(OptimizerExhaustedError):
        optimizer.ask()


def test_unknown_and_duplicate_tell_fail_closed() -> None:
    optimizer = engine()
    candidate = optimizer.ask()
    with pytest.raises(UnknownCandidateError):
        optimizer.tell(ObjectiveFeedback("other", 1.0))
    optimizer.tell(ObjectiveFeedback(candidate.candidate_id, 1.0))
    with pytest.raises(DuplicateObservationError):
        optimizer.tell(ObjectiveFeedback(candidate.candidate_id, 1.0))


def test_checkpoint_round_trip_preserves_outstanding_and_future_sequence() -> None:
    optimizer = engine()
    first, second = optimizer.ask(), optimizer.ask()
    optimizer.tell(ObjectiveFeedback(second.candidate_id, 2.0))
    checkpoint = optimizer.checkpoint()
    restored = BoundedUniformOptimizer.from_checkpoint(
        OptimizerCheckpoint.from_json(checkpoint.to_json())
    )

    assert restored.outstanding == (first,)
    assert restored.feedback == optimizer.feedback
    assert restored.ask() == optimizer.ask()
    assert restored.checkpoint() == optimizer.checkpoint()


def test_empty_space_and_fixed_coordinate_sampling() -> None:
    empty = BoundedUniformOptimizer(
        ParameterSpace(()), seed=0, max_evaluations=1, run_id="empty"
    )
    fixed = BoundedUniformOptimizer(
        ParameterSpace((ParameterBound("x", 4.0, 4.0),)),
        seed=99,
        max_evaluations=1,
        run_id="fixed",
    )
    assert empty.ask().values == ()
    assert fixed.ask().values == (4.0,)
