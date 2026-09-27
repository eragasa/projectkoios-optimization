import pytest

from projectkoios.optimization import (
    BoundedUniformOptimizer,
    Measurement,
    ParameterBound,
    ParameterSpace,
    RawEvaluation,
    WeightedLossTerm,
    WeightedResultsHandler,
    WeightedSquaredLoss,
)


def make(run_id: str) -> BoundedUniformOptimizer:
    return BoundedUniformOptimizer(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        seed=0,
        max_evaluations=1,
        run_id=run_id,
    )


def test_results_handler_retains_raw_observation_and_tells_proposing_engine() -> None:
    optimizer = make("proposer")
    candidate = optimizer.ask()
    handler = WeightedResultsHandler(
        optimizer,
        WeightedSquaredLoss((WeightedLossTerm("q", 2.0, 3.0),)),
    )
    raw = RawEvaluation(candidate.candidate_id, (Measurement("q", 1.0),))

    update = handler.handle(candidate, raw)

    assert update.best_candidate_id == candidate.candidate_id
    assert optimizer.feedback[0].loss == 3.0
    assert handler.observations == ((candidate, raw),)


def test_results_handler_is_bound_to_exact_engine_instance_and_correlation() -> None:
    proposer = make("same-run")
    other = make("same-run")
    candidate = proposer.ask()
    assert other.ask() == candidate
    handler = WeightedResultsHandler(
        proposer,
        WeightedSquaredLoss((WeightedLossTerm("q", 0.0, 1.0),)),
    )
    with pytest.raises(ValueError, match="identities differ"):
        handler.handle(
            candidate,
            RawEvaluation("wrong", (Measurement("q", 0.0),)),
        )
    handler.handle(
        candidate,
        RawEvaluation(candidate.candidate_id, (Measurement("q", 0.0),)),
    )
    assert len(proposer.feedback) == 1
    assert other.feedback == ()
