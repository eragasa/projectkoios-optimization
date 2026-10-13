import pytest

from projectkoios.optimization.algorithms.coordinate import (
    CoordinateSearch,
    CoordinateSearchSettings,
)
from projectkoios.optimization.engines.engine import WeightedResultsHandler
from projectkoios.optimization.engines.loss import WeightedLossTerm, WeightedSquaredLoss
from projectkoios.optimization.engines.records import Measurement, RawEvaluation
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


def make(run_id: str) -> CoordinateSearch:
    return CoordinateSearch(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        (0.5,),
        CoordinateSearchSettings(1, 0, 0.25, 0.5, 0.01),
        run_id=run_id,
    )


def test_handler_scores_raw_results_and_updates_the_proposing_engine() -> None:
    optimizer = make("proposer")
    candidate = optimizer.ask()
    handler = WeightedResultsHandler(
        optimizer,
        WeightedSquaredLoss((WeightedLossTerm("q", 2.0, 3.0),)),
    )
    raw = RawEvaluation(candidate.candidate_id, (Measurement("q", 1.0),))

    update = handler.handle(candidate, raw)

    assert update.best_candidate_id == candidate.candidate_id
    assert optimizer.observations[0][1].loss == 3.0
    assert handler.observations == ((candidate, raw),)


def test_handler_rejects_results_for_a_different_candidate() -> None:
    optimizer = make("proposer")
    candidate = optimizer.ask()
    handler = WeightedResultsHandler(
        optimizer,
        WeightedSquaredLoss((WeightedLossTerm("q", 0.0, 1.0),)),
    )

    with pytest.raises(ValueError, match="identities differ"):
        handler.handle(candidate, RawEvaluation("wrong", (Measurement("q", 0.0),)))
