"""Behaviorful optimizer, forward-evaluator, and results-handler ports."""

from __future__ import annotations

from typing import Protocol

from .checkpoint import OptimizerCheckpoint
from .identity import OptimizerDescriptor
from .loss import WeightedSquaredLoss
from .records import (
    Candidate,
    ObjectiveFeedback,
    OptimizerUpdate,
    RawEvaluation,
)


class OptimizationEngine(Protocol):
    """Ask/tell optimizer role with explicit checkpoint state."""

    @property
    def descriptor(self) -> OptimizerDescriptor:
        """Return exact algorithm and checkpoint identity."""
        ...

    @property
    def run_id(self) -> str:
        """Return the identity of this optimizer run instance."""
        ...

    def ask(self) -> Candidate:
        """Propose the next candidate according to owned state."""
        ...

    def tell(self, feedback: ObjectiveFeedback, /) -> OptimizerUpdate:
        """Apply feedback for an exact outstanding candidate."""
        ...

    def checkpoint(self) -> OptimizerCheckpoint:
        """Return immutable versioned serializable state."""
        ...


class ForwardEvaluator(Protocol):
    """Injected side-effect boundary returning raw candidate observations."""

    def evaluate(self, candidate: Candidate, /) -> RawEvaluation:
        """Evaluate without updating the optimizer."""
        ...


class ResultsHandler(Protocol):
    """Separate port that retains raw results, derives feedback, and calls tell."""

    def handle(
        self,
        candidate: Candidate,
        evaluation: RawEvaluation,
        /,
    ) -> OptimizerUpdate:
        """Deliver derived feedback to the handler's exact bound engine instance."""
        ...


class WeightedResultsHandler:
    """In-memory results handler for one engine and weighted-loss transform.

    This object has no persistence or execution authority. Construction binds the
    exact proposing engine object; candidate and evaluation identities are checked
    before feedback is delivered to that same object.
    """

    __slots__ = ("_engine", "_loss", "_observations")

    def __init__(self, engine: OptimizationEngine, loss: WeightedSquaredLoss) -> None:
        if type(loss) is not WeightedSquaredLoss:
            raise TypeError("loss must be WeightedSquaredLoss")
        self._engine = engine
        self._loss = loss
        self._observations: list[tuple[Candidate, RawEvaluation]] = []

    @property
    def observations(self) -> tuple[tuple[Candidate, RawEvaluation], ...]:
        """Return retained immutable candidate/raw-evaluation pairs."""
        return tuple(self._observations)

    def handle(
        self,
        candidate: Candidate,
        evaluation: RawEvaluation,
        /,
    ) -> OptimizerUpdate:
        """Derive feedback and call tell on the exact bound proposing engine."""
        if candidate.optimizer_run_id != self._engine.run_id:
            raise ValueError("candidate was not proposed by this optimizer run")
        if evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("evaluation and candidate identities differ")
        weighted = self._loss.evaluate(evaluation)
        update = self._engine.tell(
            ObjectiveFeedback(
                candidate.candidate_id,
                weighted.value,
                weighted.objectives,
            )
        )
        self._observations.append((candidate, evaluation))
        return update
