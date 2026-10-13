"""Ports and handling behavior for the forward-evaluated optimization engine block.

This module belongs to the provenance-bound engine block originating in Eugene J.
Ragasa's PhD work and preserved across PyPosPack and multiple research repositories.
Do not move, split, or refactor this block without explicit human authorization.
"""

from __future__ import annotations

from typing import Protocol

from .loss import WeightedSquaredLoss
from .records import (
    Candidate,
    ObjectiveFeedback,
    OptimizerUpdate,
    RawEvaluation,
)


class OptimizationEngine(Protocol):
    """Numerical ask/tell engine independent of workflow lifecycle."""

    @property
    def run_id(self) -> str:
        """Return the identity of this optimizer run instance."""
        ...

    def ask(self) -> Candidate:
        """Return the next candidate according to intrinsic optimizer state."""
        ...

    def tell(self, feedback: ObjectiveFeedback, /) -> OptimizerUpdate:
        """Apply feedback for an exact outstanding candidate."""
        ...


class ForwardEvaluator(Protocol):
    """Injected simulation-calculator boundary returning raw observations."""

    def evaluate(self, candidate: Candidate, /) -> RawEvaluation:
        """Evaluate one candidate without updating the optimizer."""
        ...


class ResultsHandler(Protocol):
    """Retain raw results, derive feedback, and update the proposing engine."""

    def handle(
        self,
        candidate: Candidate,
        evaluation: RawEvaluation,
        /,
    ) -> OptimizerUpdate:
        """Deliver derived feedback to the exact proposing engine instance."""
        ...


class WeightedResultsHandler:
    """Apply weighted loss and deliver feedback to one exact engine instance."""

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
        """Derive feedback and call ``tell`` on the exact proposing engine."""
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


__all__ = [
    "ForwardEvaluator",
    "OptimizationEngine",
    "ResultsHandler",
    "WeightedResultsHandler",
]
