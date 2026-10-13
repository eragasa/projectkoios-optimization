"""Pure deterministic optimization result and Pareto records."""

from __future__ import annotations

from dataclasses import dataclass

from ..engines.records import Candidate, ObjectiveFeedback
from ..multiobjective.pareto.algorithm import pareto


@dataclass(frozen=True, slots=True)
class ScoredResult:
    """One correlated candidate and finite objective feedback."""

    candidate: Candidate
    feedback: ObjectiveFeedback

    def __post_init__(self) -> None:
        if type(self.candidate) is not Candidate:
            raise TypeError("candidate must be Candidate")
        if type(self.feedback) is not ObjectiveFeedback:
            raise TypeError("feedback must be ObjectiveFeedback")
        if self.candidate.candidate_id != self.feedback.candidate_id:
            raise ValueError("candidate and feedback identities must agree")


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Canonical scalar ranking and nondominated candidate identities."""

    ranked: tuple[ScoredResult, ...]
    pareto_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.ranked) is not tuple or any(
            type(value) is not ScoredResult for value in self.ranked
        ):
            raise TypeError("ranked must be a tuple of ScoredResult")
        expected = tuple(
            sorted(
                self.ranked,
                key=lambda value: (
                    value.feedback.loss,
                    value.candidate.candidate_id,
                ),
            )
        )
        if self.ranked != expected:
            raise ValueError("ranked results must use loss then candidate identity")
        if type(self.pareto_candidate_ids) is not tuple or any(
            type(value) is not str for value in self.pareto_candidate_ids
        ):
            raise TypeError("pareto_candidate_ids must be a tuple of strings")
        if self.pareto_candidate_ids != tuple(sorted(self.pareto_candidate_ids)):
            raise ValueError("Pareto candidate identities must be sorted")
        represented = {value.candidate.candidate_id for value in self.ranked}
        if not set(self.pareto_candidate_ids) <= represented:
            raise ValueError("Pareto identities must be represented in ranked results")


def build_report(results: tuple[ScoredResult, ...]) -> OptimizationReport:
    """Rank results and derive Pareto membership with deterministic tie policy."""
    if type(results) is not tuple or any(
        type(value) is not ScoredResult for value in results
    ):
        raise TypeError("results must be a tuple of ScoredResult")
    ids = tuple(value.candidate.candidate_id for value in results)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate identities must be unique")
    if results:
        objective_names = tuple(value.name for value in results[0].feedback.objectives)
        for result in results:
            if (
                tuple(value.name for value in result.feedback.objectives)
                != objective_names
            ):
                raise ValueError("objective names and order must agree")
    objective_vectors = tuple(
        tuple(objective.value for objective in result.feedback.objectives)
        for result in results
    )
    pareto_candidate_ids = tuple(
        sorted(
            results[index].candidate.candidate_id for index in pareto(objective_vectors)
        )
    )
    ranked = tuple(
        sorted(
            results,
            key=lambda value: (value.feedback.loss, value.candidate.candidate_id),
        )
    )
    return OptimizationReport(ranked, pareto_candidate_ids)
