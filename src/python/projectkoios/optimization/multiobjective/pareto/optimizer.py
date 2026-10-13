"""Iterative Pareto optimizer with interchangeable parameter samplers.

This provenance-bound optimizer originates in Eugene J. Ragasa's PhD work preserved
across PyPosPack and multiple research repositories. Do not move, split, or refactor it
without explicit human authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...engines.records import Candidate
from ...exceptions import (
    DuplicateObservationError,
    OptimizerExhaustedError,
    OptimizerStateError,
    UnknownCandidateError,
)
from ...spaces.parameters import ParameterSpace
from ...validation.validation import (
    candidate_identifier,
    finite_float,
    run_identifier,
    u32,
    u64,
)
from ..base import MultiObjectiveOptimizer
from .algorithm import pareto


@runtime_checkable
class ParameterSampler(Protocol):
    """Generate decision vectors from the preceding Pareto set and front."""

    def sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: int,
        pareto_set: tuple[tuple[float, ...], ...],
        pareto_front: tuple[tuple[float, ...], ...],
        evaluated_decisions: tuple[tuple[float, ...], ...] = (),
        evaluated_objectives: tuple[tuple[float, ...], ...] = (),
    ) -> tuple[tuple[float, ...], ...]:
        """Return exactly ``sample_count`` ordered decision vectors."""
        ...


@dataclass(frozen=True, slots=True)
class ParetoSamplingStage:
    """One parameter sampler and its exact candidate count."""

    sampler: ParameterSampler
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, ParameterSampler):
            raise TypeError("sampler must implement ParameterSampler")
        u32(self.sample_count, "sample_count", allow_zero=False)


class ParetoOptimizer(MultiObjectiveOptimizer):
    """Sample decision vectors and retain their nondominated pullback."""

    __slots__ = (
        "_all_candidates",
        "_all_objectives",
        "_archive",
        "_chunk_size",
        "_next_value_index",
        "_next_values",
        "_objective_dimension",
        "_outstanding",
        "_run_id",
        "_seed",
        "_space",
        "_stage_candidates",
        "_stage_index",
        "_stage_objectives",
        "_stages",
    )

    def __init__(
        self,
        space: ParameterSpace,
        stages: tuple[ParetoSamplingStage, ...],
        *,
        seed: int,
        run_id: str,
        chunk_size: int = 500,
    ) -> None:
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        if (
            type(stages) is not tuple
            or not stages
            or any(type(stage) is not ParetoSamplingStage for stage in stages)
        ):
            raise TypeError("stages must be a nonempty tuple of ParetoSamplingStage")
        if type(chunk_size) is not int:
            raise TypeError("chunk_size must be a built-in integer")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        proposal_count = sum(stage.sample_count for stage in stages)
        self._space = space
        self._stages = stages
        self._seed = u64(seed, "seed")
        self._run_id = run_identifier(run_id, proposal_count - 1)
        self._chunk_size = chunk_size
        self._stage_index = 0
        self._next_values: tuple[tuple[float, ...], ...] = ()
        self._next_value_index = 0
        self._stage_candidates: list[Candidate] = []
        self._stage_objectives: dict[str, tuple[float, ...]] = {}
        self._all_candidates: dict[str, Candidate] = {}
        self._all_objectives: dict[str, tuple[float, ...]] = {}
        self._outstanding: dict[str, Candidate] = {}
        self._archive: tuple[tuple[Candidate, tuple[float, ...]], ...] = ()
        self._objective_dimension: int | None = None

    @property
    def run_id(self) -> str:
        """Return this optimizer-run identity."""
        return self._run_id

    @property
    def archive(self) -> tuple[tuple[Candidate, tuple[float, ...]], ...]:
        """Return nondominated candidate/objective pairs."""
        return self._archive

    @property
    def outstanding(self) -> tuple[Candidate, ...]:
        """Return outstanding candidates in proposal order."""
        return tuple(sorted(self._outstanding.values(), key=lambda item: item.sequence))

    @property
    def exhausted(self) -> bool:
        """Return whether every configured sampling stage is complete."""
        return self._stage_index == len(self._stages)

    def ask(self) -> Candidate:
        """Return the next sampled decision vector."""
        if self.exhausted:
            raise OptimizerExhaustedError("Pareto sampling stages are exhausted")
        if not self._next_values and not self._stage_candidates:
            self._next_values = self._sample_stage()
            self._next_value_index = 0
        if self._next_value_index == len(self._next_values):
            raise OptimizerStateError(
                "current stage needs all outstanding objective vectors"
            )
        values = self._next_values[self._next_value_index]
        self._next_value_index += 1
        sequence = len(self._all_candidates)
        candidate = Candidate(
            candidate_identifier(self._run_id, sequence),
            self._run_id,
            sequence,
            values,
        )
        self._stage_candidates.append(candidate)
        self._all_candidates[candidate.candidate_id] = candidate
        self._outstanding[candidate.candidate_id] = candidate
        return candidate

    def tell(
        self,
        candidate: Candidate,
        objectives: tuple[float, ...],
        /,
    ) -> None:
        """Accept an objective vector for an exact outstanding candidate."""
        if type(candidate) is not Candidate:
            raise TypeError("candidate must be Candidate")
        if candidate.candidate_id in self._all_objectives:
            raise DuplicateObservationError(
                "candidate objectives were already accepted"
            )
        expected = self._outstanding.get(candidate.candidate_id)
        if expected is None:
            raise UnknownCandidateError("candidate is not outstanding for this run")
        if candidate != expected:
            raise ValueError("candidate differs from the outstanding proposal")
        if type(objectives) is not tuple or not objectives:
            raise TypeError("objectives must be a nonempty tuple")
        normalized = tuple(
            finite_float(value, "objective value") for value in objectives
        )
        if self._objective_dimension is None:
            self._objective_dimension = len(normalized)
        elif len(normalized) != self._objective_dimension:
            raise ValueError("objective-vector dimension must remain constant")

        del self._outstanding[candidate.candidate_id]
        self._stage_objectives[candidate.candidate_id] = normalized
        self._all_objectives[candidate.candidate_id] = normalized
        if self._next_value_index == len(self._next_values) and len(
            self._stage_objectives
        ) == len(self._stage_candidates):
            self._complete_stage()

    def _sample_stage(self) -> tuple[tuple[float, ...], ...]:
        stage = self._stages[self._stage_index]
        evaluated_candidates = tuple(
            sorted(self._all_candidates.values(), key=lambda item: item.sequence)
        )
        values = stage.sampler.sample(
            stage.sample_count,
            random_seed=(self._seed + self._stage_index) & ((1 << 64) - 1),
            pareto_set=self.pareto_set,
            pareto_front=self.pareto_front,
            evaluated_decisions=tuple(
                candidate.values for candidate in evaluated_candidates
            ),
            evaluated_objectives=tuple(
                self._all_objectives[candidate.candidate_id]
                for candidate in evaluated_candidates
            ),
        )
        if type(values) is not tuple or len(values) != stage.sample_count:
            raise ValueError("sampler returned the wrong sample count")
        for value in values:
            if type(value) is not tuple:
                raise TypeError("sampled decision vectors must be tuples")
            if not self._space.admits(value):
                raise ValueError(
                    "sampled decision vector lies outside the parameter space"
                )
        return values

    def _complete_stage(self) -> None:
        evaluated = self._archive + tuple(
            (candidate, self._stage_objectives[candidate.candidate_id])
            for candidate in self._stage_candidates
        )
        selected = pareto(
            tuple(objectives for _, objectives in evaluated),
            chunk_sz=self._chunk_size,
        )
        self._archive = tuple(evaluated[index] for index in selected)
        self._stage_index += 1
        self._next_values = ()
        self._next_value_index = 0
        self._stage_candidates = []
        self._stage_objectives = {}


__all__ = ["ParameterSampler", "ParetoOptimizer", "ParetoSamplingStage"]
