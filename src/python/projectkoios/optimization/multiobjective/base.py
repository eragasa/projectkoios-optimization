"""Behavioral base for forward-evaluated multi-objective optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..engines.records import Candidate


class MultiObjectiveOptimizer(ABC):
    """Ask for decision vectors and accept their ordered objective vectors."""

    __slots__ = ()

    @abstractmethod
    def ask(self) -> Candidate:
        """Return the next candidate solution to evaluate."""
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        candidate: Candidate,
        objectives: tuple[float, ...],
        /,
    ) -> None:
        """Accept one objective vector for an exact outstanding candidate."""
        raise NotImplementedError

    @property
    @abstractmethod
    def archive(self) -> tuple[tuple[Candidate, tuple[float, ...]], ...]:
        """Return nondominated candidate/objective pairs in deterministic order."""
        raise NotImplementedError

    @property
    def pareto_set(self) -> tuple[tuple[float, ...], ...]:
        """Return the nondominated decision-vector approximation."""
        return tuple(candidate.values for candidate, _ in self.archive)

    @property
    def pareto_front(self) -> tuple[tuple[float, ...], ...]:
        """Return the corresponding objective-vector approximation."""
        return tuple(objectives for _, objectives in self.archive)


__all__ = ["MultiObjectiveOptimizer"]
