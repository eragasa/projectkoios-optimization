"""Resumable bounded coordinate-pattern ask/tell minimizer."""

from __future__ import annotations

from dataclasses import dataclass

from ..engines.records import Candidate, ObjectiveFeedback, OptimizerUpdate
from ..exceptions import (
    DuplicateObservationError,
    OptimizerExhaustedError,
    OptimizerStateError,
    UnknownCandidateError,
)
from ..spaces.parameters import ParameterSpace
from ..validation.validation import (
    candidate_identifier,
    finite_float,
    run_identifier,
    u32,
)


@dataclass(frozen=True, slots=True)
class CoordinateSearchSettings:
    """Finite budgets and normalized step controls for coordinate search."""

    max_evaluations: int
    max_iterations: int
    initial_step_fraction: float
    step_reduction: float
    step_tolerance: float

    def __post_init__(self) -> None:
        u32(self.max_evaluations, "max_evaluations", allow_zero=False)
        u32(self.max_iterations, "max_iterations", allow_zero=True)
        for name in (
            "initial_step_fraction",
            "step_reduction",
            "step_tolerance",
        ):
            object.__setattr__(self, name, finite_float(getattr(self, name), name))
        if not 0.0 < self.initial_step_fraction <= 1.0:
            raise ValueError("initial_step_fraction must be in (0, 1]")
        if not 0.0 < self.step_reduction < 1.0:
            raise ValueError("step_reduction must be in (0, 1)")
        if not 0.0 < self.step_tolerance < self.initial_step_fraction:
            raise ValueError(
                "step_tolerance must be positive and below initial_step_fraction"
            )


@dataclass(frozen=True, slots=True)
class CoordinateSearchState:
    """Immutable intrinsic state of a bounded coordinate-pattern optimizer.

    The record contains typed algorithm inputs and observations only. External wrappers
    may adapt it for persistence without affecting the optimization contract.
    """

    space: ParameterSpace
    initial_values: tuple[float, ...]
    settings: CoordinateSearchSettings
    run_id: str
    feedback: tuple[ObjectiveFeedback, ...]
    outstanding: Candidate | None

    def __post_init__(self) -> None:
        """Validate the state record's intrinsic structural invariants."""
        if type(self.space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        if type(self.initial_values) is not tuple:
            raise TypeError("initial_values must be a tuple")
        normalized = Candidate("initial", "initial", 0, self.initial_values).values
        object.__setattr__(self, "initial_values", normalized)
        if not self.space.admits(normalized):
            raise ValueError("initial_values must match dimensions and bounds")
        if type(self.settings) is not CoordinateSearchSettings:
            raise TypeError("settings must be CoordinateSearchSettings")
        run_identifier(self.run_id, self.settings.max_evaluations - 1)
        if type(self.feedback) is not tuple or any(
            type(item) is not ObjectiveFeedback for item in self.feedback
        ):
            raise TypeError("feedback must be a tuple of ObjectiveFeedback")
        if self.outstanding is not None and type(self.outstanding) is not Candidate:
            raise TypeError("outstanding must be Candidate or None")


class CoordinateSearch:
    """Deterministic local optimizer with one outstanding candidate.

    The initial point is evaluated first. Each iteration evaluates feasible negative
    then positive neighbors in coordinate order. Only strict improvement is accepted;
    ties retain the earlier proposal. Repeated :meth:`ask` calls while a candidate is
    outstanding return that same immutable candidate.
    """

    __slots__ = (
        "_feedback",
        "_history",
        "_incumbent",
        "_incumbent_loss",
        "_initial",
        "_iteration",
        "_iteration_best",
        "_iteration_best_loss",
        "_neighbor_position",
        "_neighbor_values",
        "_outstanding",
        "_run_id",
        "_settings",
        "_space",
        "_steps",
        "_termination",
    )

    def __init__(
        self,
        space: ParameterSpace,
        initial_values: tuple[float, ...],
        settings: CoordinateSearchSettings,
        *,
        run_id: str,
    ) -> None:
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        if type(initial_values) is not tuple:
            raise TypeError("initial_values must be a tuple")
        initial = Candidate("initial", "initial", 0, initial_values).values
        if not space.admits(initial):
            raise ValueError("initial_values must match dimensions and bounds")
        if type(settings) is not CoordinateSearchSettings:
            raise TypeError("settings must be CoordinateSearchSettings")
        normalized_run_id = run_identifier(run_id, settings.max_evaluations - 1)
        self._space = space
        self._initial = initial
        self._settings = settings
        self._run_id = normalized_run_id
        self._history: list[Candidate] = []
        self._feedback: list[ObjectiveFeedback] = []
        self._outstanding: Candidate | None = None
        self._incumbent: Candidate | None = None
        self._incumbent_loss: float | None = None
        self._steps = tuple(
            settings.initial_step_fraction * b.width for b in space.bounds
        )
        self._iteration = 0
        self._neighbor_values: tuple[tuple[float, ...], ...] = ()
        self._neighbor_position = 0
        self._iteration_best: Candidate | None = None
        self._iteration_best_loss: float | None = None
        self._termination: str | None = None

    @property
    def run_id(self) -> str:
        """Return this optimizer-run identity."""
        return self._run_id

    @property
    def termination(self) -> str | None:
        """Return the closed termination value, if search has ended."""
        return self._termination

    @property
    def iterations(self) -> int:
        """Return completed or budget-truncated pattern iterations."""
        return self._iteration

    @property
    def observations(self) -> tuple[tuple[Candidate, ObjectiveFeedback], ...]:
        """Return accepted candidate/feedback pairs in evaluation order."""
        return tuple(zip(self._history, self._feedback, strict=True))

    @property
    def best_candidate(self) -> Candidate | None:
        """Return the best fully observed candidate so far."""
        if self._iteration_best is not None:
            return self._iteration_best
        return self._incumbent

    @property
    def best_loss(self) -> float | None:
        """Return the best finite loss observed so far."""
        if self._iteration_best_loss is not None:
            return self._iteration_best_loss
        return self._incumbent_loss

    def ask(self) -> Candidate:
        """Return the pending or next candidate, failing after termination."""
        if self._outstanding is not None:
            return self._outstanding
        if self._termination is not None:
            raise OptimizerExhaustedError(f"pattern search ended: {self._termination}")
        if self._incumbent is None:
            return self._propose(self._initial)
        self._prepare_iteration()
        if self._termination is not None:
            raise OptimizerExhaustedError(f"pattern search ended: {self._termination}")
        values = self._neighbor_values[self._neighbor_position]
        return self._propose(values)

    def _propose(self, values: tuple[float, ...]) -> Candidate:
        sequence = len(self._history)
        candidate = Candidate(
            candidate_identifier(self._run_id, sequence),
            self._run_id,
            sequence,
            values,
        )
        self._outstanding = candidate
        return candidate

    def tell(self, feedback: ObjectiveFeedback, /) -> OptimizerUpdate:
        """Accept feedback for the sole outstanding candidate exactly once."""
        if type(feedback) is not ObjectiveFeedback:
            raise TypeError("feedback must be ObjectiveFeedback")
        if any(value.candidate_id == feedback.candidate_id for value in self._feedback):
            raise DuplicateObservationError("candidate feedback was already accepted")
        if (
            self._outstanding is None
            or feedback.candidate_id != self._outstanding.candidate_id
        ):
            raise UnknownCandidateError("candidate is not outstanding for this run")
        candidate = self._outstanding
        self._outstanding = None
        self._history.append(candidate)
        self._feedback.append(feedback)

        if self._incumbent is None:
            self._incumbent = candidate
            self._incumbent_loss = feedback.loss
            self._prepare_iteration()
        else:
            self._accept_neighbor(candidate, feedback.loss)

        best = self.best_candidate
        if best is None:
            raise RuntimeError("accepted feedback did not establish a best candidate")
        return OptimizerUpdate(
            feedback.candidate_id,
            len(self._feedback),
            best.candidate_id,
            self._termination is not None,
            self._termination,
        )

    def _accept_neighbor(self, candidate: Candidate, loss: float) -> None:
        if self._iteration_best_loss is None or self._iteration_best is None:
            raise RuntimeError("pattern iteration has no incumbent")
        if loss < self._iteration_best_loss:
            self._iteration_best = candidate
            self._iteration_best_loss = loss
        self._neighbor_position += 1
        if len(self._history) >= self._settings.max_evaluations:
            self._finish_iteration()
            self._termination = "evaluation_limit"
            return
        if self._neighbor_position == len(self._neighbor_values):
            self._finish_iteration()
            self._prepare_iteration()

    def _finish_iteration(self) -> None:
        if (
            self._incumbent_loss is None
            or self._incumbent is None
            or self._iteration_best_loss is None
            or self._iteration_best is None
        ):
            raise RuntimeError("cannot finish an uninitialized pattern iteration")
        if self._iteration_best_loss < self._incumbent_loss:
            self._incumbent = self._iteration_best
            self._incumbent_loss = self._iteration_best_loss
        else:
            self._steps = tuple(
                step * self._settings.step_reduction for step in self._steps
            )
        self._iteration += 1
        self._neighbor_values = ()
        self._neighbor_position = 0
        self._iteration_best = None
        self._iteration_best_loss = None

    def _prepare_iteration(self) -> None:
        if self._termination is not None or self._neighbor_values:
            return
        if self._incumbent is None or self._incumbent_loss is None:
            return
        active = tuple(
            step / bound.width
            for step, bound in zip(self._steps, self._space.bounds, strict=True)
            if bound.width > 0.0
        )
        if not active or max(active) <= self._settings.step_tolerance:
            self._termination = "step_tolerance"
            return
        if len(self._history) >= self._settings.max_evaluations:
            self._termination = "evaluation_limit"
            return
        if self._iteration >= self._settings.max_iterations:
            self._termination = "iteration_limit"
            return
        neighbors: list[tuple[float, ...]] = []
        for index, (bound, step) in enumerate(
            zip(self._space.bounds, self._steps, strict=True)
        ):
            if bound.width == 0.0:
                continue
            for direction in (-1.0, 1.0):
                values = list(self._incumbent.values)
                coordinate = values[index]
                if direction < 0.0:
                    distance = coordinate - bound.lower
                    values[index] = (
                        bound.lower if step >= distance else coordinate - step
                    )
                else:
                    distance = bound.upper - coordinate
                    values[index] = (
                        bound.upper if step >= distance else coordinate + step
                    )
                candidate = tuple(values)
                if candidate != self._incumbent.values and candidate not in neighbors:
                    neighbors.append(candidate)
        if not neighbors:
            self._termination = "step_tolerance"
            return
        self._neighbor_values = tuple(neighbors)
        self._neighbor_position = 0
        self._iteration_best = self._incumbent
        self._iteration_best_loss = self._incumbent_loss

    def state(self) -> CoordinateSearchState:
        """Return replay-complete intrinsic optimizer state.

        Returns
        -------
        CoordinateSearchState
            Concrete immutable state with no persistence or encoding behavior.
        """
        return CoordinateSearchState(
            space=self._space,
            initial_values=self._initial,
            settings=self._settings,
            run_id=self._run_id,
            feedback=tuple(self._feedback),
            outstanding=self._outstanding,
        )

    @classmethod
    def from_state(
        cls,
        state: CoordinateSearchState,
    ) -> CoordinateSearch:
        """Restore an engine through deterministic replay of intrinsic state.

        Parameters
        ----------
        state
            Exact typed record returned by :meth:`state`.

        Returns
        -------
        CoordinateSearch
            Restored engine with the same observations and pending proposal.

        Raises
        ------
        TypeError
            If ``state`` has the wrong concrete type.
        OptimizerStateError
            If otherwise valid records do not form reachable optimizer state.
        """
        if type(state) is not CoordinateSearchState:
            raise TypeError("state must be CoordinateSearchState")
        try:
            engine = cls(
                state.space,
                state.initial_values,
                state.settings,
                run_id=state.run_id,
            )
            for feedback in state.feedback:
                expected = engine.ask()
                if expected.candidate_id != feedback.candidate_id:
                    raise OptimizerStateError("state feedback order is incompatible")
                engine.tell(feedback)
            if state.outstanding is not None:
                expected = engine.ask()
                if expected != state.outstanding:
                    raise OptimizerStateError("state pending candidate is incompatible")
        except (
            TypeError,
            ValueError,
            OptimizerExhaustedError,
            UnknownCandidateError,
            DuplicateObservationError,
        ) as error:
            raise OptimizerStateError("pattern state replay failed") from error
        return engine
