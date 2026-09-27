"""Resumable bounded coordinate-pattern ask/tell minimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._serialization import (
    decode_candidate,
    decode_feedback,
    decode_space,
    encode_candidate,
    encode_feedback,
    encode_space,
    integer_value,
    list_value,
    number_value,
    object_value,
    required,
    string_value,
)
from .checkpoint import JsonValue, OptimizerCheckpoint
from .exceptions import (
    CheckpointError,
    DuplicateObservationError,
    OptimizerExhaustedError,
    UnknownCandidateError,
)
from .identity import OptimizerDescriptor, OptimizerIdentity
from .records import Candidate, ObjectiveFeedback, OptimizerUpdate, ParameterSpace


@dataclass(frozen=True, slots=True)
class PatternSearchSettings:
    """Finite budgets and normalized step controls for coordinate search."""

    max_evaluations: int
    max_iterations: int
    initial_step_fraction: float
    step_reduction: float
    step_tolerance: float

    def __post_init__(self) -> None:
        if type(self.max_evaluations) is not int or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive built-in integer")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative built-in integer")
        for name in (
            "initial_step_fraction",
            "step_reduction",
            "step_tolerance",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a built-in int or float")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)
        if not 0.0 < self.initial_step_fraction <= 1.0:
            raise ValueError("initial_step_fraction must be in (0, 1]")
        if not 0.0 < self.step_reduction < 1.0:
            raise ValueError("step_reduction must be in (0, 1)")
        if not 0.0 < self.step_tolerance < self.initial_step_fraction:
            raise ValueError(
                "step_tolerance must be positive and below initial_step_fraction"
            )


class BoundedCoordinatePatternOptimizer:
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

    _DESCRIPTOR = OptimizerDescriptor(
        OptimizerIdentity("bounded-coordinate-pattern", 1),
        "bounded-coordinate-pattern:1",
        1,
    )

    def __init__(
        self,
        space: ParameterSpace,
        initial_values: tuple[float, ...],
        settings: PatternSearchSettings,
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
        if type(settings) is not PatternSearchSettings:
            raise TypeError("settings must be PatternSearchSettings")
        if type(run_id) is not str:
            raise TypeError("run_id must be a built-in string")
        if not run_id or run_id.strip() != run_id:
            raise ValueError("run_id must be nonempty trimmed text")
        self._space = space
        self._initial = initial
        self._settings = settings
        self._run_id = run_id
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
    def descriptor(self) -> OptimizerDescriptor:
        """Return the exact version-one local-search descriptor."""
        return self._DESCRIPTOR

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
            f"{self._run_id}.candidate.{sequence:08d}",
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
                values[index] = min(
                    bound.upper,
                    max(bound.lower, values[index] + direction * step),
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

    def checkpoint(self) -> OptimizerCheckpoint:
        """Return a canonical replay-complete pattern-search checkpoint."""
        payload: dict[str, JsonValue] = {
            "feedback": [encode_feedback(value) for value in self._feedback],
            "initial_values": list(self._initial),
            "kind": "bounded-coordinate-pattern",
            "outstanding": (
                None
                if self._outstanding is None
                else encode_candidate(self._outstanding)
            ),
            "run_id": self._run_id,
            "settings": {
                "initial_step_fraction": self._settings.initial_step_fraction,
                "max_evaluations": self._settings.max_evaluations,
                "max_iterations": self._settings.max_iterations,
                "step_reduction": self._settings.step_reduction,
                "step_tolerance": self._settings.step_tolerance,
            },
            "space": encode_space(self._space),
        }
        return OptimizerCheckpoint.create(self.descriptor.identity.value, payload)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: OptimizerCheckpoint
    ) -> BoundedCoordinatePatternOptimizer:
        """Restore state through deterministic replay and exact pending validation."""
        if type(checkpoint) is not OptimizerCheckpoint:
            raise TypeError("checkpoint must be OptimizerCheckpoint")
        if checkpoint.optimizer_identity != cls._DESCRIPTOR.identity.value:
            raise CheckpointError("checkpoint optimizer identity is incompatible")
        data = required(
            checkpoint.payload_object(),
            {
                "feedback",
                "initial_values",
                "kind",
                "outstanding",
                "run_id",
                "settings",
                "space",
            },
            "pattern checkpoint",
        )
        if string_value(data["kind"], "kind") != "bounded-coordinate-pattern":
            raise CheckpointError("checkpoint algorithm kind is incompatible")
        settings = required(
            object_value(data["settings"], "settings"),
            {
                "initial_step_fraction",
                "max_evaluations",
                "max_iterations",
                "step_reduction",
                "step_tolerance",
            },
            "settings",
        )
        try:
            engine = cls(
                decode_space(data["space"]),
                tuple(
                    number_value(value, "initial coordinate")
                    for value in list_value(data["initial_values"], "initial_values")
                ),
                PatternSearchSettings(
                    integer_value(settings["max_evaluations"], "max_evaluations"),
                    integer_value(settings["max_iterations"], "max_iterations"),
                    number_value(
                        settings["initial_step_fraction"], "initial_step_fraction"
                    ),
                    number_value(settings["step_reduction"], "step_reduction"),
                    number_value(settings["step_tolerance"], "step_tolerance"),
                ),
                run_id=string_value(data["run_id"], "run_id"),
            )
            for item in list_value(data["feedback"], "feedback"):
                expected = engine.ask()
                feedback = decode_feedback(item)
                if expected.candidate_id != feedback.candidate_id:
                    raise CheckpointError("checkpoint feedback order is incompatible")
                engine.tell(feedback)
            pending = data["outstanding"]
            if pending is not None:
                expected = engine.ask()
                if expected != decode_candidate(pending):
                    raise CheckpointError(
                        "checkpoint pending candidate is incompatible"
                    )
        except (
            TypeError,
            ValueError,
            OptimizerExhaustedError,
            UnknownCandidateError,
            DuplicateObservationError,
        ) as error:
            raise CheckpointError("pattern checkpoint replay failed") from error
        return engine
