"""Immutable records exchanged by the forward-evaluated engine block.

This module belongs to the provenance-bound engine block originating in Eugene J.
Ragasa's PhD work and preserved across PyPosPack and multiple research repositories.
Do not move, split, or refactor this block without explicit human authorization.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..validation import validation


@dataclass(frozen=True, slots=True)
class Candidate:
    """One immutable proposal from one exact optimizer run."""

    candidate_id: str
    optimizer_run_id: str
    sequence: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        validation.text(self.candidate_id, "candidate_id")
        validation.text(self.optimizer_run_id, "optimizer_run_id")
        sequence = validation.integer(self.sequence, "sequence")
        if sequence < 0:
            raise ValueError("sequence must be nonnegative")
        if type(self.values) is not tuple:
            raise TypeError("values must be a tuple")
        object.__setattr__(
            self,
            "values",
            tuple(
                validation.finite_float(value, "candidate coordinate")
                for value in self.values
            ),
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """One named finite raw forward-evaluation value."""

    name: str
    value: float

    def __post_init__(self) -> None:
        validation.measurement_name(self.name)
        object.__setattr__(
            self,
            "value",
            validation.finite_float(self.value, "measurement value"),
        )


@dataclass(frozen=True, slots=True)
class RawEvaluation:
    """Raw observations returned by a forward evaluator for one candidate."""

    candidate_id: str
    measurements: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        validation.text(self.candidate_id, "candidate_id")
        if type(self.measurements) is not tuple or any(
            type(value) is not Measurement for value in self.measurements
        ):
            raise TypeError("measurements must be a tuple of Measurement")
        names = tuple(value.name for value in self.measurements)
        if len(names) != len(set(names)):
            raise ValueError("measurement names must be unique")


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """One named finite nonnegative residual objective."""

    name: str
    value: float

    def __post_init__(self) -> None:
        validation.text(self.name, "objective name")
        value = validation.finite_float(self.value, "objective value")
        if value < 0.0:
            raise ValueError("objective value must be nonnegative")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class ObjectiveFeedback:
    """Scalar loss and optional residual objectives for one candidate."""

    candidate_id: str
    loss: float
    objectives: tuple[ObjectiveValue, ...] = ()

    def __post_init__(self) -> None:
        validation.text(self.candidate_id, "candidate_id")
        loss = validation.finite_float(self.loss, "loss")
        if loss < 0.0:
            raise ValueError("loss must be nonnegative")
        object.__setattr__(self, "loss", loss)
        if type(self.objectives) is not tuple or any(
            type(value) is not ObjectiveValue for value in self.objectives
        ):
            raise TypeError("objectives must be a tuple of ObjectiveValue")
        names = tuple(value.name for value in self.objectives)
        if len(names) != len(set(names)):
            raise ValueError("objective names must be unique")


@dataclass(frozen=True, slots=True)
class OptimizerUpdate:
    """Observable state summary returned after one accepted tell."""

    candidate_id: str
    evaluations: int
    best_candidate_id: str
    exhausted: bool
    termination: str | None

    def __post_init__(self) -> None:
        validation.text(self.candidate_id, "candidate_id")
        evaluations = validation.integer(self.evaluations, "evaluations")
        if evaluations < 1:
            raise ValueError("evaluations must be positive")
        validation.text(self.best_candidate_id, "best_candidate_id")
        if type(self.exhausted) is not bool:
            raise TypeError("exhausted must be a built-in bool")
        if self.termination is not None:
            validation.text(self.termination, "termination")
        if self.exhausted != (self.termination is not None):
            raise ValueError("exhausted and termination must agree")


__all__ = [
    "Candidate",
    "Measurement",
    "ObjectiveFeedback",
    "ObjectiveValue",
    "OptimizerUpdate",
    "RawEvaluation",
]
