"""Dependency-free immutable optimization records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast


def _text(name: str, value: object, *, maximum: int = 256) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in string")
    if not value or len(value) > maximum or value.strip() != value:
        raise ValueError(f"{name} must be nonempty trimmed text")
    return value


def _finite(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a built-in int or float")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ParameterBound:
    """One finite closed bound; equal endpoints define a fixed coordinate."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _text("name", self.name)
        lower = _finite("lower", self.lower)
        upper = _finite("upper", self.upper)
        if lower > upper:
            raise ValueError("lower bound must not exceed upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def width(self) -> float:
        """Return the finite bound width."""
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Ordered uniquely named finite bounds, including an empty scalar space."""

    bounds: tuple[ParameterBound, ...]

    def __post_init__(self) -> None:
        if type(self.bounds) is not tuple or any(
            type(value) is not ParameterBound for value in self.bounds
        ):
            raise TypeError("bounds must be a tuple of ParameterBound")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("bound names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return canonical coordinate names."""
        return tuple(bound.name for bound in self.bounds)

    @property
    def dimension(self) -> int:
        """Return the number of coordinates."""
        return len(self.bounds)

    def admits(self, values: tuple[float, ...]) -> bool:
        """Return whether values have the exact dimension and lie in all bounds."""
        if type(values) is not tuple:
            raise TypeError("values must be a tuple")
        if len(values) != self.dimension:
            return False
        normalized = tuple(_finite("coordinate", value) for value in values)
        return all(
            bound.lower <= value <= bound.upper
            for bound, value in zip(self.bounds, normalized, strict=True)
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One immutable proposal from one exact optimizer run."""

    candidate_id: str
    optimizer_run_id: str
    sequence: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        _text("optimizer_run_id", self.optimizer_run_id)
        if type(self.sequence) is not int:
            raise TypeError("sequence must be a built-in integer excluding bool")
        if self.sequence < 0:
            raise ValueError("sequence must be nonnegative")
        if type(self.values) is not tuple:
            raise TypeError("values must be a tuple")
        object.__setattr__(
            self,
            "values",
            tuple(_finite("candidate coordinate", value) for value in self.values),
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """One named finite raw forward-evaluation value."""

    name: str
    value: float

    def __post_init__(self) -> None:
        _text("measurement name", self.name)
        object.__setattr__(self, "value", _finite("measurement value", self.value))


@dataclass(frozen=True, slots=True)
class RawEvaluation:
    """Raw observations returned by a forward evaluator for one candidate."""

    candidate_id: str
    measurements: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        if type(self.measurements) is not tuple or any(
            type(value) is not Measurement for value in self.measurements
        ):
            raise TypeError("measurements must be a tuple of Measurement")
        names = tuple(value.name for value in self.measurements)
        if len(names) != len(set(names)):
            raise ValueError("measurement names must be unique")


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """One ordered finite nonnegative objective value."""

    name: str
    value: float

    def __post_init__(self) -> None:
        _text("objective name", self.name)
        value = _finite("objective value", self.value)
        if value < 0.0:
            raise ValueError("objective value must be nonnegative")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class ObjectiveFeedback:
    """Scalar loss and optional ordered objectives for one proposed candidate."""

    candidate_id: str
    loss: float
    objectives: tuple[ObjectiveValue, ...] = ()

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        loss = _finite("loss", self.loss)
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
        _text("candidate_id", self.candidate_id)
        if type(self.evaluations) is not int or self.evaluations < 1:
            raise ValueError("evaluations must be a positive built-in integer")
        _text("best_candidate_id", self.best_candidate_id)
        if type(self.exhausted) is not bool:
            raise TypeError("exhausted must be a built-in bool")
        if self.termination is not None:
            _text("termination", self.termination)
        if self.exhausted != (self.termination is not None):
            raise ValueError("exhausted and termination must agree")
