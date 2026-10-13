"""Weighted loss behavior for the forward-evaluated optimization engine block.

This module belongs to the provenance-bound engine block originating in Eugene J.
Ragasa's PhD work and preserved across PyPosPack and multiple research repositories.
Do not move, split, or refactor this block without explicit human authorization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..validation.validation import finite_float, measurement_name
from .records import Measurement, ObjectiveValue, RawEvaluation


@dataclass(frozen=True, slots=True)
class WeightedLossTerm:
    """One positive-weight squared residual against a finite target."""

    measurement: str
    target: float
    weight: float

    def __post_init__(self) -> None:
        measurement_name(self.measurement, "measurement")
        for name, value in (("target", self.target), ("weight", self.weight)):
            object.__setattr__(self, name, finite_float(value, name))
        if self.weight <= 0.0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True, slots=True)
class WeightedLoss:
    """Finite scalar loss and ordered absolute-residual objectives."""

    value: float
    objectives: tuple[ObjectiveValue, ...]

    def __post_init__(self) -> None:
        value = finite_float(self.value, "value")
        if value < 0.0:
            raise ValueError("value must be nonnegative")
        object.__setattr__(self, "value", value)
        if type(self.objectives) is not tuple or any(
            type(item) is not ObjectiveValue for item in self.objectives
        ):
            raise TypeError("objectives must be a tuple of ObjectiveValue")
        if not self.objectives:
            raise ValueError("objectives must be nonempty")


class WeightedSquaredLoss:
    """Evaluate an ordered, positive-weight squared residual sum."""

    __slots__ = ("_terms",)

    def __init__(self, terms: tuple[WeightedLossTerm, ...]) -> None:
        if type(terms) is not tuple or any(
            type(value) is not WeightedLossTerm for value in terms
        ):
            raise TypeError("terms must be a tuple of WeightedLossTerm")
        if not terms:
            raise ValueError("terms must be nonempty")
        names = tuple(value.measurement for value in terms)
        if len(names) != len(set(names)):
            raise ValueError("loss-term measurement names must be unique")
        self._terms = terms

    @property
    def terms(self) -> tuple[WeightedLossTerm, ...]:
        """Return the immutable ordered loss definition."""
        return self._terms

    def evaluate(self, evaluation: RawEvaluation) -> WeightedLoss:
        """Evaluate raw observations, failing on missing values or overflow."""
        if type(evaluation) is not RawEvaluation:
            raise TypeError("evaluation must be RawEvaluation")
        values = {item.name: item.value for item in evaluation.measurements}
        loss = 0.0
        objectives: list[ObjectiveValue] = []
        for term in self._terms:
            if term.measurement not in values:
                raise ValueError(f"missing measurement {term.measurement!r}")
            residual = abs(values[term.measurement] - term.target)
            if not math.isfinite(residual):
                raise ValueError("residual is nonfinite")
            objectives.append(ObjectiveValue(term.measurement, residual))
            loss += term.weight * residual * residual
            if not math.isfinite(loss):
                raise ValueError("weighted squared loss is nonfinite")
        return WeightedLoss(loss, tuple(objectives))


def raw_evaluation(candidate_id: str, **measurements: float) -> RawEvaluation:
    """Construct one raw evaluation while preserving caller keyword order."""
    return RawEvaluation(
        candidate_id,
        tuple(Measurement(name, value) for name, value in measurements.items()),
    )


__all__ = [
    "WeightedLoss",
    "WeightedLossTerm",
    "WeightedSquaredLoss",
    "raw_evaluation",
]
