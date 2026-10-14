"""Dependency-free deterministic selection from finite objective values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from ..validation.validation import finite_float, integer


@final
@dataclass(frozen=True, slots=True)
class FiniteMinimum:
    """First minimum in an ordered finite objective inventory."""

    index: int
    value: float

    def __post_init__(self) -> None:
        index = integer(self.index, "index")
        if index < 0:
            raise ValueError("index must be nonnegative")
        object.__setattr__(self, "value", finite_float(self.value, "value"))


@final
@dataclass(frozen=True, slots=True)
class FiniteAmbiguitySet:
    """First minimum and every index inside an absolute objective tolerance."""

    minimum: FiniteMinimum
    tolerance: float
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.minimum) is not FiniteMinimum:
            raise TypeError("minimum must be FiniteMinimum")
        tolerance = finite_float(self.tolerance, "tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if type(self.indices) is not tuple or not self.indices:
            raise TypeError("indices must be a nonempty tuple")
        indices = tuple(integer(value, "indices value") for value in self.indices)
        if any(value < 0 for value in indices):
            raise ValueError("indices must be nonnegative")
        if any(
            left >= right for left, right in zip(indices, indices[1:], strict=False)
        ):
            raise ValueError("indices must be strictly increasing")
        if self.minimum.index not in indices:
            raise ValueError("indices must contain the minimum index")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "indices", indices)


def first_minimum(values: tuple[float, ...], /) -> FiniteMinimum:
    """Return the first exact minimum from a nonempty ordered value tuple."""
    normalized = _finite_values(values)
    return _first_minimum(normalized)


def indices_at_or_below(
    values: tuple[float, ...],
    ceiling: float,
    /,
) -> tuple[int, ...]:
    """Return every index whose value is at or below a finite ceiling."""
    normalized = _finite_values(values)
    normalized_ceiling = finite_float(ceiling, "ceiling")
    return tuple(
        index for index, value in enumerate(normalized) if value <= normalized_ceiling
    )


def absolute_ambiguity_set(
    values: tuple[float, ...],
    tolerance: float,
    /,
) -> FiniteAmbiguitySet:
    """Return all indices within an absolute tolerance of the exact minimum."""
    normalized = _finite_values(values)
    normalized_tolerance = finite_float(tolerance, "tolerance")
    if normalized_tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    minimum = _first_minimum(normalized)
    indices = tuple(
        index
        for index, value in enumerate(normalized)
        if value - minimum.value <= normalized_tolerance
    )
    return FiniteAmbiguitySet(minimum, normalized_tolerance, indices)


def _finite_values(values: object) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise TypeError("values must be a tuple")
    if not values:
        raise ValueError("values must be nonempty")
    return tuple(
        finite_float(value, f"values[{index}]") for index, value in enumerate(values)
    )


def _first_minimum(values: tuple[float, ...]) -> FiniteMinimum:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value < best_value:
            best_index = index
            best_value = value
    return FiniteMinimum(best_index, best_value)


__all__ = [
    "FiniteAmbiguitySet",
    "FiniteMinimum",
    "absolute_ambiguity_set",
    "first_minimum",
    "indices_at_or_below",
]
