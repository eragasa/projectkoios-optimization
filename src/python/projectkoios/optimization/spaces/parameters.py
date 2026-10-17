"""Finite ordered parameter spaces for bounded numerical optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

from projectkoios.numerics import validation as numerical_validation

from ..validation import validation


@dataclass(frozen=True, slots=True)
class ParameterBound:
    """One finite closed bound; equal endpoints define a fixed coordinate."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        validation.text(self.name, "name")
        lower = numerical_validation.finite_float(self.lower, "lower")
        upper = numerical_validation.finite_float(self.upper, "upper")
        if lower > upper:
            raise ValueError("lower bound must not exceed upper bound")
        if not math.isfinite(upper - lower):
            raise ValueError("bound width must be representable as a finite float")
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
        normalized = tuple(
            numerical_validation.finite_float(value, "coordinate") for value in values
        )
        return all(
            bound.lower <= value <= bound.upper
            for bound, value in zip(self.bounds, normalized, strict=True)
        )


__all__ = ["ParameterBound", "ParameterSpace"]
