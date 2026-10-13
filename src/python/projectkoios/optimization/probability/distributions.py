"""Dependency-free concrete probability distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..validation import validation
from .base import (
    InverseProbabilityDistributionFunction,
    ProbabilityDistribution,
    ProbabilityDistributionFunction,
    ProbabilityError,
)


@dataclass(frozen=True, slots=True)
class UniformProbabilityDistributionParameters:
    """Finite endpoints of a nondegenerate closed-open uniform distribution."""

    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        try:
            lower = validation.finite_float(self.lower_bound, "lower_bound")
            upper = validation.finite_float(self.upper_bound, "upper_bound")
        except ValueError as error:
            raise ProbabilityError("uniform bounds must be finite floats") from error
        if lower >= upper:
            raise ProbabilityError("lower_bound must be less than upper_bound")
        if not math.isfinite(upper - lower):
            raise ProbabilityError(
                "uniform width must be representable as a finite float"
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    @property
    def width(self) -> float:
        """Return the positive finite interval width."""
        return self.upper_bound - self.lower_bound


@dataclass(frozen=True, slots=True)
class UniformProbabilityDistributionFunction(ProbabilityDistributionFunction):
    """Evaluate the CDF of one finite continuous uniform distribution."""

    parameters: UniformProbabilityDistributionParameters

    def __post_init__(self) -> None:
        if type(self.parameters) is not UniformProbabilityDistributionParameters:
            raise TypeError(
                "parameters must be UniformProbabilityDistributionParameters"
            )

    def _evaluate(self, value: float, /) -> float:
        if value <= self.parameters.lower_bound:
            return 0.0
        if value >= self.parameters.upper_bound:
            return 1.0
        return (value - self.parameters.lower_bound) / self.parameters.width


@dataclass(frozen=True, slots=True)
class UniformInverseProbabilityDistributionFunction(
    InverseProbabilityDistributionFunction
):
    """Evaluate the inverse CDF of one finite continuous uniform distribution."""

    parameters: UniformProbabilityDistributionParameters

    def __post_init__(self) -> None:
        if type(self.parameters) is not UniformProbabilityDistributionParameters:
            raise TypeError(
                "parameters must be UniformProbabilityDistributionParameters"
            )

    def _evaluate(self, probability: float, /) -> float:
        if probability == 0.0:
            return self.parameters.lower_bound
        if probability == 1.0:
            return self.parameters.upper_bound
        return self.parameters.lower_bound + probability * self.parameters.width


@dataclass(frozen=True, slots=True)
class UniformProbabilityDistribution(ProbabilityDistribution):
    """Immutable dependency-free continuous uniform distribution."""

    parameters: UniformProbabilityDistributionParameters
    _cumulative_function: UniformProbabilityDistributionFunction = field(
        init=False,
        repr=False,
    )
    _quantile_function: UniformInverseProbabilityDistributionFunction = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.parameters) is not UniformProbabilityDistributionParameters:
            raise TypeError(
                "parameters must be UniformProbabilityDistributionParameters"
            )
        object.__setattr__(
            self,
            "_cumulative_function",
            UniformProbabilityDistributionFunction(self.parameters),
        )
        object.__setattr__(
            self,
            "_quantile_function",
            UniformInverseProbabilityDistributionFunction(self.parameters),
        )

    @property
    def distribution_function(self) -> ProbabilityDistributionFunction:
        """Return the uniform cumulative distribution function."""
        return self._cumulative_function

    @property
    def inverse_distribution_function(
        self,
    ) -> InverseProbabilityDistributionFunction:
        """Return the uniform inverse cumulative distribution function."""
        return self._quantile_function


__all__ = [
    "UniformInverseProbabilityDistributionFunction",
    "UniformProbabilityDistribution",
    "UniformProbabilityDistributionFunction",
    "UniformProbabilityDistributionParameters",
]
