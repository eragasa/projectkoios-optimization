"""Dependency-free concrete probability distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from projectkoios.numerics import validation as numerical_validation

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
            numerical_validation.exact_finite_float(self.lower_bound, "lower_bound")
            numerical_validation.exact_finite_float(self.upper_bound, "upper_bound")
        except ValueError as error:
            raise ProbabilityError("uniform bounds must be finite floats") from error
        if self.lower_bound >= self.upper_bound:
            raise ProbabilityError("lower_bound must be less than upper_bound")
        if not math.isfinite(self.upper_bound - self.lower_bound):
            raise ProbabilityError(
                "uniform width must be representable as a finite float"
            )

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

    def __post_init__(self) -> None:
        if type(self.parameters) is not UniformProbabilityDistributionParameters:
            raise TypeError(
                "parameters must be UniformProbabilityDistributionParameters"
            )

    @property
    def distribution_function(self) -> ProbabilityDistributionFunction:
        """Return the uniform cumulative distribution function."""
        return UniformProbabilityDistributionFunction(self.parameters)

    @property
    def inverse_distribution_function(
        self,
    ) -> InverseProbabilityDistributionFunction:
        """Return the uniform inverse cumulative distribution function."""
        return UniformInverseProbabilityDistributionFunction(self.parameters)


__all__ = [
    "UniformInverseProbabilityDistributionFunction",
    "UniformProbabilityDistribution",
    "UniformProbabilityDistributionFunction",
    "UniformProbabilityDistributionParameters",
]
