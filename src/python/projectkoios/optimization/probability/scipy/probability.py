"""Optional SciPy probability-function adapter slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from projectkoios.numerics import validation as numerical_validation

from ..base import (
    InverseProbabilityDistributionFunction,
    ProbabilityDistributionFunction,
)


class ScipyProbabilityDistributionSource(Protocol):
    """Narrow surface supplied by a frozen ``scipy.stats`` distribution."""

    def cdf(self, value: float, /) -> object:
        """Return a scalar cumulative probability."""
        ...

    def ppf(self, probability: float, /) -> object:
        """Return a scalar quantile."""
        ...


@dataclass(frozen=True, slots=True)
class ScipyProbabilityDistributionFunction(ProbabilityDistributionFunction):
    """Adapt a frozen SciPy distribution's ``cdf`` method."""

    source: ScipyProbabilityDistributionSource

    def __post_init__(self) -> None:
        if not callable(getattr(self.source, "cdf", None)):
            raise TypeError("source must provide a callable cdf")

    def _evaluate(self, value: float, /) -> float:
        return numerical_validation.finite_float(
            self.source.cdf(value), "SciPy CDF result"
        )


@dataclass(frozen=True, slots=True)
class ScipyInverseProbabilityDistributionFunction(
    InverseProbabilityDistributionFunction
):
    """Adapt a frozen SciPy distribution's ``ppf`` method."""

    source: ScipyProbabilityDistributionSource

    def __post_init__(self) -> None:
        if not callable(getattr(self.source, "ppf", None)):
            raise TypeError("source must provide a callable ppf")

    def _evaluate(self, probability: float, /) -> float:
        return numerical_validation.finite_float(
            self.source.ppf(probability),
            "SciPy inverse CDF result",
        )


__all__ = [
    "ScipyInverseProbabilityDistributionFunction",
    "ScipyProbabilityDistributionFunction",
    "ScipyProbabilityDistributionSource",
]
