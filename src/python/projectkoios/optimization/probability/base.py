"""Dependency-free scalar probability-distribution contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import SupportsFloat

from projectkoios.numerics import validation as numerical_validation


class ProbabilityError(ValueError):
    """A probability distribution or transformation is invalid."""


def probability_value(value: SupportsFloat, name: str = "probability") -> float:
    """Return one finite built-in probability in the closed unit interval."""
    try:
        result = numerical_validation.finite_float(value, name)
    except ValueError as error:
        raise ProbabilityError(f"{name} must be finite") from error
    if not 0.0 <= result <= 1.0:
        raise ProbabilityError(f"{name} must lie in [0, 1]")
    return result


class ProbabilityDistributionFunction(ABC):
    """Evaluate a scalar cumulative distribution function (CDF)."""

    __slots__ = ()

    def evaluate(self, value: float, /) -> float:
        """Return the cumulative probability at one finite coordinate."""
        coordinate = numerical_validation.finite_float(value, "distribution coordinate")
        return probability_value(
            self._evaluate(coordinate),
            "probability distribution function result",
        )

    def __call__(self, value: float, /) -> float:
        """Return the cumulative probability at one finite coordinate."""
        return self.evaluate(value)

    @abstractmethod
    def _evaluate(self, value: float, /) -> float:
        """Evaluate one already-normalized finite coordinate."""
        raise NotImplementedError


class InverseProbabilityDistributionFunction(ABC):
    """Evaluate a scalar inverse CDF, also called a quantile function or PPF."""

    __slots__ = ()

    def evaluate(self, probability: float, /) -> float:
        """Return one finite quantile for a probability in ``[0, 1]``."""
        normalized = probability_value(probability)
        try:
            return numerical_validation.finite_float(
                self._evaluate(normalized),
                "inverse probability distribution function result",
            )
        except ValueError as error:
            raise ProbabilityError(
                "inverse probability distribution function result must be finite"
            ) from error

    def __call__(self, probability: float, /) -> float:
        """Return one finite quantile for a probability in ``[0, 1]``."""
        return self.evaluate(probability)

    @abstractmethod
    def _evaluate(self, probability: float, /) -> float:
        """Evaluate one already-normalized probability."""
        raise NotImplementedError


class ProbabilityDistribution(ABC):
    """Compose a CDF and inverse CDF behind scalar validated behavior."""

    __slots__ = ()

    @property
    @abstractmethod
    def distribution_function(self) -> ProbabilityDistributionFunction:
        """Return this distribution's cumulative distribution function."""
        raise NotImplementedError

    @property
    @abstractmethod
    def inverse_distribution_function(
        self,
    ) -> InverseProbabilityDistributionFunction:
        """Return this distribution's inverse cumulative distribution function."""
        raise NotImplementedError

    def cumulative_probability(self, value: float, /) -> float:
        """Return the cumulative probability at one finite coordinate."""
        return self.distribution_function.evaluate(value)

    def quantile(self, probability: float, /) -> float:
        """Return one finite quantile for a probability in ``[0, 1]``."""
        return self.inverse_distribution_function.evaluate(probability)


@dataclass(frozen=True, slots=True)
class FunctionalProbabilityDistribution(ProbabilityDistribution):
    """Immutable distribution composed from one compatible CDF and inverse CDF."""

    cumulative_function: ProbabilityDistributionFunction
    quantile_function: InverseProbabilityDistributionFunction

    def __post_init__(self) -> None:
        if not isinstance(
            self.cumulative_function,
            ProbabilityDistributionFunction,
        ):
            raise TypeError(
                "cumulative_function must be ProbabilityDistributionFunction"
            )
        if not isinstance(
            self.quantile_function,
            InverseProbabilityDistributionFunction,
        ):
            raise TypeError(
                "quantile_function must be InverseProbabilityDistributionFunction"
            )

    @property
    def distribution_function(self) -> ProbabilityDistributionFunction:
        """Return the configured cumulative distribution function."""
        return self.cumulative_function

    @property
    def inverse_distribution_function(
        self,
    ) -> InverseProbabilityDistributionFunction:
        """Return the configured inverse cumulative distribution function."""
        return self.quantile_function


__all__ = [
    "FunctionalProbabilityDistribution",
    "InverseProbabilityDistributionFunction",
    "ProbabilityDistribution",
    "ProbabilityDistributionFunction",
    "ProbabilityError",
    "probability_value",
]
