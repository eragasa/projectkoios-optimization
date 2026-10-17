"""Slot-composed probability sampling and SplitMix64 unit sampling."""

from __future__ import annotations

from dataclasses import dataclass

from projectkoios.numerics import validation as numerical_validation

from ..sampling.base import RandomSeed, Sampler, SamplingError
from .base import InverseProbabilityDistributionFunction, ProbabilityError


class UnitUniformSampler(Sampler):
    """Sample finite fractions in the closed-open unit interval."""

    __slots__ = ()

    def _normalize_sample(self, sample: float, /) -> float:
        try:
            fraction = numerical_validation.finite_float(sample, "unit-uniform sample")
        except (TypeError, ValueError) as error:
            raise SamplingError("unit-uniform samples must be finite floats") from error
        if not 0.0 <= fraction < 1.0:
            raise SamplingError("unit-uniform samples must lie in [0, 1)")
        return fraction


@dataclass(frozen=True, slots=True)
class ProbabilityDistributionSampler(Sampler):
    """Sampling pipeline with unit-source and inverse-distribution slots."""

    source_slot: UnitUniformSampler
    transform_slot: InverseProbabilityDistributionFunction

    def __post_init__(self) -> None:
        if not isinstance(self.source_slot, UnitUniformSampler):
            raise TypeError("source_slot must be UnitUniformSampler")
        if not isinstance(
            self.transform_slot,
            InverseProbabilityDistributionFunction,
        ):
            raise TypeError(
                "transform_slot must be InverseProbabilityDistributionFunction"
            )

    def _sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: RandomSeed,
    ) -> tuple[float, ...]:
        probabilities = self.source_slot.sample(
            sample_count,
            random_seed=random_seed,
        )
        return tuple(self.transform_slot(value) for value in probabilities)

    def _normalize_sample(self, sample: float, /) -> float:
        try:
            return numerical_validation.finite_float(
                sample,
                "probability-distribution sample",
            )
        except (TypeError, ValueError) as error:
            raise ProbabilityError(
                "probability-distribution samples must be finite floats"
            ) from error


class SplitMix64UnitUniformSampler(UnitUniformSampler):
    """Sample deterministic upper-53-bit fractions from SplitMix64 words."""

    __slots__ = ()

    @staticmethod
    def unit_fraction(seed: int, counter: int) -> float:
        """Return one upper-53-bit SplitMix64 fraction in ``[0, 1)``."""
        normalized_seed = numerical_validation.u64(seed, "seed")
        normalized_counter = numerical_validation.u64(counter, "counter")
        word = (
            normalized_seed + 0x9E3779B97F4A7C15 * (normalized_counter + 1)
        ) & numerical_validation.MAX_U64
        word = (
            (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9
        ) & numerical_validation.MAX_U64
        word = (
            (word ^ (word >> 27)) * 0x94D049BB133111EB
        ) & numerical_validation.MAX_U64
        word ^= word >> 31
        return float(word >> 11) / 9_007_199_254_740_992.0

    def _sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: RandomSeed,
    ) -> tuple[float, ...]:
        if random_seed is None:
            raise TypeError("random_seed must be an exact built-in integer")
        seed = numerical_validation.u64(random_seed, "random_seed")
        return tuple(self.unit_fraction(seed, index) for index in range(sample_count))


__all__ = [
    "ProbabilityDistributionSampler",
    "SplitMix64UnitUniformSampler",
    "UnitUniformSampler",
]
