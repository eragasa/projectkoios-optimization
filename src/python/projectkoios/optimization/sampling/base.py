"""Dependency-free scalar sampler contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from projectkoios.numerics import validation as numerical_validation

type RandomSeed = int | None


class SamplingError(ValueError):
    """A sampling request or produced sample is invalid."""


class Sampler(ABC):
    """Produce an exact ordered batch of finite scalar samples.

    The public operation owns structural validation. Concrete scalar samplers implement
    only batch generation and, when needed, per-value normalization.
    """

    __slots__ = ()

    @final
    def sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: RandomSeed,
    ) -> tuple[float, ...]:
        """Return exactly ``sample_count`` ordered scalar values."""
        count = numerical_validation.u32(sample_count, "sample_count", allow_zero=False)
        values = self._sample(count, random_seed=random_seed)
        if type(values) is not tuple or len(values) != count:
            raise SamplingError(
                "sampler must return a tuple containing exactly sample_count values"
            )
        return tuple(self._normalize_sample(value) for value in values)

    @abstractmethod
    def _sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: RandomSeed,
    ) -> tuple[float, ...]:
        """Produce one unvalidated scalar batch for a normalized positive count."""
        raise NotImplementedError

    def _normalize_sample(self, sample: float, /) -> float:
        """Normalize or validate one produced scalar before publication."""
        return sample


__all__ = ["RandomSeed", "Sampler", "SamplingError"]
