"""Dependency-free sampler contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from .._validation import u32


class SamplingError(ValueError):
    """A sampling request or produced sample is invalid."""


class Sampler[SampleT, SeedT](ABC):
    """Produce an exact ordered batch of samples for an explicit seed.

    The public operation owns structural validation. Concrete samplers implement
    only batch generation and, when needed, per-value normalization.
    """

    __slots__ = ()

    @final
    def sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: SeedT,
    ) -> tuple[SampleT, ...]:
        """Return exactly ``sample_count`` ordered values."""
        count = u32(sample_count, "sample_count", allow_zero=False)
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
        random_seed: SeedT,
    ) -> tuple[SampleT, ...]:
        """Produce one unvalidated batch for a normalized positive count."""
        raise NotImplementedError

    def _normalize_sample(self, sample: SampleT, /) -> SampleT:
        """Normalize or validate one produced value before publication."""
        return sample


__all__ = ["Sampler", "SamplingError"]
