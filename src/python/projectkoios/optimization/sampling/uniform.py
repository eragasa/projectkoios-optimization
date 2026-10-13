"""Deterministic uniform random sampling over a finite parameter space."""

from __future__ import annotations

from ..probability.sampling import SplitMix64UnitUniformSampler
from ..spaces.parameters import ParameterSpace
from ..validation.validation import integer, u32, u64


class UniformRandomSampler:
    """Generate deterministic uniform decision vectors inside a parameter space."""

    __slots__ = ("_space",)

    def __init__(self, space: ParameterSpace) -> None:
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        self._space = space

    @property
    def space(self) -> ParameterSpace:
        """Return the sampled parameter space."""
        return self._space

    def sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: int,
        pareto_set: tuple[tuple[float, ...], ...] = (),
        pareto_front: tuple[tuple[float, ...], ...] = (),
        evaluated_decisions: tuple[tuple[float, ...], ...] = (),
        evaluated_objectives: tuple[tuple[float, ...], ...] = (),
    ) -> tuple[tuple[float, ...], ...]:
        """Return an exact batch; observation inputs do not bias uniformity."""
        del (
            pareto_set,
            pareto_front,
            evaluated_decisions,
            evaluated_objectives,
        )
        count = u32(sample_count, "sample_count", allow_zero=False)
        seed = u64(random_seed, "random_seed")
        return tuple(
            self.sample_values(self._space, seed, index) for index in range(count)
        )

    @staticmethod
    def unit_fraction(seed: int, counter: int) -> float:
        """Return one upper-53-bit SplitMix64 fraction in ``[0, 1)``."""
        return SplitMix64UnitUniformSampler.unit_fraction(seed, counter)

    @classmethod
    def sample_values(
        cls,
        space: ParameterSpace,
        seed: int,
        sample_index: int,
    ) -> tuple[float, ...]:
        """Map one sample-major counter block into the configured closed bounds."""
        if type(space) is not ParameterSpace:
            raise TypeError("space must be ParameterSpace")
        normalized_seed = u64(seed, "seed")
        normalized_index = integer(sample_index, "sample_index")
        if normalized_index < 0:
            raise ValueError("sample_index must be nonnegative")
        dimension = space.dimension
        return tuple(
            bound.lower
            + cls.unit_fraction(
                normalized_seed,
                normalized_index * dimension + index,
            )
            * bound.width
            for index, bound in enumerate(space.bounds)
        )


__all__ = ["UniformRandomSampler"]
