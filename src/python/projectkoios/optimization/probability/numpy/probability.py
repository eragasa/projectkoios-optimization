"""Optional NumPy probability functions and sampler slots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import (
    InverseProbabilityDistributionFunction,
    ProbabilityDistributionFunction,
    ProbabilityError,
)
from ..distributions import UniformProbabilityDistributionParameters
from ..sampling import UnitUniformSampler

type NumpyProbabilityCallable = Callable[
    [npt.NDArray[np.float64]],
    npt.ArrayLike,
]


def _one_float(value: npt.ArrayLike, name: str) -> float:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProbabilityError(f"{name} must return one finite number") from error
    if result.size != 1:
        raise ProbabilityError(f"{name} must return exactly one value")
    return float(result.reshape(-1)[0])


def _random_state(random_seed: int | None) -> np.random.RandomState:
    if random_seed is not None and type(random_seed) is not int:
        raise TypeError("random_seed must be a built-in integer or None")
    try:
        return np.random.RandomState(random_seed)
    except ValueError as error:
        raise ProbabilityError("random_seed is outside NumPy's valid range") from error


@dataclass(frozen=True, slots=True)
class NumpyProbabilityDistributionFunction(ProbabilityDistributionFunction):
    """Adapt a NumPy array callable to the scalar CDF contract."""

    function: NumpyProbabilityCallable

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable")

    def _evaluate(self, value: float, /) -> float:
        argument = np.asarray((value,), dtype=np.float64)
        return _one_float(self.function(argument), "NumPy CDF")


@dataclass(frozen=True, slots=True)
class NumpyInverseProbabilityDistributionFunction(
    InverseProbabilityDistributionFunction
):
    """Adapt a NumPy array callable to the scalar inverse-CDF contract."""

    function: NumpyProbabilityCallable

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable")

    def _evaluate(self, probability: float, /) -> float:
        argument = np.asarray((probability,), dtype=np.float64)
        return _one_float(self.function(argument), "NumPy inverse CDF")


class NumpyRandomStateUnitUniformSampler(UnitUniformSampler[int | None]):
    """NumPy RandomState implementation of the sampler source slot."""

    __slots__ = ()

    def _sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: int | None,
    ) -> tuple[float, ...]:
        samples = _random_state(random_seed).uniform(
            low=0.0,
            high=1.0,
            size=sample_count,
        )
        return tuple(float(value) for value in samples)


class NumpyUniformCompatibilitySampler:
    """Preserve the frozen direct NumPy RandomState uniform sampling API."""

    __slots__ = ()

    def sample_uniform(
        self,
        lower_bound: float,
        upper_bound: float,
        sample_count: int,
        /,
        *,
        random_seed: int | None,
    ) -> tuple[float, ...]:
        """Return the frozen one-dimensional RandomState uniform sequence."""
        try:
            parameters = UniformProbabilityDistributionParameters(
                lower_bound,
                upper_bound,
            )
        except (TypeError, ValueError) as error:
            raise ProbabilityError("uniform bounds are invalid") from error
        if type(sample_count) is not int or sample_count <= 0:
            raise ProbabilityError("sample_count must be a positive integer")
        if random_seed is not None and type(random_seed) is not int:
            raise ProbabilityError("random_seed must be an integer or None")
        samples = _random_state(random_seed).uniform(
            low=parameters.lower_bound,
            high=parameters.upper_bound,
            size=sample_count,
        )
        return tuple(float(value) for value in samples)

    def sample_independent_uniform(
        self,
        bounds: tuple[tuple[float, float], ...],
        sample_count: int,
        /,
        *,
        random_seed: int | None,
    ) -> tuple[tuple[float, ...], ...]:
        """Return sample-major RandomState draws from independent dimensions."""
        if type(bounds) is not tuple or not bounds:
            raise ProbabilityError("bounds must be a nonempty tuple")
        parameters: list[UniformProbabilityDistributionParameters] = []
        for index, bound in enumerate(bounds):
            if type(bound) is not tuple or len(bound) != 2:
                raise ProbabilityError(
                    f"bounds[{index}] must contain lower and upper bounds"
                )
            try:
                parameters.append(UniformProbabilityDistributionParameters(*bound))
            except (TypeError, ValueError) as error:
                raise ProbabilityError(f"bounds[{index}] is invalid") from error
        if type(sample_count) is not int or sample_count <= 0:
            raise ProbabilityError("sample_count must be a positive integer")
        if random_seed is not None and type(random_seed) is not int:
            raise ProbabilityError("random_seed must be an integer or None")
        lower = np.asarray(
            tuple(value.lower_bound for value in parameters),
            dtype=np.float64,
        )
        upper = np.asarray(
            tuple(value.upper_bound for value in parameters),
            dtype=np.float64,
        )
        samples = _random_state(random_seed).uniform(
            low=lower,
            high=upper,
            size=(sample_count, len(parameters)),
        )
        return tuple(tuple(float(value) for value in row) for row in samples)


__all__ = [
    "NumpyInverseProbabilityDistributionFunction",
    "NumpyProbabilityDistributionFunction",
    "NumpyRandomStateUnitUniformSampler",
    "NumpyUniformCompatibilitySampler",
]
