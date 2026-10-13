import math

import pytest

pytestmark = pytest.mark.unit

from projectkoios.optimization.probability.base import (  # noqa: E402
    ProbabilityDistribution,
    ProbabilityDistributionFunction,
    ProbabilityError,
)
from projectkoios.optimization.probability.distributions import (  # noqa: E402
    UniformInverseProbabilityDistributionFunction,
    UniformProbabilityDistribution,
    UniformProbabilityDistributionFunction,
    UniformProbabilityDistributionParameters,
)
from projectkoios.optimization.probability.sampling import (  # noqa: E402
    ProbabilityDistributionSampler,
    SplitMix64UnitUniformSampler,
)
from projectkoios.optimization.sampling.base import (  # noqa: E402
    Sampler,
    SamplingError,
)


def parameters() -> UniformProbabilityDistributionParameters:
    return UniformProbabilityDistributionParameters(1.5, 2.5)


def test_uniform_distribution_composes_cdf_and_inverse_cdf() -> None:
    distribution = UniformProbabilityDistribution(parameters())

    assert isinstance(distribution, ProbabilityDistribution)
    assert isinstance(
        distribution.distribution_function, ProbabilityDistributionFunction
    )
    assert isinstance(
        distribution.distribution_function,
        UniformProbabilityDistributionFunction,
    )
    assert isinstance(
        distribution.inverse_distribution_function,
        UniformInverseProbabilityDistributionFunction,
    )
    assert distribution.cumulative_probability(1.0) == 0.0
    assert distribution.cumulative_probability(2.0) == 0.5
    assert distribution.cumulative_probability(3.0) == 1.0
    assert distribution.quantile(0.0) == 1.5
    assert distribution.quantile(0.5) == 2.0
    assert distribution.quantile(1.0) == 2.5


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (math.nan, 1.0),
        (0.0, math.inf),
        (1.0, 1.0),
        (2.0, 1.0),
        (-1e308, 1e308),
    ],
)
def test_uniform_definition_rejects_invalid_intervals(
    lower: float, upper: float
) -> None:
    with pytest.raises(ProbabilityError):
        UniformProbabilityDistributionParameters(lower, upper)


def test_probability_and_inverse_results_are_validated() -> None:
    distribution = UniformProbabilityDistribution(parameters())
    with pytest.raises(ProbabilityError):
        distribution.quantile(-0.1)
    with pytest.raises(ProbabilityError):
        distribution.quantile(1.1)
    with pytest.raises(ValueError):
        distribution.cumulative_probability(math.inf)


def test_probability_sampler_is_a_slot_composed_sampler() -> None:
    distribution = UniformProbabilityDistribution(parameters())
    source = SplitMix64UnitUniformSampler()
    sampler = ProbabilityDistributionSampler(
        source_slot=source,
        transform_slot=distribution.inverse_distribution_function,
    )

    first = sampler.sample(4, random_seed=7)
    second = sampler.sample(4, random_seed=7)

    assert isinstance(sampler, Sampler)
    assert sampler.source_slot is source
    assert sampler.transform_slot is distribution.inverse_distribution_function
    assert first == second
    assert first == tuple(
        1.5 + SplitMix64UnitUniformSampler.unit_fraction(7, index) for index in range(4)
    )
    assert all(1.5 <= value < 2.5 for value in first)


@pytest.mark.parametrize("sample_count", [0, -1, True])
def test_sampler_rejects_invalid_counts(sample_count: int) -> None:
    sampler = SplitMix64UnitUniformSampler()
    with pytest.raises((TypeError, ValueError)):
        sampler.sample(sample_count, random_seed=7)


class WrongLengthSampler(Sampler[int, int]):
    def _sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: int,
    ) -> tuple[int, ...]:
        del sample_count, random_seed
        return ()


def test_sampler_contract_rejects_wrong_batch_length() -> None:
    with pytest.raises(SamplingError):
        WrongLengthSampler().sample(1, random_seed=7)
