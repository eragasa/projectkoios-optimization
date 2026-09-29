from typing import Any

import pytest

stats: Any = pytest.importorskip("scipy.stats")

from projectkoios.optimization import (  # noqa: E402
    ProbabilityDistributionSampler,
    SplitMix64UnitUniformSampler,
)
from projectkoios.optimization.adapters.scipy.probability import (  # noqa: E402
    ScipyInverseProbabilityDistributionFunction,
    ScipyProbabilityDistributionFunction,
)

pytestmark = pytest.mark.integration


def test_scipy_function_adapters_wrap_frozen_distribution() -> None:
    source = stats.uniform(loc=1.5, scale=1.0)
    cumulative = ScipyProbabilityDistributionFunction(source)
    inverse = ScipyInverseProbabilityDistributionFunction(source)

    assert cumulative(1.5) == 0.0
    assert cumulative(2.0) == 0.5
    assert cumulative(2.5) == 1.0
    assert inverse(0.25) == 1.75


def test_scipy_transform_fills_probability_sampler_pipeline_slot() -> None:
    transform = ScipyInverseProbabilityDistributionFunction(
        stats.uniform(loc=-2.0, scale=5.0)
    )
    sampler = ProbabilityDistributionSampler(
        source_slot=SplitMix64UnitUniformSampler(),
        transform_slot=transform,
    )

    samples = sampler.sample(8, random_seed=1729)

    assert len(samples) == 8
    assert all(-2.0 <= value < 3.0 for value in samples)
    assert samples == sampler.sample(8, random_seed=1729)


def test_scipy_nonfinite_endpoint_quantiles_fail_closed() -> None:
    inverse = ScipyInverseProbabilityDistributionFunction(stats.norm())

    with pytest.raises(ValueError):
        inverse(0.0)
    with pytest.raises(ValueError):
        inverse(1.0)
