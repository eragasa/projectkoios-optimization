from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization import ProbabilityDistributionSampler  # noqa: E402
from projectkoios.optimization.adapters.numpy.probability import (  # noqa: E402
    NumpyInverseProbabilityDistributionFunction,
    NumpyProbabilityDistributionFunction,
    NumpyRandomStateUnitUniformSampler,
    NumpyUniformCompatibilitySampler,
)

pytestmark = pytest.mark.integration


def test_numpy_function_adapters_support_scalar_probability_contracts() -> None:
    cumulative = NumpyProbabilityDistributionFunction(
        lambda values: np.clip(values - 1.5, 0.0, 1.0)
    )
    inverse = NumpyInverseProbabilityDistributionFunction(
        lambda probabilities: 1.5 + probabilities
    )

    assert cumulative(1.0) == 0.0
    assert cumulative(2.0) == 0.5
    assert cumulative(3.0) == 1.0
    assert inverse(0.25) == 1.75


def test_numpy_source_fills_probability_sampler_pipeline_slot() -> None:
    source = NumpyRandomStateUnitUniformSampler()
    transform = NumpyInverseProbabilityDistributionFunction(
        lambda probabilities: 1.5 + probabilities
    )
    sampler = ProbabilityDistributionSampler(
        source_slot=source,
        transform_slot=transform,
    )

    assert sampler.sample(4, random_seed=7) == (
        1.5763082893739573,
        2.2799187922401147,
        1.9384092314408936,
        2.223465177830941,
    )


def test_numpy_compatibility_sampler_preserves_frozen_uniform_sequences() -> None:
    sampler = NumpyUniformCompatibilitySampler()

    assert sampler.sample_uniform(1.5, 2.5, 4, random_seed=7) == (
        1.5763082893739573,
        2.2799187922401147,
        1.9384092314408936,
        2.223465177830941,
    )
    assert sampler.sample_independent_uniform(
        ((1.5, 2.5), (10.0, 20.0)),
        2,
        random_seed=7,
    ) == (
        (1.5763082893739573, 17.799187922401146),
        (1.9384092314408936, 17.234651778309413),
    )


def test_numpy_adapters_do_not_mutate_global_random_state() -> None:
    sampler = NumpyUniformCompatibilitySampler()
    original_state = np.random.get_state()
    try:
        np.random.seed(19)
        expected = (float(np.random.uniform()), float(np.random.uniform()))
        np.random.seed(19)
        first = float(np.random.uniform())

        sampler.sample_uniform(1.5, 2.5, 2, random_seed=7)

        second = float(np.random.uniform())
        assert (first, second) == expected
    finally:
        np.random.set_state(original_state)


def test_numpy_compatibility_sampler_rejects_invalid_requests() -> None:
    sampler = NumpyUniformCompatibilitySampler()
    with pytest.raises(ValueError):
        sampler.sample_uniform(1.5, 2.5, 1, random_seed=-1)
    for count in (0, -1, True):
        with pytest.raises(ValueError):
            sampler.sample_uniform(1.5, 2.5, count, random_seed=7)
    with pytest.raises(ValueError):
        sampler.sample_uniform(1.5, 2.5, 1, random_seed=True)
