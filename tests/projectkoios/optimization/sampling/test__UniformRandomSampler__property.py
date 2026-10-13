import random

import pytest

from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.property


def test_generated_samples_are_deterministic_finite_and_bounded() -> None:
    generator = random.Random(90210)
    for case in range(100):
        bounds = []
        for index in range(generator.randrange(0, 8)):
            lower = generator.uniform(-100.0, 10.0)
            width = 0.0 if case % 11 == 0 else generator.uniform(0.0, 100.0)
            bounds.append(ParameterBound(f"x{index}", lower, lower + width))
        sampler = UniformRandomSampler(ParameterSpace(tuple(bounds)))
        seed = generator.randrange(0, 2**64)

        first = sampler.sample(4, random_seed=seed)
        second = sampler.sample(4, random_seed=seed)

        assert first == second
        assert all(sampler.space.admits(values) for values in first)
