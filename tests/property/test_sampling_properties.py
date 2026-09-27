import random

import pytest

from projectkoios.optimization import (
    BoundedUniformOptimizer,
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.property


def test_generated_samples_are_deterministic_finite_and_bounded() -> None:
    generator = random.Random(90210)
    for case in range(100):
        dimension = generator.randrange(0, 8)
        bounds = []
        for index in range(dimension):
            lower = generator.uniform(-100.0, 10.0)
            width = 0.0 if case % 11 == 0 else generator.uniform(0.0, 100.0)
            bounds.append(ParameterBound(f"x{index}", lower, lower + width))
        space = ParameterSpace(tuple(bounds))
        seed = generator.randrange(0, 2**64)
        first = BoundedUniformOptimizer.sample_values(space, seed, case)
        second = BoundedUniformOptimizer.sample_values(space, seed, case)
        assert first == second
        assert space.admits(first)
