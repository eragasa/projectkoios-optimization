import pytest

from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


def test_sampler_returns_exact_deterministic_batch() -> None:
    sampler = UniformRandomSampler(ParameterSpace((ParameterBound("x", -1.0, 1.0),)))

    first = sampler.sample(3, random_seed=17)
    second = sampler.sample(3, random_seed=17)

    assert first == second
    assert len(first) == 3
    assert all(sampler.space.admits(values) for values in first)


def test_empty_space_and_fixed_coordinates_are_supported() -> None:
    empty = UniformRandomSampler(ParameterSpace(()))
    fixed = UniformRandomSampler(ParameterSpace((ParameterBound("x", 4.0, 4.0),)))

    assert empty.sample(1, random_seed=0) == ((),)
    assert fixed.sample(1, random_seed=99) == ((4.0,),)


def test_pareto_context_does_not_bias_uniform_sampling() -> None:
    sampler = UniformRandomSampler(ParameterSpace((ParameterBound("x", 0.0, 1.0),)))

    baseline = sampler.sample(2, random_seed=4)
    contextual = sampler.sample(
        2,
        random_seed=4,
        pareto_set=((0.25,),),
        pareto_front=((1.0, 2.0),),
    )

    assert contextual == baseline
