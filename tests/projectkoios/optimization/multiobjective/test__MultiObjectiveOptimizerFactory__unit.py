import pytest

from projectkoios.optimization.multiobjective.factory import (
    MultiObjectiveOptimizerFactory,
)
from projectkoios.optimization.multiobjective.pareto.optimizer import (
    ParetoOptimizer,
    ParetoSamplingStage,
)
from projectkoios.optimization.probability.factory import ProbabilitySamplerFactory
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


def test_factory_constructs_concrete_pareto_optimizer() -> None:
    space = ParameterSpace((ParameterBound("x", 0.0, 1.0),))
    sampler = ProbabilitySamplerFactory().create_uniform_random(space)

    optimizer = MultiObjectiveOptimizerFactory().create_pareto(
        space,
        (ParetoSamplingStage(sampler, 2),),
        seed=4,
        run_id="pareto",
    )

    assert type(optimizer) is ParetoOptimizer
