from dataclasses import dataclass, field

import pytest

from projectkoios.optimization.multiobjective.base import MultiObjectiveOptimizer
from projectkoios.optimization.multiobjective.pareto.algorithm import dominates, pareto
from projectkoios.optimization.multiobjective.pareto.optimizer import (
    ParetoOptimizer,
    ParetoSamplingStage,
)
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


@dataclass
class RecordingSampler:
    values: tuple[tuple[float, ...], ...]
    received_sets: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)
    received_fronts: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)
    received_decisions: list[tuple[tuple[float, ...], ...]] = field(
        default_factory=list
    )
    received_objectives: list[tuple[tuple[float, ...], ...]] = field(
        default_factory=list
    )

    def sample(
        self,
        sample_count: int,
        /,
        *,
        random_seed: int,
        pareto_set: tuple[tuple[float, ...], ...],
        pareto_front: tuple[tuple[float, ...], ...],
        evaluated_decisions: tuple[tuple[float, ...], ...] = (),
        evaluated_objectives: tuple[tuple[float, ...], ...] = (),
    ) -> tuple[tuple[float, ...], ...]:
        del random_seed
        assert sample_count == len(self.values)
        self.received_sets.append(pareto_set)
        self.received_fronts.append(pareto_front)
        self.received_decisions.append(evaluated_decisions)
        self.received_objectives.append(evaluated_objectives)
        return self.values


def test_pareto_selection_preserves_historical_minimization_semantics() -> None:
    points = ((3.0, 4.0), (2.0, 2.0), (1.0, 3.0))

    assert dominates(points[1], points[0]) is True
    assert pareto(points, chunk_sz=2) == [1, 2]


def test_optimizer_pulls_nondominated_objectives_back_to_decision_vectors() -> None:
    initial = RecordingSampler(((0.2,), (0.8,)))
    adaptive = RecordingSampler(((0.5,),))
    optimizer = ParetoOptimizer(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        (
            ParetoSamplingStage(initial, 2),
            ParetoSamplingStage(adaptive, 1),
        ),
        seed=9,
        run_id="pareto",
        chunk_size=2,
    )

    first = optimizer.ask()
    second = optimizer.ask()
    optimizer.tell(first, (1.0, 3.0))
    optimizer.tell(second, (3.0, 1.0))
    third = optimizer.ask()

    assert isinstance(optimizer, MultiObjectiveOptimizer)
    assert initial.received_sets == [()]
    assert adaptive.received_sets == [((0.2,), (0.8,))]
    assert adaptive.received_fronts == [((1.0, 3.0), (3.0, 1.0))]
    assert adaptive.received_decisions == [((0.2,), (0.8,))]
    assert adaptive.received_objectives == [((1.0, 3.0), (3.0, 1.0))]

    optimizer.tell(third, (0.5, 0.5))

    assert type(optimizer) is ParetoOptimizer
    assert optimizer.exhausted is True
    assert optimizer.pareto_set == ((0.5,),)
    assert optimizer.pareto_front == ((0.5, 0.5),)


def test_adaptive_sampler_receives_dominated_observation_history() -> None:
    initial = RecordingSampler(((0.1,), (0.5,), (0.9,)))
    adaptive = RecordingSampler(((0.2,),))
    optimizer = ParetoOptimizer(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        (
            ParetoSamplingStage(initial, 3),
            ParetoSamplingStage(adaptive, 1),
        ),
        seed=11,
        run_id="shell-history",
    )

    candidates = tuple(optimizer.ask() for _ in range(3))
    optimizer.tell(candidates[0], (1.0, 3.0))
    optimizer.tell(candidates[1], (3.0, 3.0))
    optimizer.tell(candidates[2], (3.0, 1.0))
    optimizer.ask()

    assert adaptive.received_sets == [((0.1,), (0.9,))]
    assert adaptive.received_decisions == [((0.1,), (0.5,), (0.9,))]
    assert adaptive.received_objectives == [((1.0, 3.0), (3.0, 3.0), (3.0, 1.0))]
