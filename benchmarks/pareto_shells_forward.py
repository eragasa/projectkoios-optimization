"""Forward-evaluation benchmarks for Pareto-shell contrastive sampling.

The optimizers receive only decision vectors and returned objective vectors. Analytic
problem definitions and known reference fronts are owned exclusively by this benchmark
harness and are never exposed to a sampler.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy

from projectkoios.optimization.multiobjective.pareto.algorithm import pareto
from projectkoios.optimization.multiobjective.pareto.optimizer import (
    ParameterSampler,
    ParetoOptimizer,
    ParetoSamplingStage,
)
from projectkoios.optimization.sampling.scipy.gmm_kde import (
    CovarianceStabilizationMethod,
    GmmClusterAllocation,
    GmmComponentCriterion,
    GmmFeatureSpace,
    ScipyGmmConditionedKdeSampler,
)
from projectkoios.optimization.sampling.scipy.kde import (
    KdeBandwidthMethod,
    ScipyGaussianKdeSampler,
)
from projectkoios.optimization.sampling.scipy.pareto_shells import (
    ScipyParetoShellContrastiveKdeSampler,
)
from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace


class ForwardProblem(Protocol):
    """A bounded benchmark evaluator hidden behind its forward operation."""

    @property
    def name(self) -> str: ...

    @property
    def decision_dimension(self) -> int: ...

    @property
    def objective_dimension(self) -> int: ...

    def evaluate(self, decision: tuple[float, ...], /) -> tuple[float, ...]: ...

    def is_valid(self, decision: tuple[float, ...], /) -> bool: ...

    def score(self, decision: tuple[float, ...], /) -> tuple[float, ...]: ...

    def decision_mode(self, decision: tuple[float, ...], /) -> int | None: ...

    def reference_front(self) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class ZdtProblem:
    """ZDT1 or ZDT3 with a configurable decision dimension."""

    variant: int
    decision_dimension: int
    objective_dimension: int = 2

    @property
    def name(self) -> str:
        return f"ZDT{self.variant}-{self.decision_dimension}D"

    def evaluate(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        first = decision[0]
        g = 1.0 + 9.0 * sum(decision[1:]) / (len(decision) - 1)
        ratio = first / g
        if self.variant == 1:
            second = g * (1.0 - math.sqrt(ratio))
        elif self.variant == 3:
            second = g * (
                1.0 - math.sqrt(ratio) - ratio * math.sin(10.0 * math.pi * first)
            )
        else:
            raise ValueError("supported ZDT variants are 1 and 3")
        return (first, second)

    def is_valid(self, decision: tuple[float, ...], /) -> bool:
        del decision
        return True

    def score(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        return self.evaluate(decision)

    def decision_mode(self, decision: tuple[float, ...], /) -> int | None:
        del decision
        return None

    def reference_front(self) -> tuple[tuple[float, ...], ...]:
        if self.variant == 1:
            first_values = (index / 1000.0 for index in range(1001))
        else:
            intervals = (
                (0.0, 0.0830015349),
                (0.1822287280, 0.2577623634),
                (0.4093136748, 0.4538821041),
                (0.6183967944, 0.6525117038),
                (0.8233317983, 0.8518328654),
            )
            first_values = (
                lower + (upper - lower) * index / 200.0
                for lower, upper in intervals
                for index in range(201)
            )
        return tuple(
            (
                first,
                1.0
                - math.sqrt(first)
                - (
                    first * math.sin(10.0 * math.pi * first)
                    if self.variant == 3
                    else 0.0
                ),
            )
            for first in first_values
        )


@dataclass(frozen=True, slots=True)
class Dtlz2Problem:
    """Three-objective DTLZ2 with configurable decision dimension."""

    decision_dimension: int
    objective_dimension: int = 3

    @property
    def name(self) -> str:
        return f"DTLZ2-{self.decision_dimension}D-{self.objective_dimension}O"

    def evaluate(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        objective_count = self.objective_dimension
        g = sum((value - 0.5) ** 2 for value in decision[objective_count - 1 :])
        result: list[float] = []
        for objective_index in range(objective_count):
            value = 1.0 + g
            for decision_index in range(objective_count - objective_index - 1):
                value *= math.cos(decision[decision_index] * math.pi / 2.0)
            if objective_index:
                decision_index = objective_count - objective_index - 1
                value *= math.sin(decision[decision_index] * math.pi / 2.0)
            result.append(value)
        return tuple(result)

    def is_valid(self, decision: tuple[float, ...], /) -> bool:
        del decision
        return True

    def score(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        return self.evaluate(decision)

    def decision_mode(self, decision: tuple[float, ...], /) -> int | None:
        del decision
        return None

    def reference_front(self) -> tuple[tuple[float, ...], ...]:
        if self.objective_dimension != 3:
            raise ValueError("this benchmark currently generates a 3-D DTLZ2 reference")
        return tuple(
            (
                math.cos(first_angle) * math.cos(second_angle),
                math.cos(first_angle) * math.sin(second_angle),
                math.sin(first_angle),
            )
            for first_angle in (index * math.pi / 80.0 for index in range(41))
            for second_angle in (index * math.pi / 80.0 for index in range(41))
        )


@functools.cache
def _rotation(dimension: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    generator = np.random.default_rng(20261013 + dimension)
    matrix, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class IrregularParetoProblem:
    """Controlled irregular forward map with a known ZDT1 objective front."""

    variant: str
    decision_dimension: int = 20

    @property
    def objective_dimension(self) -> int:
        return 3 if self.variant == "multimodal_three_objective" else 2

    @property
    def name(self) -> str:
        variant = self.variant.upper().replace("_", "-")
        return f"IRREGULAR-{variant}-{self.decision_dimension}D"

    def _target(
        self, first: float, branch: int
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        indices = np.arange(1, self.decision_dimension, dtype=np.float64)
        if branch == 0:
            return 0.30 + 0.20 * np.sin(2.0 * np.pi * first + 0.73 * indices)
        return 0.70 + 0.20 * np.cos(2.0 * np.pi * first + 0.51 * indices)

    def _distance(self, decision: tuple[float, ...]) -> float:
        values = np.asarray(decision, dtype=np.float64)
        first = float(values[0])
        if self.variant == "multimodal_three_objective":
            second = float(values[1])
            indices = np.arange(2, self.decision_dimension, dtype=np.float64)
            target_a = 0.30 + 0.20 * np.sin(
                2.0 * np.pi * first + 0.37 * indices
            ) * np.cos(2.0 * np.pi * second)
            target_b = 0.70 + 0.20 * np.cos(
                2.0 * np.pi * second + 0.53 * indices
            ) * np.sin(2.0 * np.pi * first)
            return float(
                min(
                    np.mean(np.square(values[2:] - target_a)),
                    np.mean(np.square(values[2:] - target_b)),
                )
            )
        residual_a = values[1:] - self._target(first, 0)
        residual_b = values[1:] - self._target(first, 1)
        if self.variant in {"coupled_manifold", "noisy_coupled"}:
            rotated = _rotation(self.decision_dimension - 1) @ residual_a
            weights = np.linspace(0.6, 1.4, self.decision_dimension - 1)
            adjacent = np.mean(np.abs(rotated[:-1] * rotated[1:]))
            return float(np.mean(np.square(weights * rotated)) + 0.20 * adjacent)
        if self.variant == "multimodal_pullback":
            return float(
                min(np.mean(np.square(residual_a)), np.mean(np.square(residual_b)))
            )
        if self.variant == "switched_failures":
            residual = residual_a if int(first * 8.0) % 2 == 0 else residual_b
            return float(np.mean(np.square(residual)))
        if self.variant == "rugged_multimodal":
            distance = min(
                np.mean(np.square(residual_a)),
                np.mean(np.square(residual_b)),
            )
            phase = float(np.dot(values, np.arange(1, len(values) + 1)))
            return float(distance * (1.0 + 0.75 * math.sin(13.0 * phase) ** 2))
        raise ValueError(f"unknown irregular problem variant: {self.variant}")

    def is_valid(self, decision: tuple[float, ...], /) -> bool:
        return self.variant != "switched_failures" or self._distance(decision) <= 0.18

    def decision_mode(self, decision: tuple[float, ...], /) -> int | None:
        values = np.asarray(decision, dtype=np.float64)
        first = float(values[0])
        if self.variant == "multimodal_three_objective":
            second = float(values[1])
            indices = np.arange(2, self.decision_dimension, dtype=np.float64)
            target_a = 0.30 + 0.20 * np.sin(
                2.0 * np.pi * first + 0.37 * indices
            ) * np.cos(2.0 * np.pi * second)
            target_b = 0.70 + 0.20 * np.cos(
                2.0 * np.pi * second + 0.53 * indices
            ) * np.sin(2.0 * np.pi * first)
            distance_a = np.mean(np.square(values[2:] - target_a))
            distance_b = np.mean(np.square(values[2:] - target_b))
        elif self.variant in {"multimodal_pullback", "rugged_multimodal"}:
            distance_a = np.mean(np.square(values[1:] - self._target(first, 0)))
            distance_b = np.mean(np.square(values[1:] - self._target(first, 1)))
        else:
            return None
        return int(distance_b < distance_a)

    def score(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        first = decision[0]
        distance = self._distance(decision)
        if not self.is_valid(decision):
            return (10.0 + distance + first, 11.0 + distance - first)
        if self.variant == "multimodal_three_objective":
            second = decision[1]
            g = 1.0 + distance
            return (
                g * math.cos(first * math.pi / 2.0) * math.cos(second * math.pi / 2.0),
                g * math.cos(first * math.pi / 2.0) * math.sin(second * math.pi / 2.0),
                g * math.sin(first * math.pi / 2.0),
            )
        g = 1.0 + 9.0 * distance
        return (first, g * (1.0 - math.sqrt(first / g)))

    def evaluate(self, decision: tuple[float, ...], /) -> tuple[float, ...]:
        clean = self.score(decision)
        if self.variant != "noisy_coupled":
            return clean
        values = np.asarray(decision, dtype=np.float64)
        indices = np.arange(1, len(values) + 1, dtype=np.float64)
        phase = float(np.dot(values, indices))
        first_noise = 0.015 * math.sin(104729.0 * math.sin(phase * 12.9898))
        second_noise = 0.015 * math.sin(130363.0 * math.sin(phase * 78.233))
        return (clean[0] + first_noise, clean[1] + second_noise)

    def reference_front(self) -> tuple[tuple[float, ...], ...]:
        if self.variant == "multimodal_three_objective":
            return tuple(
                (
                    math.cos(first_angle) * math.cos(second_angle),
                    math.cos(first_angle) * math.sin(second_angle),
                    math.sin(first_angle),
                )
                for first_angle in (index * math.pi / 80.0 for index in range(41))
                for second_angle in (index * math.pi / 80.0 for index in range(41))
            )
        return tuple(
            (first, 1.0 - math.sqrt(first))
            for first in (index / 1000.0 for index in range(1001))
        )


class ReflectedParameterSampler:
    """Reflect an injected sampler's proposals into a finite benchmark box."""

    __slots__ = ("_sampler", "_space")

    def __init__(self, sampler: ParameterSampler, space: ParameterSpace) -> None:
        self._sampler = sampler
        self._space = space

    @property
    def sampler(self) -> ParameterSampler:
        return self._sampler

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
        proposals = self._sampler.sample(
            sample_count,
            random_seed=random_seed,
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            evaluated_decisions=evaluated_decisions,
            evaluated_objectives=evaluated_objectives,
        )
        return tuple(self._reflect(proposal) for proposal in proposals)

    def _reflect(self, proposal: tuple[float, ...]) -> tuple[float, ...]:
        values: list[float] = []
        for value, bound in zip(proposal, self._space.bounds, strict=True):
            period_position = (value - bound.lower) % (2.0 * bound.width)
            reflected = (
                period_position
                if period_position <= bound.width
                else 2.0 * bound.width - period_position
            )
            values.append(bound.lower + reflected)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    problem: str
    strategy: str
    seed: int
    evaluation_count: int
    invalid_evaluation_count: int
    valid_evaluation_fraction: float
    sequential_batches: int
    nondominated_count: int | None
    inverted_generational_distance: float | None
    final_population_nondominated_count: int | None
    final_population_inverted_generational_distance: float | None
    decision_mode_count: int | None
    decision_mode_balance: float | None
    gmm_component_counts: tuple[int, ...] | None
    elapsed_seconds: float
    status: str
    error: str | None


def _space(dimension: int) -> ParameterSpace:
    return ParameterSpace(
        tuple(ParameterBound(f"x{index}", 0.0, 1.0) for index in range(dimension))
    )


def _adaptive_sampler(strategy: str, space: ParameterSpace) -> ParameterSampler:
    if strategy == "uniform":
        return UniformRandomSampler(space)
    if strategy == "global_kde":
        return ReflectedParameterSampler(
            ScipyGaussianKdeSampler(KdeBandwidthMethod.SILVERMAN_1986),
            space,
        )
    if strategy in {"gmm_kde_historical", "gmm_kde_stabilized"}:
        stabilization = (
            CovarianceStabilizationMethod.HISTORICAL_ZERO_EIGENVALUE
            if strategy == "gmm_kde_historical"
            else CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR
        )
        return ReflectedParameterSampler(
            ScipyGmmConditionedKdeSampler(
                KdeBandwidthMethod.SILVERMAN_1986,
                feature_space=GmmFeatureSpace.DECISIONS_AND_OBJECTIVES,
                component_criterion=GmmComponentCriterion.BIC,
                max_components=10,
                allocation=GmmClusterAllocation.EQUAL,
                covariance_stabilization=stabilization,
                relative_eigenvalue_floor=1e-10,
                standardize_features=False,
            ),
            space,
        )
    if strategy == "shell_contrastive_kde":
        return ReflectedParameterSampler(
            ScipyParetoShellContrastiveKdeSampler(
                KdeBandwidthMethod.SILVERMAN_1986,
                shell_weights=(1.0, 0.5),
                candidate_multiplier=4,
            ),
            space,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def _inverted_generational_distance(
    approximation: tuple[tuple[float, ...], ...],
    reference: tuple[tuple[float, ...], ...],
) -> float:
    observed = np.asarray(approximation, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    distances = np.linalg.norm(
        expected[:, np.newaxis, :] - observed[np.newaxis, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1)))


def _decision_mode_metrics(
    problem: ForwardProblem,
    decisions: tuple[tuple[float, ...], ...],
) -> tuple[int | None, float | None]:
    modes = [
        mode
        for decision in decisions
        if (mode := problem.decision_mode(decision)) is not None
    ]
    if not modes:
        return None, None
    counts = np.bincount(np.asarray(modes, dtype=np.int64))
    positive = counts[counts > 0]
    balance = float(np.min(positive) / np.max(positive)) if len(positive) > 1 else 0.0
    return len(positive), balance


def _gmm_component_counts(
    sampler: ParameterSampler,
) -> tuple[int, ...] | None:
    if not isinstance(sampler, ReflectedParameterSampler):
        return None
    inner = sampler.sampler
    if not isinstance(inner, ScipyGmmConditionedKdeSampler):
        return None
    return inner.component_count_history


def run_forward_benchmark(
    problem: ForwardProblem,
    strategy: str,
    seed: int,
    stage_counts: tuple[int, ...],
) -> BenchmarkRun:
    """Spend the declared budget only through calls to ``problem.evaluate``."""
    space = _space(problem.decision_dimension)
    initial = UniformRandomSampler(space)
    adaptive = _adaptive_sampler(strategy, space)
    samplers = (initial,) + (adaptive,) * (len(stage_counts) - 1)
    optimizer = ParetoOptimizer(
        space,
        tuple(
            ParetoSamplingStage(sampler, count)
            for sampler, count in zip(samplers, stage_counts, strict=True)
        ),
        seed=seed,
        run_id=f"benchmark-{problem.name.lower()}-{strategy}-{seed}",
    )

    evaluation_count = 0
    invalid_evaluation_count = 0
    started = time.perf_counter()
    try:
        for count in stage_counts:
            candidates = tuple(optimizer.ask() for _ in range(count))
            for candidate in candidates:
                objectives = problem.evaluate(candidate.values)
                evaluation_count += 1
                invalid_evaluation_count += not problem.is_valid(candidate.values)
                optimizer.tell(candidate, objectives)
        elapsed = time.perf_counter() - started
        archive_decisions = tuple(
            candidate.values for candidate, _ in optimizer.archive
        )
        mode_count, mode_balance = _decision_mode_metrics(problem, archive_decisions)
        return BenchmarkRun(
            problem=problem.name,
            strategy=strategy,
            seed=seed,
            evaluation_count=evaluation_count,
            invalid_evaluation_count=invalid_evaluation_count,
            valid_evaluation_fraction=(
                1.0 - invalid_evaluation_count / evaluation_count
            ),
            sequential_batches=len(stage_counts),
            nondominated_count=len(optimizer.pareto_front),
            inverted_generational_distance=_inverted_generational_distance(
                tuple(
                    problem.score(candidate.values)
                    for candidate, _ in optimizer.archive
                ),
                problem.reference_front(),
            ),
            final_population_nondominated_count=None,
            final_population_inverted_generational_distance=None,
            decision_mode_count=mode_count,
            decision_mode_balance=mode_balance,
            gmm_component_counts=_gmm_component_counts(adaptive),
            elapsed_seconds=elapsed,
            status="passed",
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - benchmark records algorithm failures
        return BenchmarkRun(
            problem=problem.name,
            strategy=strategy,
            seed=seed,
            evaluation_count=evaluation_count,
            invalid_evaluation_count=invalid_evaluation_count,
            valid_evaluation_fraction=(
                1.0 - invalid_evaluation_count / evaluation_count
                if evaluation_count
                else 1.0
            ),
            sequential_batches=len(stage_counts),
            nondominated_count=None,
            inverted_generational_distance=None,
            final_population_nondominated_count=None,
            final_population_inverted_generational_distance=None,
            decision_mode_count=None,
            decision_mode_balance=None,
            gmm_component_counts=_gmm_component_counts(adaptive),
            elapsed_seconds=time.perf_counter() - started,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


def run_pymoo_nsga2_forward_benchmark(
    problem: ForwardProblem,
    seed: int,
    evaluation_budget: int,
    *,
    population_size: int,
    offspring_size: int,
    strategy: str,
    initial_decisions: tuple[tuple[float, ...], ...] | None = None,
) -> BenchmarkRun:
    """Run an explicit pymoo 0.6.2 NSGA-II schedule through the evaluator."""
    pymoo = importlib.import_module("pymoo")
    if pymoo.__version__ != "0.6.2":
        raise RuntimeError("the NSGA-II baseline requires pymoo 0.6.2")
    ElementwiseProblem = importlib.import_module(
        "pymoo.core.problem"
    ).ElementwiseProblem
    NSGA2 = importlib.import_module("pymoo.algorithms.moo.nsga2").NSGA2
    binary_tournament = importlib.import_module(
        "pymoo.algorithms.moo.nsga2"
    ).binary_tournament
    FloatRandomSampling = importlib.import_module(
        "pymoo.operators.sampling.rnd"
    ).FloatRandomSampling
    TournamentSelection = importlib.import_module(
        "pymoo.operators.selection.tournament"
    ).TournamentSelection
    SBX = importlib.import_module("pymoo.operators.crossover.sbx").SBX
    PM = importlib.import_module("pymoo.operators.mutation.pm").PM
    RankAndCrowding = importlib.import_module(
        "pymoo.operators.survival.rank_and_crowding"
    ).RankAndCrowding
    minimize = importlib.import_module("pymoo.optimize").minimize

    class ForwardAdapter(ElementwiseProblem):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(
                n_var=problem.decision_dimension,
                n_obj=problem.objective_dimension,
                xl=np.zeros(problem.decision_dimension),
                xu=np.ones(problem.decision_dimension),
            )
            self.decisions: list[tuple[float, ...]] = []
            self.objectives: list[tuple[float, ...]] = []
            self.invalid_evaluation_count = 0

        def _evaluate(
            self,
            decision: np.ndarray[Any, np.dtype[np.float64]],
            output: dict[str, object],
            *args: object,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            normalized = tuple(float(value) for value in decision)
            objectives = problem.evaluate(normalized)
            self.invalid_evaluation_count += not problem.is_valid(normalized)
            self.decisions.append(normalized)
            self.objectives.append(objectives)
            output["F"] = np.asarray(objectives, dtype=np.float64)

    adapter = ForwardAdapter()
    sampling = (
        FloatRandomSampling()
        if initial_decisions is None
        else np.asarray(initial_decisions, dtype=np.float64)
    )
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=offspring_size,
        sampling=sampling,
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(
            prob=0.9,
            prob_var=1.0 / problem.decision_dimension,
            eta=20,
        ),
        survival=RankAndCrowding(),
        eliminate_duplicates=True,
    )
    started = time.perf_counter()
    try:
        result = minimize(
            adapter,
            algorithm,
            termination=("n_eval", evaluation_budget),
            seed=seed,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        evaluation_count = len(adapter.objectives)
        if evaluation_count != evaluation_budget:
            message = (
                f"pymoo used {evaluation_count} evaluations, "
                f"expected {evaluation_budget}"
            )
            raise RuntimeError(message)
        remaining_evaluations = evaluation_budget - population_size
        if remaining_evaluations < 0 or remaining_evaluations % offspring_size:
            raise RuntimeError("NSGA-II schedule does not divide the evaluation budget")
        sequential_batches = 1 + remaining_evaluations // offspring_size
        archive_indices = pareto(tuple(adapter.objectives))
        archive_front = tuple(
            problem.score(adapter.decisions[index]) for index in archive_indices
        )
        final_decisions = tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(result.X, dtype=np.float64)
        )
        final_front = tuple(problem.score(decision) for decision in final_decisions)
        archive_decisions = tuple(adapter.decisions[index] for index in archive_indices)
        mode_count, mode_balance = _decision_mode_metrics(problem, archive_decisions)
        reference = problem.reference_front()
        return BenchmarkRun(
            problem=problem.name,
            strategy=strategy,
            seed=seed,
            evaluation_count=evaluation_count,
            invalid_evaluation_count=adapter.invalid_evaluation_count,
            valid_evaluation_fraction=(
                1.0 - adapter.invalid_evaluation_count / evaluation_count
            ),
            sequential_batches=sequential_batches,
            nondominated_count=len(archive_front),
            inverted_generational_distance=_inverted_generational_distance(
                archive_front,
                reference,
            ),
            final_population_nondominated_count=len(final_front),
            final_population_inverted_generational_distance=(
                _inverted_generational_distance(final_front, reference)
            ),
            decision_mode_count=mode_count,
            decision_mode_balance=mode_balance,
            gmm_component_counts=None,
            elapsed_seconds=elapsed,
            status="passed",
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - benchmark records baseline failures
        return BenchmarkRun(
            problem=problem.name,
            strategy=strategy,
            seed=seed,
            evaluation_count=len(adapter.objectives),
            invalid_evaluation_count=adapter.invalid_evaluation_count,
            valid_evaluation_fraction=(
                1.0 - adapter.invalid_evaluation_count / len(adapter.objectives)
                if adapter.objectives
                else 1.0
            ),
            sequential_batches=0,
            nondominated_count=None,
            inverted_generational_distance=None,
            final_population_nondominated_count=None,
            final_population_inverted_generational_distance=None,
            decision_mode_count=None,
            decision_mode_balance=None,
            gmm_component_counts=None,
            elapsed_seconds=time.perf_counter() - started,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


def _median_gmm_component_counts(
    runs: list[BenchmarkRun],
) -> tuple[float, ...] | None:
    histories = [
        run.gmm_component_counts for run in runs if run.gmm_component_counts is not None
    ]
    if not histories:
        return None
    history_length = len(histories[0])
    if any(len(history) != history_length for history in histories):
        return None
    return tuple(
        statistics.median(history[index] for history in histories)
        for index in range(history_length)
    )


def _aggregates(runs: tuple[BenchmarkRun, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[BenchmarkRun]] = {}
    for run in runs:
        groups.setdefault((run.problem, run.strategy), []).append(run)
    result: list[dict[str, object]] = []
    for (problem, strategy), group in sorted(groups.items()):
        passed = [run for run in group if run.status == "passed"]
        result.append(
            {
                "problem": problem,
                "strategy": strategy,
                "runs": len(group),
                "passed": len(passed),
                "median_igd": (
                    statistics.median(
                        run.inverted_generational_distance
                        for run in passed
                        if run.inverted_generational_distance is not None
                    )
                    if passed
                    else None
                ),
                "median_nondominated_count": (
                    statistics.median(
                        run.nondominated_count
                        for run in passed
                        if run.nondominated_count is not None
                    )
                    if passed
                    else None
                ),
                "median_valid_evaluation_fraction": (
                    statistics.median(run.valid_evaluation_fraction for run in passed)
                    if passed
                    else None
                ),
                "median_gmm_component_counts": (_median_gmm_component_counts(passed)),
                "median_decision_mode_count": (
                    statistics.median(
                        run.decision_mode_count
                        for run in passed
                        if run.decision_mode_count is not None
                    )
                    if any(run.decision_mode_count is not None for run in passed)
                    else None
                ),
                "median_decision_mode_balance": (
                    statistics.median(
                        run.decision_mode_balance
                        for run in passed
                        if run.decision_mode_balance is not None
                    )
                    if any(run.decision_mode_balance is not None for run in passed)
                    else None
                ),
                "median_sequential_batches": (
                    statistics.median(run.sequential_batches for run in passed)
                    if passed
                    else None
                ),
                "median_final_population_igd": (
                    statistics.median(
                        run.final_population_inverted_generational_distance
                        for run in passed
                        if run.final_population_inverted_generational_distance
                        is not None
                    )
                    if any(
                        run.final_population_inverted_generational_distance is not None
                        for run in passed
                    )
                    else None
                ),
                "median_elapsed_seconds": (
                    statistics.median(run.elapsed_seconds for run in passed)
                    if passed
                    else None
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--initial", type=int, default=1000)
    parser.add_argument("--adaptive", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--suite",
        choices=("standard", "irregular"),
        default="standard",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/pareto_shells_forward.json"),
    )
    arguments = parser.parse_args()

    stage_counts = (arguments.initial,) + (arguments.adaptive,) * arguments.rounds
    if arguments.suite == "standard":
        problems: tuple[ForwardProblem, ...] = (
            ZdtProblem(1, 10),
            ZdtProblem(1, 30),
            ZdtProblem(3, 10),
            ZdtProblem(3, 30),
            Dtlz2Problem(12),
            Dtlz2Problem(30),
        )
    else:
        problems = tuple(
            IrregularParetoProblem(variant)
            for variant in (
                "coupled_manifold",
                "multimodal_pullback",
                "rugged_multimodal",
                "switched_failures",
                "noisy_coupled",
                "multimodal_three_objective",
            )
        )
    strategies = (
        "uniform",
        "global_kde",
        "gmm_kde_historical",
        "gmm_kde_stabilized",
        "shell_contrastive_kde",
    )
    native_runs = tuple(
        run_forward_benchmark(problem, strategy, seed, stage_counts)
        for problem in problems
        for strategy in strategies
        for seed in range(arguments.seeds)
    )
    evaluation_budget = sum(stage_counts)
    typical_nsga2_runs = tuple(
        run_pymoo_nsga2_forward_benchmark(
            problem,
            seed,
            evaluation_budget,
            population_size=100,
            offspring_size=100,
            strategy="pymoo_nsga2_typical",
        )
        for problem in problems
        for seed in range(arguments.seeds)
    )
    if len(set(stage_counts[1:])) != 1:
        raise ValueError("batch-matched NSGA-II requires equal adaptive stage counts")
    batch_matched_nsga2_runs = tuple(
        run_pymoo_nsga2_forward_benchmark(
            problem,
            seed,
            evaluation_budget,
            population_size=stage_counts[0],
            offspring_size=stage_counts[1],
            strategy="pymoo_nsga2_batch_matched",
            initial_decisions=UniformRandomSampler(
                _space(problem.decision_dimension)
            ).sample(stage_counts[0], random_seed=seed),
        )
        for problem in problems
        for seed in range(arguments.seeds)
    )
    runs = native_runs + typical_nsga2_runs + batch_matched_nsga2_runs
    document = {
        "schema_version": 6,
        "generated_at": datetime.now(UTC).isoformat(),
        "forward_evaluation_only": True,
        "suite": arguments.suite,
        "problems": [problem.name for problem in problems],
        "stage_counts": stage_counts,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymoo": importlib.import_module("pymoo").__version__,
        "scikit_learn": importlib.import_module("sklearn").__version__,
        "gmm_conditioned_kde": {
            "features": "unscaled decision and objective vectors",
            "component_criterion": "BIC",
            "maximum_components": 10,
            "covariance_type": "full",
            "gmm_random_state": 0,
            "cluster_allocation": "equal",
            "bandwidth": "Silverman1986",
            "historical_covariance_repair": "replace exact zero eigenvalues with 1e-15",
            "modern_covariance_repair": {
                "method": "symmetric relative eigenvalue floor",
                "relative_floor": 1e-10,
            },
        },
        "nsga2": {
            "typical": {
                "population_size": 100,
                "offspring_size": 100,
                "sequential_batches": evaluation_budget // 100,
            },
            "batch_matched": {
                "population_size": stage_counts[0],
                "offspring_size": stage_counts[1],
                "sequential_batches": len(stage_counts),
                "shared_initial_population": True,
            },
            "selection": "binary tournament by rank and crowding",
            "crossover": {"type": "SBX", "probability": 0.9, "eta": 15},
            "mutation": {
                "type": "polynomial",
                "individual_probability": 0.9,
                "variable_probability": "1 / decision_dimension",
                "eta": 20,
            },
            "survival": "rank and crowding",
            "eliminate_duplicates": True,
        },
        "runs": [asdict(run) for run in runs],
        "aggregates": _aggregates(runs),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["aggregates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
