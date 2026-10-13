from __future__ import annotations

from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from projectkoios.optimization.probability.factory import (  # noqa: E402
    ProbabilitySamplerFactory,
)
from projectkoios.optimization.sampling.scipy.gmm_kde import (  # noqa: E402
    CovarianceStabilizationMethod,
    GmmClusterAllocation,
    GmmComponentCriterion,
    GmmFeatureSpace,
    ScipyGmmConditionedKdeSampler,
    historical_zero_eigenvalue_covariance_repair,
)
from projectkoios.optimization.sampling.scipy.kde import (  # noqa: E402
    KdeBandwidthMethod,
)

pytestmark = pytest.mark.integration


def _two_mode_archive() -> tuple[
    tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]
]:
    generator = np.random.default_rng(1024)
    first = generator.normal((0.2, 0.25), 0.015, size=(60, 2))
    second = generator.normal((0.8, 0.75), 0.015, size=(60, 2))
    decisions = np.concatenate((first, second), axis=0)
    objectives = np.column_stack(
        (
            decisions[:, 0],
            1.0 - np.sqrt(np.clip(decisions[:, 0], 0.0, 1.0)),
        )
    )
    return (
        tuple(tuple(float(value) for value in row) for row in decisions),
        tuple(tuple(float(value) for value in row) for row in objectives),
    )


def test_equal_cluster_local_sampling_preserves_two_modes() -> None:
    pareto_set, pareto_front = _two_mode_archive()
    sampler = ScipyGmmConditionedKdeSampler(
        KdeBandwidthMethod.SILVERMAN_1986,
        feature_space=GmmFeatureSpace.DECISIONS_AND_OBJECTIVES,
        component_criterion=GmmComponentCriterion.BIC,
        max_components=2,
        allocation=GmmClusterAllocation.EQUAL,
        covariance_stabilization=(
            CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR
        ),
    )

    samples = sampler.sample(
        200,
        random_seed=7,
        pareto_set=pareto_set,
        pareto_front=pareto_front,
    )

    assert len(samples) == 200
    assert sampler.component_count_history == (2,)
    assert sum(sample[0] < 0.4 for sample in samples) == 100
    assert sum(sample[0] > 0.6 for sample in samples) == 100
    assert all(sample[0] < 0.4 or sample[0] > 0.6 for sample in samples)


def test_probability_factory_constructs_gmm_conditioned_sampler() -> None:
    sampler = ProbabilitySamplerFactory().create_gmm_conditioned_kde(
        KdeBandwidthMethod.SILVERMAN_1986,
        feature_space=GmmFeatureSpace.DECISIONS_AND_OBJECTIVES,
        component_criterion=GmmComponentCriterion.BIC,
        max_components=10,
        allocation=GmmClusterAllocation.EQUAL,
        covariance_stabilization=(
            CovarianceStabilizationMethod.HISTORICAL_ZERO_EIGENVALUE
        ),
    )

    assert type(sampler) is ScipyGmmConditionedKdeSampler


def test_component_selection_rejects_any_undersized_cluster() -> None:
    generator = np.random.default_rng(44)
    large = generator.normal((0.2, 0.2), 0.01, size=(60, 2))
    small = generator.normal((0.9, 0.9), 0.001, size=(2, 2))
    decisions = np.concatenate((large, small), axis=0)
    pareto_set = tuple(tuple(float(value) for value in row) for row in decisions)
    pareto_front = tuple((float(row[0]), 1.0 - float(row[0])) for row in decisions)
    sampler = ScipyGmmConditionedKdeSampler(
        max_components=2,
        covariance_stabilization=(
            CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR
        ),
    )

    sampler.sample(
        20,
        random_seed=2,
        pareto_set=pareto_set,
        pareto_front=pareto_front,
    )

    assert sampler.component_count_history == (1,)


def test_sampling_is_seed_deterministic() -> None:
    pareto_set, pareto_front = _two_mode_archive()
    sampler = ScipyGmmConditionedKdeSampler(
        max_components=2,
        covariance_stabilization=(
            CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR
        ),
    )
    first = sampler.sample(
        20,
        random_seed=23,
        pareto_set=pareto_set,
        pareto_front=pareto_front,
    )
    second = sampler.sample(
        20,
        random_seed=23,
        pareto_set=pareto_set,
        pareto_front=pareto_front,
    )

    assert first == second


def test_relative_floor_samples_a_rank_deficient_pullback() -> None:
    pareto_set = tuple(
        (value, 2.0 * value, 3.0 * value) for value in np.linspace(0.0, 1.0, 8)
    )
    pareto_front = tuple((value, 1.0 - value) for value in np.linspace(0.0, 1.0, 8))
    sampler = ScipyGmmConditionedKdeSampler(
        max_components=1,
        covariance_stabilization=(
            CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR
        ),
    )

    samples = sampler.sample(
        16,
        random_seed=11,
        pareto_set=pareto_set,
        pareto_front=pareto_front,
    )

    assert np.all(np.isfinite(np.asarray(samples)))


def test_historical_repair_replaces_only_exact_zero_eigenvalues() -> None:
    covariance = np.diag((4.0, 0.0))

    repaired = historical_zero_eigenvalue_covariance_repair(covariance)

    assert repaired == pytest.approx(np.diag((4.0, 1e-15)), rel=0.0, abs=0.0)
