from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("scipy.stats")

from projectkoios.optimization.probability.factory import (  # noqa: E402
    ProbabilitySamplerFactory,
)
from projectkoios.optimization.sampling.scipy.kde import (  # noqa: E402
    KdeBandwidthMethod,
)
from projectkoios.optimization.sampling.scipy.pareto_shells import (  # noqa: E402
    ScipyParetoShellContrastiveKdeSampler,
)

pytestmark = pytest.mark.integration


def _three_shell_observations() -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    decisions = tuple(
        (value,)
        for value in (
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            -1.2,
            -1.1,
            -1.0,
            -0.9,
            -0.8,
        )
    )
    objectives = tuple(
        (float(left + offset), float(4 - left + offset))
        for offset in range(3)
        for left in range(5)
    )
    return decisions, objectives


def test_factory_constructs_explicit_contrastive_sampler() -> None:
    sampler = ProbabilitySamplerFactory().create_pareto_shell_contrastive_kde(
        KdeBandwidthMethod.SILVERMAN_1986,
        shell_weights=(1.0, 0.5),
        candidate_multiplier=16,
    )

    assert type(sampler) is ScipyParetoShellContrastiveKdeSampler
    assert sampler.shell_weights == (1.0, 0.5)
    assert sampler.candidate_multiplier == 16


def test_contrastive_sampling_is_deterministic_and_moves_away_from_lower_shells() -> (
    None
):
    decisions, objectives = _three_shell_observations()
    sampler = ScipyParetoShellContrastiveKdeSampler(
        KdeBandwidthMethod.SILVERMAN_1986,
        shell_weights=(1.0, 0.5),
        candidate_multiplier=32,
        chunk_size=4,
    )

    first = sampler.sample(
        12,
        random_seed=1729,
        pareto_set=decisions[:5],
        pareto_front=objectives[:5],
        evaluated_decisions=decisions,
        evaluated_objectives=objectives,
    )
    second = sampler.sample(
        12,
        random_seed=1729,
        pareto_set=decisions[:5],
        pareto_front=objectives[:5],
        evaluated_decisions=decisions,
        evaluated_objectives=objectives,
    )

    assert first == second
    assert float(np.mean(first)) > float(np.mean(decisions[:5]))


def test_contrastive_sampling_requires_a_dominated_shell() -> None:
    sampler = ScipyParetoShellContrastiveKdeSampler()
    decisions = ((0.8,), (0.9,), (1.0,), (1.1,), (1.2,))
    objectives = ((0.0, 4.0), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (4.0, 0.0))

    with pytest.raises(ValueError, match="at least one dominated shell"):
        sampler.sample(
            2,
            random_seed=1,
            pareto_set=decisions,
            pareto_front=objectives,
            evaluated_decisions=decisions,
            evaluated_objectives=objectives,
        )
