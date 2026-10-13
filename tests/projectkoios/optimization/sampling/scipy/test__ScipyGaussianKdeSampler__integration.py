from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
stats: Any = pytest.importorskip("scipy.stats")

from projectkoios.optimization.probability.factory import (  # noqa: E402
    ProbabilitySamplerFactory,
)
from projectkoios.optimization.sampling.scipy.kde import (  # noqa: E402
    Chiu1999_h,
    KdeBandwidthFactory,
    KdeBandwidthMethod,
    ScipyGaussianKdeSampler,
    Silverman1986_h,
)

pytestmark = pytest.mark.integration


def test_enum_factory_resolves_the_preserved_bandwidth_functions() -> None:
    assert KdeBandwidthFactory.create(KdeBandwidthMethod.CHIU_1999) is Chiu1999_h
    assert (
        KdeBandwidthFactory.create(KdeBandwidthMethod.SILVERMAN_1986) is Silverman1986_h
    )

    with pytest.raises(TypeError, match="KdeBandwidthMethod"):
        KdeBandwidthFactory.create("silverman1986")  # type: ignore[arg-type]


def test_chiu_1999_bandwidth_matches_the_donor_reference_case() -> None:
    source = np.asarray((-1.5, -0.7, -0.2, 0.0, 0.6, 1.1, 2.0), dtype=np.float64)

    assert Chiu1999_h(source) == 1.2509765625000018


def test_silverman_1986_bandwidth_matches_the_donor_scipy_operation() -> None:
    source = np.asarray(
        ((-1.0, -0.4, 0.2, 0.7, 1.4), (1.2, -0.3, 0.5, 1.8, -1.1)),
        dtype=np.float64,
    )

    expected = stats.gaussian_kde(source, "silverman").factor

    assert Silverman1986_h(source) == expected


def test_sampler_preserves_donor_one_vector_at_a_time_resampling_order() -> None:
    pareto_set = (
        (-1.0, 1.2),
        (-0.4, -0.3),
        (0.2, 0.5),
        (0.7, 1.8),
        (1.4, -1.1),
    )
    source = np.asarray(pareto_set, dtype=np.float64).T
    bandwidth = Silverman1986_h(source)
    donor_generator = stats.gaussian_kde(source, bandwidth)
    random_state = np.random.get_state()
    try:
        np.random.seed(1729)
        expected = tuple(
            tuple(float(value[index, 0]) for index in range(source.shape[0]))
            for value in (donor_generator.resample(1) for _ in range(6))
        )
    finally:
        np.random.set_state(random_state)

    sampler = ProbabilitySamplerFactory().create_gaussian_kde(
        KdeBandwidthMethod.SILVERMAN_1986
    )
    actual = sampler.sample(
        6,
        random_seed=1729,
        pareto_set=pareto_set,
        pareto_front=(),
    )

    assert type(sampler) is ScipyGaussianKdeSampler
    assert sampler.bandwidth_method is KdeBandwidthMethod.SILVERMAN_1986
    assert actual == expected
