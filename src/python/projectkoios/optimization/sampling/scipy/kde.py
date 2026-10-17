"""PyPosPack Gaussian KDE bandwidth selection and parameter sampling.

The bandwidth functions preserve the implementation introduced by Eugene J. Ragasa in
PyPosPack commit ``07193e06886b9f072558d84ffc73bf809ee3a3e4``. The sampler removes
only PyPosMat data-file, calculator, MPI, logging, and campaign coupling. It retains the
original data orientation, SciPy ``gaussian_kde`` construction, and one-vector-at-a-time
resampling order.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import cast

import numpy as np
from scipy import optimize, stats

from projectkoios.numerics.validation import u32

from ....numerics.numpy.typing import Float64Array
from ....numerics.scipy.typing import GaussianKde, OptimizeResult

type KdeBandwidthFunction = Callable[[Float64Array], float]
type _BandwidthCandidate = Float64Array | type[float]


def Silverman1986_h(X: Float64Array) -> float:  # noqa: N802
    """Get the smoothing factor using the original Silverman method."""
    kde = cast(GaussianKde, stats.gaussian_kde(X, "silverman"))
    return float(kde.factor)


def Chiu1999_h(X: Float64Array) -> float:  # noqa: N802
    """Preserve the original PyPosPack cross-validation implementation."""

    def fhati(
        X: Float64Array,
        h: _BandwidthCandidate,
        i: int,
    ) -> Float64Array:  # noqa: N802
        if h is float:  # noqa: SIM108 - preserve donor control flow
            _h = h
        else:
            _h = cast(Float64Array, h)[0]

        Xi = np.delete(X, i)
        kde = cast(GaussianKde, stats.gaussian_kde(Xi, _h))
        return kde(X[i])

    def J(
        X: Float64Array,
        h: _BandwidthCandidate,
    ) -> float | Float64Array:  # noqa: N802
        if h is float:  # noqa: SIM108 - preserve donor control flow
            _h = h
        else:
            _h = cast(Float64Array, h)[0]

        fhat = cast(GaussianKde, stats.gaussian_kde(X, _h))
        # F1 = integrate.quad(lambda x: fhat(x)**2, -np.inf, np.inf)[0]
        F1 = fhat.integrate_kde(fhat)
        F2 = np.array([fhati(X, h, i) for i in range(X.shape[0])])
        return cast(float | Float64Array, F1 - 2 * np.mean(F2))

    # h0 = Silverman1986_h(X)
    h0 = 0.5
    results = cast(
        OptimizeResult,
        optimize.minimize(lambda h: J(X, h), h0, method="Nelder-Mead"),
    )

    return float(results.x[0])


def chiu1999_h(X: Float64Array) -> float:  # noqa: N803
    """Retain the original lowercase compatibility name."""
    return Chiu1999_h(X)


def silverman1986_h(X: Float64Array) -> float:  # noqa: N803
    """Retain the original lowercase compatibility name."""
    return Silverman1986_h(X)


class KdeBandwidthMethod(Enum):
    """Bandwidth methods retained from the PyPosPack KDE sampler."""

    CHIU_1999 = "chiu1999"
    SILVERMAN_1986 = "silverman1986"


class KdeBandwidthFactory:
    """Resolve an explicit bandwidth-method enum to its preserved implementation."""

    __slots__ = ()

    @staticmethod
    def create(method: KdeBandwidthMethod, /) -> KdeBandwidthFunction:
        """Return the exact bandwidth function selected by ``method``."""
        if type(method) is not KdeBandwidthMethod:
            raise TypeError("method must be KdeBandwidthMethod")
        if method is KdeBandwidthMethod.CHIU_1999:
            return Chiu1999_h
        return Silverman1986_h


class ScipyGaussianKdeSampler:
    """Resample decision vectors from the preceding Pareto-set approximation."""

    __slots__ = ("_bandwidth_method",)

    def __init__(
        self,
        bandwidth_method: KdeBandwidthMethod = KdeBandwidthMethod.CHIU_1999,
    ) -> None:
        KdeBandwidthFactory.create(bandwidth_method)
        self._bandwidth_method = bandwidth_method

    @property
    def bandwidth_method(self) -> KdeBandwidthMethod:
        """Return the configured bandwidth-method enum."""
        return self._bandwidth_method

    def determine_kde_bandwidth(self, X: Float64Array) -> float:  # noqa: N803
        """Return bandwidth ``h`` from the selected preserved implementation."""
        function = KdeBandwidthFactory.create(self._bandwidth_method)
        return function(X)

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
        """Fit the preceding decision vectors and preserve donor resampling order."""
        del pareto_front, evaluated_decisions, evaluated_objectives
        count = u32(sample_count, "sample_count", allow_zero=False)
        seed = u32(random_seed, "random_seed", allow_zero=True)
        if not pareto_set:
            raise ValueError("pareto_set must contain KDE source decision vectors")

        X = np.asarray(pareto_set, dtype=np.float64).T
        kde_bw = self.determine_kde_bandwidth(X)
        rv_generator = cast(GaussianKde, stats.gaussian_kde(X, kde_bw))
        random_state = np.random.RandomState(seed)

        samples: list[tuple[float, ...]] = []
        for _ in range(count):
            free_parameters = rv_generator.resample(1, seed=random_state)
            samples.append(
                tuple(float(free_parameters[index, 0]) for index in range(X.shape[0]))
            )
        return tuple(samples)


__all__ = [
    "Chiu1999_h",
    "KdeBandwidthFactory",
    "KdeBandwidthFunction",
    "KdeBandwidthMethod",
    "ScipyGaussianKdeSampler",
    "Silverman1986_h",
    "chiu1999_h",
    "silverman1986_h",
]
