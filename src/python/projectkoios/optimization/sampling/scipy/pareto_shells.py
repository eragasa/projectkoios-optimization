"""Contrastive KDE sampling from recursively pulled-back Pareto shells.

This is an owner-authored research capability, not a recovered PyPosPack behavior. It
uses the first Pareto shell as the proposal density and lower-ranked shells as negative
density evidence. A bounded candidate pool is ranked by a positive-to-negative log
density ratio, so selected proposals remain near shell zero while preferring its side
away from dominated shells.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import stats

from projectkoios.numerics.validation import finite_float, integer, u32

from ....numerics.numpy.typing import Float64Array
from ....numerics.scipy.typing import GaussianKde
from ...multiobjective.pareto.shells import pareto_shells
from .kde import KdeBandwidthFactory, KdeBandwidthMethod


class ScipyParetoShellContrastiveKdeSampler:
    """Sample near Pareto shell zero and away from dominated decision-space shells."""

    __slots__ = (
        "_bandwidth_method",
        "_candidate_multiplier",
        "_chunk_size",
        "_shell_weights",
    )

    def __init__(
        self,
        bandwidth_method: KdeBandwidthMethod = KdeBandwidthMethod.SILVERMAN_1986,
        *,
        shell_weights: tuple[float, ...] = (1.0, 0.5),
        candidate_multiplier: int = 8,
        chunk_size: int = 500,
    ) -> None:
        KdeBandwidthFactory.create(bandwidth_method)
        if type(shell_weights) is not tuple or not shell_weights:
            raise TypeError("shell_weights must be a nonempty tuple")
        normalized_weights = tuple(
            finite_float(weight, "shell weight") for weight in shell_weights
        )
        if any(weight <= 0.0 for weight in normalized_weights):
            raise ValueError("shell weights must be positive")
        multiplier = integer(candidate_multiplier, "candidate_multiplier")
        if multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        size = integer(chunk_size, "chunk_size")
        if size < 1:
            raise ValueError("chunk_size must be positive")
        self._bandwidth_method = bandwidth_method
        self._shell_weights = normalized_weights
        self._candidate_multiplier = multiplier
        self._chunk_size = size

    @property
    def bandwidth_method(self) -> KdeBandwidthMethod:
        """Return the bandwidth method applied independently to every shell."""
        return self._bandwidth_method

    @property
    def shell_weights(self) -> tuple[float, ...]:
        """Return positive negative-density weights for shells one onward."""
        return self._shell_weights

    @property
    def candidate_multiplier(self) -> int:
        """Return the bounded proposal-pool multiplier."""
        return self._candidate_multiplier

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
        """Rank shell-zero KDE proposals against dominated-shell KDE densities."""
        del pareto_set, pareto_front
        count = u32(sample_count, "sample_count", allow_zero=False)
        seed = u32(random_seed, "random_seed", allow_zero=True)
        candidate_count = u32(
            count * self._candidate_multiplier,
            "candidate pool size",
            allow_zero=False,
        )
        shells = pareto_shells(
            evaluated_decisions,
            evaluated_objectives,
            shell_count=len(self._shell_weights) + 1,
            chunk_size=self._chunk_size,
        )
        if len(shells) < 2:
            raise ValueError(
                "contrastive sampling requires Pareto shell zero and at least one "
                "dominated shell"
            )

        positive_source = np.asarray(
            shells[0].decision_vectors,
            dtype=np.float64,
        ).T
        positive_kde = self._fit_kde(positive_source)
        negative_kdes: list[tuple[float, GaussianKde]] = []
        for weight, shell in zip(self._shell_weights, shells[1:], strict=False):
            source = np.asarray(shell.decision_vectors, dtype=np.float64).T
            negative_kdes.append((weight, self._fit_kde(source)))

        random_state = np.random.RandomState(seed)
        candidate_matrix = np.empty(
            (positive_source.shape[0], candidate_count),
            dtype=np.float64,
        )
        for index in range(candidate_count):
            candidate_matrix[:, index : index + 1] = positive_kde.resample(
                1,
                seed=random_state,
            )

        positive_log_density = positive_kde.logpdf(candidate_matrix)
        negative_terms = np.vstack(
            [
                np.log(weight) + kde.logpdf(candidate_matrix)
                for weight, kde in negative_kdes
            ]
        )
        negative_log_density = np.logaddexp.reduce(negative_terms, axis=0)
        scores = positive_log_density - negative_log_density
        selected = np.argsort(-scores, kind="stable")[:count]
        return tuple(
            tuple(
                float(candidate_matrix[row, index])
                for row in range(candidate_matrix.shape[0])
            )
            for index in selected
        )

    def _fit_kde(self, source: Float64Array) -> GaussianKde:
        bandwidth_function = KdeBandwidthFactory.create(self._bandwidth_method)
        bandwidth = bandwidth_function(source)
        return cast(GaussianKde, stats.gaussian_kde(source, bandwidth))


__all__ = ["ScipyParetoShellContrastiveKdeSampler"]
