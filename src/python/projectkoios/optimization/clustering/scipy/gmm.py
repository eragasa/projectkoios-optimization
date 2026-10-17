"""Scikit-learn full-covariance Gaussian-mixture model selection.

This numeric-only extraction preserves the KSD recovery implementation's feature
standardization, bounded one-through-N model inventory, AIC/BIC diagnostics, BIC
selection, deterministic canonical component ordering, and exact input-row pullback.
It does not own scientific feature names, campaign identity, report assembly, or KDE
sampling.
"""

from __future__ import annotations

from importlib import metadata
from typing import Protocol, cast

import numpy as np

from projectkoios.numerics.validation import finite_float

from ....numerics.numpy.typing import Float64Array, Int64Array
from ..gmm import (
    GaussianMixtureAssignment,
    GaussianMixtureClusteringCompleted,
    GaussianMixtureClusteringFailed,
    GaussianMixtureClusteringFailureCode,
    GaussianMixtureClusteringRequest,
    GaussianMixtureClusteringResult,
    GaussianMixtureComponent,
    GaussianMixtureModelScore,
)


class _GaussianMixtureModel(Protocol):
    n_components: int
    means_: Float64Array
    covariances_: Float64Array
    weights_: Float64Array
    converged_: bool
    n_iter_: int

    def fit(self, values: Float64Array, /) -> _GaussianMixtureModel: ...

    def aic(self, values: Float64Array, /) -> float: ...

    def bic(self, values: Float64Array, /) -> float: ...

    def predict(self, values: Float64Array, /) -> Int64Array: ...

    def predict_proba(self, values: Float64Array, /) -> Float64Array: ...


class ScipyGaussianMixtureClusterer:
    """Standardize observations and select a full-covariance GMM by BIC."""

    __slots__ = ()

    def cluster(
        self,
        observations: tuple[tuple[float, ...], ...],
        request: GaussianMixtureClusteringRequest,
        /,
    ) -> GaussianMixtureClusteringResult:
        """Return deterministic clustering or a closed numerical fit failure."""
        if type(request) is not GaussianMixtureClusteringRequest:
            raise TypeError("request must be GaussianMixtureClusteringRequest")
        if type(observations) is not tuple:
            raise TypeError("observations must be a tuple of tuple rows")
        if not observations:
            return GaussianMixtureClusteringFailed(
                GaussianMixtureClusteringFailureCode.NO_OBSERVATIONS,
                "observations must contain at least one row",
            )
        matrix = self._matrix(observations)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0.0] = 1.0
        standardized = np.asarray((matrix - means) / scales, dtype=np.float64)
        if standardized.shape[0] == 1:
            return self._single_observation(means, scales)

        from sklearn.mixture import GaussianMixture

        maximum = min(
            request.maximum_components,
            max(1, standardized.shape[0] // 2),
        )
        models: list[_GaussianMixtureModel] = []
        scores: list[GaussianMixtureModelScore] = []
        try:
            for component_count in range(1, maximum + 1):
                model = cast(
                    _GaussianMixtureModel,
                    GaussianMixture(
                        n_components=component_count,
                        covariance_type="full",
                        random_state=request.random_seed,
                        n_init=5,
                        max_iter=500,
                        reg_covar=1.0e-8,
                    ),
                )
                model.fit(standardized)
                models.append(model)
                scores.append(
                    GaussianMixtureModelScore(
                        component_count,
                        float(model.aic(standardized)),
                        float(model.bic(standardized)),
                    )
                )
        except (FloatingPointError, ValueError) as error:
            return GaussianMixtureClusteringFailed(
                GaussianMixtureClusteringFailureCode.FIT_FAILED,
                f"Gaussian-mixture fitting failed: {error}",
            )
        selected_index = min(
            range(len(models)),
            key=lambda index: (
                scores[index].bayesian_information_criterion,
                scores[index].component_count,
            ),
        )
        return self._completed(
            means,
            scales,
            standardized,
            models[selected_index],
            tuple(scores),
        )

    @staticmethod
    def _matrix(
        observations: tuple[tuple[float, ...], ...],
    ) -> Float64Array:
        rows: list[tuple[float, ...]] = []
        feature_count: int | None = None
        for row_index, row in enumerate(observations):
            if type(row) is not tuple:
                raise TypeError("observation rows must be tuples")
            if not row:
                raise ValueError("observation rows must be nonempty")
            if feature_count is None:
                feature_count = len(row)
            elif len(row) != feature_count:
                raise ValueError("observations must form a rectangular matrix")
            rows.append(
                tuple(
                    finite_float(value, f"observation {row_index} value")
                    for value in row
                )
            )
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _completed(
        means: Float64Array,
        scales: Float64Array,
        standardized: Float64Array,
        model: _GaussianMixtureModel,
        scores: tuple[GaussianMixtureModelScore, ...],
    ) -> GaussianMixtureClusteringCompleted:
        labels = model.predict(standardized)
        probabilities = model.predict_proba(standardized)
        original_order = tuple(
            sorted(
                range(model.n_components),
                key=lambda index: (
                    tuple(float(value) for value in model.means_[index]),
                    index,
                ),
            )
        )
        canonical = {original: new for new, original in enumerate(original_order)}
        assignments = tuple(
            GaussianMixtureAssignment(
                observation_index=index,
                cluster_id=canonical[int(labels[index])],
                probability=float(probabilities[index, int(labels[index])]),
            )
            for index in range(len(standardized))
        )
        components = tuple(
            GaussianMixtureComponent(
                cluster_id=canonical[original],
                weight=float(model.weights_[original]),
                mean=tuple(float(value) for value in model.means_[original]),
                covariance=tuple(
                    tuple(float(value) for value in row)
                    for row in model.covariances_[original]
                ),
            )
            for original in original_order
        )
        return GaussianMixtureClusteringCompleted(
            feature_means=tuple(float(value) for value in means),
            feature_scales=tuple(float(value) for value in scales),
            model_scores=scores,
            selected_components=model.n_components,
            assignments=assignments,
            components=components,
            converged=bool(model.converged_),
            iterations=int(model.n_iter_),
            backend_version=metadata.version("scikit-learn"),
        )

    @staticmethod
    def _single_observation(
        means: Float64Array,
        scales: Float64Array,
    ) -> GaussianMixtureClusteringCompleted:
        feature_count = len(means)
        return GaussianMixtureClusteringCompleted(
            feature_means=tuple(float(value) for value in means),
            feature_scales=tuple(float(value) for value in scales),
            model_scores=(GaussianMixtureModelScore(1, 0.0, 0.0),),
            selected_components=1,
            assignments=(GaussianMixtureAssignment(0, 0, 1.0),),
            components=(
                GaussianMixtureComponent(
                    cluster_id=0,
                    weight=1.0,
                    mean=(0.0,) * feature_count,
                    covariance=tuple(
                        tuple(0.0 for _ in range(feature_count))
                        for _ in range(feature_count)
                    ),
                ),
            ),
            converged=True,
            iterations=0,
            backend_version=metadata.version("scikit-learn"),
        )


__all__ = ["ScipyGaussianMixtureClusterer"]
