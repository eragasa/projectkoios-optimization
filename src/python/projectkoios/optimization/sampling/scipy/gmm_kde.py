"""PyPosPack-lineage GMM-conditioned local Gaussian KDE sampling.

Historical evidence fits full-covariance scikit-learn Gaussian mixtures with a fixed
``random_state=0``, selects component counts with AIC/BIC, and assigns every retained
observation to a cluster. The cluster sampling engine subsequently fits a separate KDE
to each cluster's decision vectors and requests the same sample count per cluster.

This extraction removes files, campaigns, MPI, plotting, and scientific column names.
It makes three adaptations explicit: deterministic sorted cluster order, rejection of
component partitions too small for their local decision-space KDE, and selectable
covariance stabilization. The historical repair replaces exactly zero eigenvalues with
``1e-15``. The relative floor is a separately identified numerical modernization.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, cast

import numpy as np

from projectkoios.numerics.validation import u32

from ....numerics.numpy.typing import Float64Array, Int64Array
from .kde import KdeBandwidthFactory, KdeBandwidthMethod


class GmmFeatureSpace(Enum):
    """Numerical observations used to assign Pareto points to GMM clusters."""

    DECISIONS = "decisions"
    OBJECTIVES = "objectives"
    DECISIONS_AND_OBJECTIVES = "decisions_and_objectives"


class GmmComponentCriterion(Enum):
    """Historical information criterion used to choose a component count."""

    AIC = "aic"
    BIC = "bic"


class GmmClusterAllocation(Enum):
    """Rule for allocating a requested batch across fitted clusters."""

    EQUAL = "equal"
    MIXTURE_WEIGHT = "mixture_weight"


class CovarianceStabilizationMethod(Enum):
    """Covariance repair applied before local Gaussian-kernel resampling."""

    HISTORICAL_ZERO_EIGENVALUE = "historical_zero_eigenvalue"
    RELATIVE_EIGENVALUE_FLOOR = "relative_eigenvalue_floor"


class _GaussianMixtureModel(Protocol):
    weights_: Float64Array

    def aic(self, values: Float64Array, /) -> float: ...

    def bic(self, values: Float64Array, /) -> float: ...

    def predict(self, values: Float64Array, /) -> Int64Array: ...


def historical_zero_eigenvalue_covariance_repair(
    covariance: Float64Array,
) -> Float64Array:
    """Apply the exact PyPosPack zero-eigenvalue covariance workaround."""
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    repaired = eigenvalues.copy()
    repaired[repaired == 0] = 1e-15
    result = np.linalg.multi_dot(
        (eigenvectors, np.diag(repaired), np.linalg.inv(eigenvectors))
    )
    return np.asarray(np.real_if_close(result), dtype=np.float64)


def relative_eigenvalue_covariance_repair(
    covariance: Float64Array,
    relative_floor: float,
) -> Float64Array:
    """Apply the modern symmetric scale-relative eigenspectrum floor."""
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).eps)
    repaired = np.maximum(eigenvalues, relative_floor * scale)
    result = (eigenvectors * repaired) @ eigenvectors.T
    return np.asarray((result + result.T) / 2.0, dtype=np.float64)


def _silverman_factor(dimension: int, sample_count: int) -> float:
    return float((sample_count * (dimension + 2.0) / 4.0) ** (-1.0 / (dimension + 4.0)))


class ScipyGmmConditionedKdeSampler:
    """Fit GMM clusters to a Pareto archive and sample cluster-local KDEs."""

    __slots__ = (
        "_allocation",
        "_bandwidth_method",
        "_component_count_history",
        "_component_criterion",
        "_covariance_stabilization",
        "_feature_space",
        "_max_components",
        "_relative_eigenvalue_floor",
        "_standardize_features",
    )

    def __init__(  # noqa: C901
        self,
        bandwidth_method: KdeBandwidthMethod = KdeBandwidthMethod.SILVERMAN_1986,
        *,
        feature_space: GmmFeatureSpace = GmmFeatureSpace.DECISIONS_AND_OBJECTIVES,
        component_criterion: GmmComponentCriterion = GmmComponentCriterion.BIC,
        max_components: int = 10,
        allocation: GmmClusterAllocation = GmmClusterAllocation.EQUAL,
        covariance_stabilization: CovarianceStabilizationMethod = (
            CovarianceStabilizationMethod.HISTORICAL_ZERO_EIGENVALUE
        ),
        relative_eigenvalue_floor: float = 1e-10,
        standardize_features: bool = False,
    ) -> None:
        if type(bandwidth_method) is not KdeBandwidthMethod:
            raise TypeError("bandwidth_method must be KdeBandwidthMethod")
        if type(feature_space) is not GmmFeatureSpace:
            raise TypeError("feature_space must be GmmFeatureSpace")
        if type(component_criterion) is not GmmComponentCriterion:
            raise TypeError("component_criterion must be GmmComponentCriterion")
        if type(max_components) is not int:
            raise TypeError("max_components must be a built-in integer")
        if max_components < 1:
            raise ValueError("max_components must be positive")
        if type(allocation) is not GmmClusterAllocation:
            raise TypeError("allocation must be GmmClusterAllocation")
        if type(covariance_stabilization) is not CovarianceStabilizationMethod:
            raise TypeError(
                "covariance_stabilization must be CovarianceStabilizationMethod"
            )
        if type(relative_eigenvalue_floor) is not float:
            raise TypeError("relative_eigenvalue_floor must be a built-in float")
        if not 0.0 < relative_eigenvalue_floor < 1.0:
            raise ValueError("relative_eigenvalue_floor must lie between zero and one")
        if type(standardize_features) is not bool:
            raise TypeError("standardize_features must be a built-in boolean")
        self._bandwidth_method = bandwidth_method
        self._component_count_history: list[int] = []
        self._feature_space = feature_space
        self._component_criterion = component_criterion
        self._max_components = max_components
        self._allocation = allocation
        self._covariance_stabilization = covariance_stabilization
        self._relative_eigenvalue_floor = relative_eigenvalue_floor
        self._standardize_features = standardize_features

    @property
    def component_count_history(self) -> tuple[int, ...]:
        """Return selected GMM component counts in adaptive-stage order."""
        return tuple(self._component_count_history)

    @property
    def covariance_stabilization(self) -> CovarianceStabilizationMethod:
        """Return the configured historical or modern repair rule."""
        return self._covariance_stabilization

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
        """Fit deterministic GMM labels and resample every cluster-local KDE."""
        del evaluated_decisions, evaluated_objectives
        count = u32(sample_count, "sample_count", allow_zero=False)
        seed = u32(random_seed, "random_seed", allow_zero=True)
        decisions, objectives = self._validate_archive(pareto_set, pareto_front)
        features = self._features(decisions, objectives)
        labels, weights = self._fit_labels(features, decisions.shape[1])
        cluster_ids = tuple(sorted(int(value) for value in np.unique(labels)))
        self._component_count_history.append(len(cluster_ids))
        allocations = self._allocate(count, cluster_ids, weights)
        random_state = np.random.RandomState(seed)
        samples: list[tuple[float, ...]] = []
        for cluster_id, cluster_count in zip(cluster_ids, allocations, strict=True):
            if not cluster_count:
                continue
            cluster = decisions[labels == cluster_id]
            samples.extend(self._sample_local_kde(cluster, cluster_count, random_state))
        if len(samples) != count:
            raise RuntimeError(
                "GMM-conditioned sampler produced the wrong sample count"
            )
        return tuple(samples)

    def _validate_archive(
        self,
        pareto_set: tuple[tuple[float, ...], ...],
        pareto_front: tuple[tuple[float, ...], ...],
    ) -> tuple[Float64Array, Float64Array]:
        if not pareto_set:
            raise ValueError("pareto_set must contain GMM/KDE seed decision vectors")
        if len(pareto_set) != len(pareto_front):
            raise ValueError("pareto_set and pareto_front must have equal lengths")
        decisions = np.asarray(pareto_set, dtype=np.float64)
        objectives = np.asarray(pareto_front, dtype=np.float64)
        if decisions.ndim != 2 or decisions.shape[1] == 0:
            raise ValueError("pareto_set must be a rectangular decision matrix")
        if objectives.ndim != 2 or objectives.shape[1] == 0:
            raise ValueError("pareto_front must be a rectangular objective matrix")
        if not np.all(np.isfinite(decisions)) or not np.all(np.isfinite(objectives)):
            raise ValueError("GMM/KDE seed observations must be finite")
        return decisions, objectives

    def _features(
        self,
        decisions: Float64Array,
        objectives: Float64Array,
    ) -> Float64Array:
        if self._feature_space is GmmFeatureSpace.DECISIONS:
            features = decisions.copy()
        elif self._feature_space is GmmFeatureSpace.OBJECTIVES:
            features = objectives.copy()
        else:
            features = np.concatenate((decisions, objectives), axis=1)
        if self._standardize_features:
            means = np.mean(features, axis=0)
            scales = np.std(features, axis=0)
            scales[scales == 0.0] = 1.0
            features = (features - means) / scales
        return np.asarray(features, dtype=np.float64)

    def _fit_labels(
        self,
        features: Float64Array,
        decision_dimension: int,
    ) -> tuple[Int64Array, Float64Array]:
        from sklearn.mixture import GaussianMixture

        max_valid_components = max(1, len(features) // (decision_dimension + 1))
        max_components = min(self._max_components, len(features), max_valid_components)
        candidates: list[tuple[float, _GaussianMixtureModel, Int64Array]] = []
        for component_count in range(1, max_components + 1):
            model = cast(
                _GaussianMixtureModel,
                GaussianMixture(
                    n_components=component_count,
                    covariance_type="full",
                    random_state=0,
                ).fit(features),
            )
            labels = np.asarray(model.predict(features), dtype=np.int64)
            cluster_sizes = np.bincount(labels, minlength=component_count)
            if np.any(cluster_sizes <= decision_dimension):
                continue
            criterion = (
                model.aic(features)
                if self._component_criterion is GmmComponentCriterion.AIC
                else model.bic(features)
            )
            candidates.append((float(criterion), model, labels))
        if not candidates:
            model = cast(
                _GaussianMixtureModel,
                GaussianMixture(
                    n_components=1,
                    covariance_type="full",
                    random_state=0,
                ).fit(features),
            )
            labels = np.asarray(model.predict(features), dtype=np.int64)
            return labels, np.asarray(model.weights_, dtype=np.float64)
        _, selected, labels = min(candidates, key=lambda item: item[0])
        return labels, np.asarray(selected.weights_, dtype=np.float64)

    def _allocate(
        self,
        sample_count: int,
        cluster_ids: tuple[int, ...],
        weights: Float64Array,
    ) -> tuple[int, ...]:
        if self._allocation is GmmClusterAllocation.EQUAL:
            quotient, remainder = divmod(sample_count, len(cluster_ids))
            return tuple(
                quotient + (index < remainder) for index in range(len(cluster_ids))
            )
        selected_weights = np.asarray(
            [weights[cluster_id] for cluster_id in cluster_ids], dtype=np.float64
        )
        selected_weights /= np.sum(selected_weights)
        exact = selected_weights * sample_count
        allocations = np.floor(exact).astype(np.int64)
        remainder = sample_count - int(np.sum(allocations))
        order = sorted(
            range(len(cluster_ids)),
            key=lambda index: (
                -(exact[index] - allocations[index]),
                cluster_ids[index],
            ),
        )
        for index in order[:remainder]:
            allocations[index] += 1
        return tuple(int(value) for value in allocations)

    def _sample_local_kde(
        self,
        cluster: Float64Array,
        sample_count: int,
        random_state: np.random.RandomState,
    ) -> list[tuple[float, ...]]:
        means = np.mean(cluster, axis=0)
        scales = np.std(cluster, axis=0)
        scales[scales == 0.0] = 1.0
        standardized = (cluster - means) / scales
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False, bias=False))
        factor = self._bandwidth_factor(standardized)
        kernel_covariance = np.asarray(covariance * factor**2, dtype=np.float64)
        if (
            self._covariance_stabilization
            is CovarianceStabilizationMethod.HISTORICAL_ZERO_EIGENVALUE
        ):
            kernel_covariance = historical_zero_eigenvalue_covariance_repair(
                kernel_covariance
            )
        else:
            kernel_covariance = relative_eigenvalue_covariance_repair(
                kernel_covariance, self._relative_eigenvalue_floor
            )
        np.linalg.cholesky((kernel_covariance + kernel_covariance.T) / 2.0)

        samples: list[tuple[float, ...]] = []
        for _ in range(sample_count):
            noise = random_state.multivariate_normal(
                np.zeros(cluster.shape[1]), kernel_covariance
            )
            source_index = int(random_state.choice(cluster.shape[0]))
            value = (standardized[source_index] + noise) * scales + means
            samples.append(tuple(float(item) for item in value))
        return samples

    def _bandwidth_factor(self, standardized: Float64Array) -> float:
        if self._bandwidth_method is KdeBandwidthMethod.SILVERMAN_1986:
            return _silverman_factor(standardized.shape[1], standardized.shape[0])
        function = KdeBandwidthFactory.create(self._bandwidth_method)
        return function(standardized.T)


__all__ = [
    "CovarianceStabilizationMethod",
    "GmmClusterAllocation",
    "GmmComponentCriterion",
    "GmmFeatureSpace",
    "ScipyGmmConditionedKdeSampler",
    "historical_zero_eigenvalue_covariance_repair",
    "relative_eigenvalue_covariance_repair",
]
