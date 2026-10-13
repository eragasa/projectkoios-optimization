"""Dependency-free contracts and records for Gaussian-mixture clustering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..validation.validation import finite_float, integer, text, u32


@dataclass(frozen=True, slots=True)
class GaussianMixtureClusteringRequest:
    """Bounded deterministic full-covariance model-selection request."""

    maximum_components: int
    random_seed: int

    def __post_init__(self) -> None:
        maximum = integer(self.maximum_components, "maximum_components")
        if not 1 <= maximum <= 64:
            raise ValueError("maximum_components must be between one and 64")
        object.__setattr__(
            self,
            "random_seed",
            u32(self.random_seed, "random_seed", allow_zero=True),
        )


@dataclass(frozen=True, slots=True)
class GaussianMixtureModelScore:
    """Finite AIC and BIC diagnostics for one component count."""

    component_count: int
    akaike_information_criterion: float
    bayesian_information_criterion: float

    def __post_init__(self) -> None:
        count = integer(self.component_count, "component_count")
        if count < 1:
            raise ValueError("component_count must be positive")
        object.__setattr__(
            self,
            "akaike_information_criterion",
            finite_float(
                self.akaike_information_criterion,
                "akaike_information_criterion",
            ),
        )
        object.__setattr__(
            self,
            "bayesian_information_criterion",
            finite_float(
                self.bayesian_information_criterion,
                "bayesian_information_criterion",
            ),
        )


@dataclass(frozen=True, slots=True)
class GaussianMixtureAssignment:
    """Canonical cluster assignment for one input observation row."""

    observation_index: int
    cluster_id: int
    probability: float

    def __post_init__(self) -> None:
        index = integer(self.observation_index, "observation_index")
        if index < 0:
            raise ValueError("observation_index must be nonnegative")
        cluster_id = integer(self.cluster_id, "cluster_id")
        if cluster_id < 0:
            raise ValueError("cluster_id must be nonnegative")
        probability = finite_float(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie between zero and one")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True)
class GaussianMixtureComponent:
    """One canonical full-covariance component in standardized feature space."""

    cluster_id: int
    weight: float
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        cluster_id = integer(self.cluster_id, "cluster_id")
        if cluster_id < 0:
            raise ValueError("cluster_id must be nonnegative")
        weight = finite_float(self.weight, "weight")
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must lie in (0, 1]")
        object.__setattr__(self, "weight", weight)
        mean = _finite_vector(self.mean, "mean")
        if not mean:
            raise ValueError("mean must be nonempty")
        object.__setattr__(self, "mean", mean)
        if type(self.covariance) is not tuple:
            raise TypeError("covariance must be a tuple of tuple rows")
        covariance = tuple(
            _finite_vector(row, "covariance row") for row in self.covariance
        )
        if len(covariance) != len(mean) or any(
            len(row) != len(mean) for row in covariance
        ):
            raise ValueError("covariance must be square and match the mean dimension")
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class GaussianMixtureClusteringCompleted:
    """BIC-selected Gaussian-mixture result with exact row assignments."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    model_scores: tuple[GaussianMixtureModelScore, ...]
    selected_components: int
    assignments: tuple[GaussianMixtureAssignment, ...]
    components: tuple[GaussianMixtureComponent, ...]
    converged: bool
    iterations: int
    backend_version: str

    def __post_init__(self) -> None:
        means = _finite_vector(self.feature_means, "feature_means")
        scales = _finite_vector(self.feature_scales, "feature_scales")
        if not means:
            raise ValueError("feature_means must be nonempty")
        if len(scales) != len(means):
            raise ValueError("feature statistics must have equal dimensions")
        if any(value <= 0.0 for value in scales):
            raise ValueError("feature_scales must be positive")
        object.__setattr__(self, "feature_means", means)
        object.__setattr__(self, "feature_scales", scales)

        if (
            type(self.model_scores) is not tuple
            or not self.model_scores
            or any(
                type(value) is not GaussianMixtureModelScore
                for value in self.model_scores
            )
        ):
            raise TypeError(
                "model_scores must be a nonempty tuple of GaussianMixtureModelScore"
            )
        expected_counts = tuple(range(1, len(self.model_scores) + 1))
        observed_counts = tuple(value.component_count for value in self.model_scores)
        if observed_counts != expected_counts:
            raise ValueError("model_scores must cover consecutive component counts")

        selected = integer(self.selected_components, "selected_components")
        if selected not in expected_counts:
            raise ValueError("selected_components must identify a scored model")
        if type(self.components) is not tuple or any(
            type(value) is not GaussianMixtureComponent for value in self.components
        ):
            raise TypeError("components must be a tuple of GaussianMixtureComponent")
        if len(self.components) != selected:
            raise ValueError("component count must equal selected_components")
        component_ids = tuple(value.cluster_id for value in self.components)
        if component_ids != tuple(range(selected)):
            raise ValueError("component cluster identities must be canonical")
        if any(len(value.mean) != len(means) for value in self.components):
            raise ValueError("component dimensions must match feature statistics")

        if (
            type(self.assignments) is not tuple
            or not self.assignments
            or any(
                type(value) is not GaussianMixtureAssignment
                for value in self.assignments
            )
        ):
            raise TypeError(
                "assignments must be a nonempty tuple of GaussianMixtureAssignment"
            )
        if tuple(value.observation_index for value in self.assignments) != tuple(
            range(len(self.assignments))
        ):
            raise ValueError("assignments must preserve every input row in order")
        if any(value.cluster_id >= selected for value in self.assignments):
            raise ValueError("assignment references an unavailable cluster")

        if type(self.converged) is not bool:
            raise TypeError("converged must be a built-in bool")
        iterations = integer(self.iterations, "iterations")
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        object.__setattr__(
            self,
            "backend_version",
            text(self.backend_version, "backend_version"),
        )


class GaussianMixtureClusteringFailureCode(Enum):
    """Closed numerical clustering failure classification."""

    NO_OBSERVATIONS = "no_observations"
    FIT_FAILED = "fit_failed"


@dataclass(frozen=True, slots=True)
class GaussianMixtureClusteringFailed:
    """Closed inability to produce a Gaussian-mixture clustering result."""

    code: GaussianMixtureClusteringFailureCode
    message: str

    def __post_init__(self) -> None:
        if type(self.code) is not GaussianMixtureClusteringFailureCode:
            raise TypeError("code must be GaussianMixtureClusteringFailureCode")
        object.__setattr__(self, "message", text(self.message, "message"))


type GaussianMixtureClusteringResult = (
    GaussianMixtureClusteringCompleted | GaussianMixtureClusteringFailed
)


class GaussianMixtureClusterer(Protocol):
    """Structural boundary for clustering an ordered finite numeric matrix."""

    def cluster(
        self,
        observations: tuple[tuple[float, ...], ...],
        request: GaussianMixtureClusteringRequest,
        /,
    ) -> GaussianMixtureClusteringResult:
        """Cluster observation rows without changing or compressing them."""
        ...


def _finite_vector(values: object, name: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    return tuple(finite_float(value, f"{name} value") for value in values)


__all__ = [
    "GaussianMixtureAssignment",
    "GaussianMixtureClusterer",
    "GaussianMixtureClusteringCompleted",
    "GaussianMixtureClusteringFailed",
    "GaussianMixtureClusteringFailureCode",
    "GaussianMixtureClusteringRequest",
    "GaussianMixtureClusteringResult",
    "GaussianMixtureComponent",
    "GaussianMixtureModelScore",
]
