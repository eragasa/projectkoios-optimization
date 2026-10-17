"""Explicit construction of probability-driven samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sampling.uniform import UniformRandomSampler
from ..spaces.parameters import ParameterSpace
from .base import ProbabilityDistribution
from .sampling import (
    ProbabilityDistributionSampler,
    SplitMix64UnitUniformSampler,
    UnitUniformSampler,
)

if TYPE_CHECKING:
    from ..sampling.scipy.gmm_kde import (
        CovarianceStabilizationMethod,
        GmmClusterAllocation,
        GmmComponentCriterion,
        GmmFeatureSpace,
        ScipyGmmConditionedKdeSampler,
    )
    from ..sampling.scipy.kde import KdeBandwidthMethod, ScipyGaussianKdeSampler
    from ..sampling.scipy.pareto_shells import (
        ScipyParetoShellContrastiveKdeSampler,
    )


class ProbabilitySamplerFactory:
    """Construct maintained samplers through concrete named methods."""

    __slots__ = ()

    def create_uniform_random(
        self,
        space: ParameterSpace,
    ) -> UniformRandomSampler:
        """Construct SplitMix64 uniform random sampling over a parameter space."""
        return UniformRandomSampler(space)

    def create_gaussian_kde(
        self,
        bandwidth_method: KdeBandwidthMethod,
    ) -> ScipyGaussianKdeSampler:
        """Construct optional SciPy KDE sampling with enum-selected bandwidth."""
        from ..sampling.scipy.kde import (  # noqa: PLC0415
            KdeBandwidthFactory,
            ScipyGaussianKdeSampler,
        )

        KdeBandwidthFactory.create(bandwidth_method)
        return ScipyGaussianKdeSampler(bandwidth_method)

    def create_gmm_conditioned_kde(
        self,
        bandwidth_method: KdeBandwidthMethod,
        *,
        feature_space: GmmFeatureSpace,
        component_criterion: GmmComponentCriterion,
        max_components: int,
        allocation: GmmClusterAllocation,
        covariance_stabilization: CovarianceStabilizationMethod,
        relative_eigenvalue_floor: float = 1e-10,
        standardize_features: bool = False,
    ) -> ScipyGmmConditionedKdeSampler:
        """Construct the optional PyPosPack-lineage GMM-conditioned KDE sampler."""
        from ..sampling.scipy.gmm_kde import (  # noqa: PLC0415
            ScipyGmmConditionedKdeSampler,
        )

        return ScipyGmmConditionedKdeSampler(
            bandwidth_method,
            feature_space=feature_space,
            component_criterion=component_criterion,
            max_components=max_components,
            allocation=allocation,
            covariance_stabilization=covariance_stabilization,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            standardize_features=standardize_features,
        )

    def create_pareto_shell_contrastive_kde(
        self,
        bandwidth_method: KdeBandwidthMethod,
        *,
        shell_weights: tuple[float, ...] = (1.0, 0.5),
        candidate_multiplier: int = 8,
        chunk_size: int = 500,
    ) -> ScipyParetoShellContrastiveKdeSampler:
        """Construct optional Pareto-shell contrastive SciPy KDE sampling."""
        from ..sampling.scipy.kde import KdeBandwidthFactory  # noqa: PLC0415
        from ..sampling.scipy.pareto_shells import (  # noqa: PLC0415
            ScipyParetoShellContrastiveKdeSampler,
        )

        KdeBandwidthFactory.create(bandwidth_method)
        return ScipyParetoShellContrastiveKdeSampler(
            bandwidth_method,
            shell_weights=shell_weights,
            candidate_multiplier=candidate_multiplier,
            chunk_size=chunk_size,
        )

    def create_distribution_sampler(
        self,
        distribution: ProbabilityDistribution,
        source: UnitUniformSampler,
    ) -> ProbabilityDistributionSampler:
        """Construct inverse-CDF sampling from an explicit unit-uniform source."""
        if not isinstance(distribution, ProbabilityDistribution):
            raise TypeError("distribution must be ProbabilityDistribution")
        return ProbabilityDistributionSampler(
            source,
            distribution.inverse_distribution_function,
        )

    def create_splitmix64_distribution_sampler(
        self,
        distribution: ProbabilityDistribution,
    ) -> ProbabilityDistributionSampler:
        """Construct inverse-CDF sampling driven by deterministic SplitMix64."""
        return self.create_distribution_sampler(
            distribution,
            SplitMix64UnitUniformSampler(),
        )


__all__ = ["ProbabilitySamplerFactory"]
