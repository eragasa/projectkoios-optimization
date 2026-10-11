"""Optional SciPy adapters for optimization contracts."""

from .probability import (
    ScipyInverseProbabilityDistributionFunction,
    ScipyProbabilityDistributionFunction,
    ScipyProbabilityDistributionSource,
)
from .quadratic import ScipySparsePositiveDefiniteQuadraticLoss

__all__ = [
    "ScipyInverseProbabilityDistributionFunction",
    "ScipySparsePositiveDefiniteQuadraticLoss",
    "ScipyProbabilityDistributionFunction",
    "ScipyProbabilityDistributionSource",
]
