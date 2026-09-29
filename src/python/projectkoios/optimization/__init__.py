"""Optimization contracts and deterministic algorithms for Project Koios."""

from .checkpoint import OptimizerCheckpoint
from .exceptions import (
    CheckpointError,
    DuplicateObservationError,
    OptimizationError,
    OptimizerExhaustedError,
    UnknownCandidateError,
)
from .identity import OptimizerDescriptor, OptimizerIdentity
from .loss import WeightedLoss, WeightedLossTerm, WeightedSquaredLoss, raw_evaluation
from .pattern import BoundedCoordinatePatternOptimizer, PatternSearchSettings
from .ports import (
    ForwardEvaluator,
    OptimizationEngine,
    ResultsHandler,
    WeightedResultsHandler,
)
from .probability import (
    FunctionalProbabilityDistribution,
    InverseProbabilityDistributionFunction,
    ProbabilityDistribution,
    ProbabilityDistributionFunction,
    ProbabilityDistributionSampler,
    ProbabilityError,
    SplitMix64UnitUniformSampler,
    UniformInverseProbabilityDistributionFunction,
    UniformProbabilityDistribution,
    UniformProbabilityDistributionFunction,
    UniformProbabilityDistributionParameters,
    UnitUniformSampler,
)
from .records import (
    Candidate,
    Measurement,
    ObjectiveFeedback,
    ObjectiveValue,
    OptimizerUpdate,
    ParameterBound,
    ParameterSpace,
    RawEvaluation,
)
from .reports import OptimizationReport, ScoredResult, build_report
from .sampling import Sampler, SamplingError
from .uniform import BoundedUniformOptimizer

__all__ = [
    "BoundedCoordinatePatternOptimizer",
    "BoundedUniformOptimizer",
    "Candidate",
    "CheckpointError",
    "DuplicateObservationError",
    "ForwardEvaluator",
    "FunctionalProbabilityDistribution",
    "InverseProbabilityDistributionFunction",
    "Measurement",
    "ObjectiveFeedback",
    "ObjectiveValue",
    "OptimizationEngine",
    "OptimizationError",
    "OptimizationReport",
    "OptimizerCheckpoint",
    "OptimizerDescriptor",
    "OptimizerExhaustedError",
    "OptimizerIdentity",
    "OptimizerUpdate",
    "ParameterBound",
    "ParameterSpace",
    "PatternSearchSettings",
    "ProbabilityDistribution",
    "ProbabilityDistributionFunction",
    "ProbabilityDistributionSampler",
    "ProbabilityError",
    "RawEvaluation",
    "ResultsHandler",
    "Sampler",
    "SamplingError",
    "ScoredResult",
    "SplitMix64UnitUniformSampler",
    "UniformInverseProbabilityDistributionFunction",
    "UniformProbabilityDistribution",
    "UniformProbabilityDistributionFunction",
    "UniformProbabilityDistributionParameters",
    "UnitUniformSampler",
    "UnknownCandidateError",
    "WeightedLoss",
    "WeightedLossTerm",
    "WeightedResultsHandler",
    "WeightedSquaredLoss",
    "build_report",
    "raw_evaluation",
]
