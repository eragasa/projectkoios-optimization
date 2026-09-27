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
from .uniform import BoundedUniformOptimizer

__all__ = [
    "BoundedCoordinatePatternOptimizer",
    "BoundedUniformOptimizer",
    "Candidate",
    "CheckpointError",
    "DuplicateObservationError",
    "ForwardEvaluator",
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
    "RawEvaluation",
    "ResultsHandler",
    "ScoredResult",
    "UnknownCandidateError",
    "WeightedLoss",
    "WeightedLossTerm",
    "WeightedResultsHandler",
    "WeightedSquaredLoss",
    "build_report",
    "raw_evaluation",
]
