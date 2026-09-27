"""Closed optimization contract errors."""


class OptimizationError(Exception):
    """Base error for an optimization contract violation."""


class OptimizerExhaustedError(OptimizationError):
    """Raised when no further candidate can be proposed."""


class UnknownCandidateError(OptimizationError):
    """Raised when feedback does not name an outstanding candidate."""


class DuplicateObservationError(OptimizationError):
    """Raised when feedback repeats an already accepted candidate."""


class CheckpointError(OptimizationError):
    """Raised when checkpoint syntax, integrity, or compatibility fails."""
