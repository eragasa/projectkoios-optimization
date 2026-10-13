"""Closed optimization contract errors."""


class OptimizationError(Exception):
    """Base error for an optimization contract violation."""


class OptimizerExhaustedError(OptimizationError):
    """Raised when no further candidate can be proposed."""


class UnknownCandidateError(OptimizationError):
    """Raised when feedback does not name an outstanding candidate."""


class DuplicateObservationError(OptimizationError):
    """Raised when feedback repeats an already accepted candidate."""


class OptimizerStateError(OptimizationError):
    """Raised when typed records do not describe reachable optimizer state."""
