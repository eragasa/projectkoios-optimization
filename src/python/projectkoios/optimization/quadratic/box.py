"""Dependency-free contracts for dense positive-quadratic box minimization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, SupportsFloat, final

from projectkoios.numerics.validation import (
    exact_nonnegative_float,
    finite_float,
    integer,
    u32,
    validate_finite_float_tuple,
)

from ..spaces.parameters import ParameterSpace
from ..validation.validation import text
from .constrained import (
    DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE,
    ConstrainedQuadraticTolerance,
)
from .quadratic import PositiveDefiniteQuadraticLoss
from .validation import finite_vector

DEFAULT_BOX_QUADRATIC_MAXIMUM_ITERATIONS = 10_000


class BoxQuadraticBoundStatus(StrEnum):
    """Exact bound classification for one returned minimizing coordinate."""

    FREE = "free"
    LOWER = "lower"
    UPPER = "upper"
    FIXED = "fixed"


def _nonnegative_int(value: int, name: str, /) -> int:
    result = integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: int, name: str, /) -> int:
    result = integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _kkt_violations(
    gradient: tuple[float, ...],
    statuses: tuple[BoxQuadraticBoundStatus, ...],
    /,
) -> tuple[float, ...]:
    violations: list[float] = []
    for value, status in zip(gradient, statuses, strict=True):
        if status is BoxQuadraticBoundStatus.FREE:
            violation = abs(value)
        elif status is BoxQuadraticBoundStatus.LOWER:
            violation = max(0.0, -value)
        elif status is BoxQuadraticBoundStatus.UPPER:
            violation = max(0.0, value)
        else:
            violation = 0.0
        violations.append(violation)
    return tuple(violations)


@final
@dataclass(frozen=True, slots=True)
class BoxQuadraticMinimum:
    """Global minimum and independently reconstructed box-optimality evidence."""

    coordinates: tuple[float, ...]
    squared_loss: float
    bound_statuses: tuple[BoxQuadraticBoundStatus, ...]
    gradient: tuple[float, ...]
    kkt_tolerance_bounds: tuple[float, ...]
    maximum_kkt_violation: float
    maximum_kkt_ratio: float
    iteration_count: int
    active_set_change_count: int
    linear_solve_count: int
    maximum_iterations: int
    backend_identifier: str
    backend_version: str

    @classmethod
    def from_values(
        cls,
        *,
        coordinates: tuple[SupportsFloat, ...],
        squared_loss: SupportsFloat,
        bound_statuses: tuple[BoxQuadraticBoundStatus, ...],
        gradient: tuple[SupportsFloat, ...],
        kkt_tolerance_bounds: tuple[SupportsFloat, ...],
        maximum_kkt_violation: SupportsFloat,
        maximum_kkt_ratio: SupportsFloat,
        iteration_count: int,
        active_set_change_count: int,
        linear_solve_count: int,
        maximum_iterations: int,
        backend_identifier: str,
        backend_version: str,
    ) -> Self:
        """Construct canonical immutable evidence from finite real values.

        Parameters
        ----------
        coordinates
            Candidate finite minimizing vector.
        squared_loss
            Candidate nonnegative loss.
        bound_statuses
            Exact bound classifications.
        gradient
            Candidate finite full gradient.
        kkt_tolerance_bounds
            Candidate positive row-local bounds.
        maximum_kkt_violation, maximum_kkt_ratio
            Candidate nonnegative aggregate KKT evidence.
        iteration_count, active_set_change_count, linear_solve_count
            Exact candidate execution counts.
        maximum_iterations
            Exact positive iteration limit.
        backend_identifier, backend_version
            Exact nonempty backend identity strings.

        Returns
        -------
        BoxQuadraticMinimum
            Frozen record containing canonical built-in values from initial
            construction.

        Raises
        ------
        TypeError
            If an input has the wrong nominal or exact runtime type.
        ValueError
            If normalization fails or retained evidence is relationally inconsistent.
        """
        return cls(
            coordinates=finite_vector(coordinates, "coordinates"),
            squared_loss=finite_float(squared_loss, "squared_loss"),
            bound_statuses=bound_statuses,
            gradient=finite_vector(gradient, "gradient"),
            kkt_tolerance_bounds=finite_vector(
                kkt_tolerance_bounds,
                "kkt_tolerance_bounds",
            ),
            maximum_kkt_violation=finite_float(
                maximum_kkt_violation,
                "maximum_kkt_violation",
            ),
            maximum_kkt_ratio=finite_float(
                maximum_kkt_ratio,
                "maximum_kkt_ratio",
            ),
            iteration_count=iteration_count,
            active_set_change_count=active_set_change_count,
            linear_solve_count=linear_solve_count,
            maximum_iterations=maximum_iterations,
            backend_identifier=backend_identifier,
            backend_version=backend_version,
        )

    def __post_init__(self) -> None:
        vectors = _validate_box_minimum_vectors(self)
        _validate_box_minimum_kkt(self, vectors)
        _validate_box_minimum_execution(self)


@dataclass(frozen=True, slots=True)
class _BoxMinimumVectors:
    """Hold validated canonical box-minimum vectors and scalar loss.

    Parameters
    ----------
    coordinates
        Finite minimizing coordinates.
    squared_loss
        Finite nonnegative loss at the coordinates.
    gradient
        Finite full gradient.
    tolerance_bounds
        Positive row-local KKT bounds.
    """

    coordinates: tuple[float, ...]
    squared_loss: float
    gradient: tuple[float, ...]
    tolerance_bounds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _BoxMinimumKktEvidence:
    """Hold validated aggregate KKT evidence.

    Parameters
    ----------
    maximum_violation
        Maximum per-coordinate KKT violation.
    maximum_ratio
        Maximum per-coordinate violation-to-bound ratio.
    """

    maximum_violation: float
    maximum_ratio: float


@dataclass(frozen=True, slots=True)
class _BoxMinimumExecutionEvidence:
    """Hold validated execution counts and backend identity.

    Parameters
    ----------
    iterations
        Positive number of inspected active-set states.
    changes
        Nonnegative number of active-set changes.
    solves
        Nonnegative number of free-face linear solves.
    maximum_iterations
        Positive applied iteration bound.
    identifier, version
        Nonempty trimmed backend identity strings.
    """

    iterations: int
    changes: int
    solves: int
    maximum_iterations: int
    identifier: str
    version: str


def _validate_box_minimum_vectors(
    result: BoxQuadraticMinimum,
    /,
) -> _BoxMinimumVectors:
    """Validate canonical vector-shaped minimum evidence.

    Parameters
    ----------
    result
        Box minimum under construction.

    Returns
    -------
    _BoxMinimumVectors
        Validated canonical coordinates, loss, gradient, and tolerance bounds.

    Raises
    ------
    TypeError
        If a vector, entry, or bound-status container has the wrong nominal type.
    ValueError
        If a value is nonfinite, empty, negative, nonpositive, or dimensionally
        inconsistent.
    """
    coordinates = validate_finite_float_tuple(result.coordinates, "coordinates")
    squared_loss = exact_nonnegative_float(result.squared_loss, "squared_loss")
    if type(result.bound_statuses) is not tuple or any(
        type(value) is not BoxQuadraticBoundStatus for value in result.bound_statuses
    ):
        raise TypeError("bound_statuses must be a tuple of BoxQuadraticBoundStatus")
    gradient = validate_finite_float_tuple(result.gradient, "gradient")
    tolerance_bounds = validate_finite_float_tuple(
        result.kkt_tolerance_bounds,
        "kkt_tolerance_bounds",
    )
    dimension = len(coordinates)
    if len(result.bound_statuses) != dimension:
        raise ValueError("bound_statuses must match the coordinate dimension")
    if len(gradient) != dimension:
        raise ValueError("gradient must match the coordinate dimension")
    if len(tolerance_bounds) != dimension:
        raise ValueError("kkt_tolerance_bounds must match the coordinate dimension")
    if any(value <= 0.0 for value in tolerance_bounds):
        raise ValueError("kkt_tolerance_bounds must be positive")
    return _BoxMinimumVectors(
        coordinates=coordinates,
        squared_loss=squared_loss,
        gradient=gradient,
        tolerance_bounds=tolerance_bounds,
    )


def _validate_box_minimum_kkt(
    result: BoxQuadraticMinimum,
    vectors: _BoxMinimumVectors,
    /,
) -> _BoxMinimumKktEvidence:
    """Validate aggregate KKT evidence against coordinate-local values.

    Parameters
    ----------
    result
        Structurally validated box minimum.
    vectors
        Normalized vector evidence.

    Returns
    -------
    _BoxMinimumKktEvidence
        Canonical and relationally validated aggregate evidence.

    Raises
    ------
    TypeError
        If an aggregate has the wrong nominal numeric type.
    ValueError
        If an aggregate is negative, inconsistent, or exceeds acceptance.
    """
    maximum_violation = exact_nonnegative_float(
        result.maximum_kkt_violation,
        "maximum_kkt_violation",
    )
    maximum_ratio = exact_nonnegative_float(
        result.maximum_kkt_ratio,
        "maximum_kkt_ratio",
    )
    violations = _kkt_violations(vectors.gradient, result.bound_statuses)
    expected_maximum = max(violations)
    expected_ratio = max(
        violation / bound
        for violation, bound in zip(
            violations,
            vectors.tolerance_bounds,
            strict=True,
        )
    )
    if maximum_violation != expected_maximum:
        raise ValueError("maximum_kkt_violation is inconsistent")
    if maximum_ratio != expected_ratio:
        raise ValueError("maximum_kkt_ratio is inconsistent")
    if maximum_ratio > 1.0:
        raise ValueError("maximum_kkt_ratio must not exceed one")
    return _BoxMinimumKktEvidence(maximum_violation, maximum_ratio)


def _validate_box_minimum_execution(
    result: BoxQuadraticMinimum,
    /,
) -> _BoxMinimumExecutionEvidence:
    """Validate execution counts and declared backend identity.

    Parameters
    ----------
    result
        Structurally and numerically validated box minimum.

    Returns
    -------
    _BoxMinimumExecutionEvidence
        Validated counts and backend strings.

    Raises
    ------
    TypeError
        If a count or backend field has the wrong nominal type.
    ValueError
        If counts are invalid or inconsistent, or backend text is empty or untrimmed.
    """
    iterations = _positive_int(result.iteration_count, "iteration_count")
    changes = _nonnegative_int(
        result.active_set_change_count,
        "active_set_change_count",
    )
    solves = _nonnegative_int(result.linear_solve_count, "linear_solve_count")
    maximum_iterations = u32(
        result.maximum_iterations,
        "maximum_iterations",
        allow_zero=False,
    )
    if iterations > maximum_iterations:
        raise ValueError("iteration_count must not exceed maximum_iterations")
    if solves > iterations:
        raise ValueError("linear_solve_count must not exceed iteration_count")
    identifier = text(result.backend_identifier, "backend_identifier")
    version = text(result.backend_version, "backend_version")
    return _BoxMinimumExecutionEvidence(
        iterations=iterations,
        changes=changes,
        solves=solves,
        maximum_iterations=maximum_iterations,
        identifier=identifier,
        version=version,
    )


@final
@dataclass(frozen=True, slots=True)
class _BoxQuadraticBackendResult:
    """Private candidate handoff from one dense backend to the final template."""

    coordinates: tuple[float, ...]
    iteration_count: int
    active_set_change_count: int
    linear_solve_count: int

    def __post_init__(self) -> None:
        validate_finite_float_tuple(self.coordinates, "coordinates")
        iterations = _positive_int(self.iteration_count, "iteration_count")
        _nonnegative_int(
            self.active_set_change_count,
            "active_set_change_count",
        )
        solves = _nonnegative_int(self.linear_solve_count, "linear_solve_count")
        if solves > iterations:
            raise ValueError("linear_solve_count must not exceed iteration_count")


class BoxQuadraticMinimizer(ABC):
    """Nominal dense-only boundary for finite box quadratic minimization."""

    __slots__ = ("_maximum_iterations", "_tolerance")

    def __init__(
        self,
        *,
        tolerance: ConstrainedQuadraticTolerance = (
            DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE
        ),
        maximum_iterations: int = DEFAULT_BOX_QUADRATIC_MAXIMUM_ITERATIONS,
    ) -> None:
        if type(tolerance) is not ConstrainedQuadraticTolerance:
            raise TypeError("tolerance must be ConstrainedQuadraticTolerance")
        self._tolerance = tolerance
        self._maximum_iterations = u32(
            maximum_iterations,
            "maximum_iterations",
            allow_zero=False,
        )

    @property
    def tolerance(self) -> ConstrainedQuadraticTolerance:
        """Return the capability-specific numerical acceptance tolerance."""
        return self._tolerance

    @property
    def maximum_iterations(self) -> int:
        """Return the positive deterministic active-set iteration limit."""
        return self._maximum_iterations

    @property
    @abstractmethod
    def backend_identifier(self) -> str:
        """Return the trusted API-level backend identifier declaration."""
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_version(self) -> str:
        """Return the trusted API-level backend version declaration."""
        raise NotImplementedError

    @final
    def solve(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> BoxQuadraticMinimum:
        """Return the deterministic global minimum over a finite closed box."""
        if type(loss) is not PositiveDefiniteQuadraticLoss:
            raise TypeError("loss must be PositiveDefiniteQuadraticLoss")
        if type(bounds) is not ParameterSpace:
            raise TypeError("bounds must be ParameterSpace")
        if bounds.dimension == 0:
            raise ValueError("bounds must have positive dimension")
        if bounds.dimension != loss.dimension:
            raise ValueError("bounds and loss dimensions must match")
        if loss.sparse:
            raise ValueError("box minimization requires a dense quadratic loss")
        matrix_object = loss.matrix
        if type(matrix_object) is not tuple:
            raise TypeError("dense quadratic matrix must be a tuple of tuples")

        backend_identifier = text(self.backend_identifier, "backend_identifier")
        backend_version = text(self.backend_version, "backend_version")
        backend_result = self._solve_validated(loss, bounds)
        if type(backend_result) is not _BoxQuadraticBackendResult:
            raise TypeError("backend must return _BoxQuadraticBackendResult")
        if backend_result.iteration_count > self.maximum_iterations:
            raise ValueError("backend exceeded maximum_iterations")
        if len(backend_result.coordinates) != loss.dimension:
            raise ValueError("backend coordinates have the wrong dimension")
        if not bounds.admits(backend_result.coordinates):
            raise ValueError("backend minimum must lie inside bounds")

        squared_loss = _evaluate_loss(loss, backend_result.coordinates)
        statuses = _classify_bounds(backend_result.coordinates, bounds)
        gradient, row_scales = _gradient_and_row_scales(
            matrix_object,
            loss.center,
            backend_result.coordinates,
        )
        violations = _kkt_violations(gradient, statuses)
        tolerance_bounds = tuple(
            _kkt_tolerance_bound(self.tolerance, scale) for scale in row_scales
        )
        if any(
            violation > bound
            for violation, bound in zip(
                violations,
                tolerance_bounds,
                strict=True,
            )
        ):
            raise ValueError("backend minimum does not satisfy box KKT conditions")
        maximum_violation = max(violations)
        maximum_ratio = max(
            violation / bound
            for violation, bound in zip(
                violations,
                tolerance_bounds,
                strict=True,
            )
        )
        if not math.isfinite(maximum_ratio):
            raise FloatingPointError("KKT ratio must be finite")

        return BoxQuadraticMinimum.from_values(
            coordinates=backend_result.coordinates,
            squared_loss=squared_loss,
            bound_statuses=statuses,
            gradient=gradient,
            kkt_tolerance_bounds=tolerance_bounds,
            maximum_kkt_violation=maximum_violation,
            maximum_kkt_ratio=maximum_ratio,
            iteration_count=backend_result.iteration_count,
            active_set_change_count=backend_result.active_set_change_count,
            linear_solve_count=backend_result.linear_solve_count,
            maximum_iterations=self.maximum_iterations,
            backend_identifier=backend_identifier,
            backend_version=backend_version,
        )

    @abstractmethod
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> _BoxQuadraticBackendResult:
        """Return one candidate and bounded active-set diagnostics."""
        raise NotImplementedError


def _evaluate_loss(
    loss: PositiveDefiniteQuadraticLoss,
    coordinates: tuple[float, ...],
    /,
) -> float:
    try:
        return loss.squared_loss(coordinates)
    except ValueError as error:
        raise FloatingPointError("quadratic evaluation failed") from error


def _classify_bounds(
    coordinates: tuple[float, ...],
    bounds: ParameterSpace,
    /,
) -> tuple[BoxQuadraticBoundStatus, ...]:
    statuses: list[BoxQuadraticBoundStatus] = []
    for value, bound in zip(coordinates, bounds.bounds, strict=True):
        if bound.lower == bound.upper:
            status = BoxQuadraticBoundStatus.FIXED
        elif value == bound.lower:
            status = BoxQuadraticBoundStatus.LOWER
        elif value == bound.upper:
            status = BoxQuadraticBoundStatus.UPPER
        else:
            status = BoxQuadraticBoundStatus.FREE
        statuses.append(status)
    return tuple(statuses)


def _gradient_and_row_scales(
    matrix: tuple[tuple[float, ...], ...],
    center: tuple[float, ...],
    coordinates: tuple[float, ...],
    /,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    deltas: list[float] = []
    for coordinate, center_value in zip(coordinates, center, strict=True):
        delta = coordinate - center_value
        if not math.isfinite(delta):
            raise FloatingPointError("coordinate displacement must be finite")
        deltas.append(delta)

    gradient: list[float] = []
    row_scales: list[float] = []
    for row in matrix:
        terms: list[float] = []
        absolute_terms: list[float] = []
        for entry, delta in zip(row, deltas, strict=True):
            term = entry * delta
            if not math.isfinite(term):
                raise FloatingPointError("gradient terms must be finite")
            terms.append(term)
            absolute_terms.append(abs(term))
        try:
            row_sum = math.fsum(terms)
            absolute_sum = math.fsum(absolute_terms)
        except OverflowError as error:
            raise FloatingPointError("gradient sums must be finite") from error
        value = 2.0 * row_sum
        scale = 2.0 * absolute_sum
        if not math.isfinite(value) or not math.isfinite(scale):
            raise FloatingPointError("gradient and row scales must be finite")
        gradient.append(value)
        row_scales.append(scale)
    return tuple(gradient), tuple(row_scales)


def _kkt_tolerance_bound(
    tolerance: ConstrainedQuadraticTolerance,
    scale: float,
    /,
) -> float:
    result = tolerance.absolute + tolerance.relative * max(1.0, scale)
    if not math.isfinite(result) or result <= 0.0:
        raise FloatingPointError("KKT tolerance bounds must be positive and finite")
    return result


__all__ = [
    "BoxQuadraticBoundStatus",
    "BoxQuadraticMinimum",
    "BoxQuadraticMinimizer",
    "DEFAULT_BOX_QUADRATIC_MAXIMUM_ITERATIONS",
]
