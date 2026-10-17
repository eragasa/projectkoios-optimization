"""Dependency-free contracts for dense positive-quadratic box minimization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from projectkoios.numerics.validation import finite_float, integer, u32

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


def _nonnegative_int(value: object, name: str, /) -> int:
    result = integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, name: str, /) -> int:
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

    def __post_init__(self) -> None:
        coordinates = finite_vector(self.coordinates, "coordinates")
        squared_loss = finite_float(self.squared_loss, "squared_loss")
        if squared_loss < 0.0:
            raise ValueError("squared_loss must be nonnegative")
        if type(self.bound_statuses) is not tuple or any(
            type(value) is not BoxQuadraticBoundStatus for value in self.bound_statuses
        ):
            raise TypeError("bound_statuses must be a tuple of BoxQuadraticBoundStatus")
        gradient = finite_vector(self.gradient, "gradient")
        tolerance_bounds = finite_vector(
            self.kkt_tolerance_bounds,
            "kkt_tolerance_bounds",
        )
        dimension = len(coordinates)
        if len(self.bound_statuses) != dimension:
            raise ValueError("bound_statuses must match the coordinate dimension")
        if len(gradient) != dimension:
            raise ValueError("gradient must match the coordinate dimension")
        if len(tolerance_bounds) != dimension:
            raise ValueError("kkt_tolerance_bounds must match the coordinate dimension")
        if any(value <= 0.0 for value in tolerance_bounds):
            raise ValueError("kkt_tolerance_bounds must be positive")

        maximum_violation = finite_float(
            self.maximum_kkt_violation,
            "maximum_kkt_violation",
        )
        maximum_ratio = finite_float(self.maximum_kkt_ratio, "maximum_kkt_ratio")
        if maximum_violation < 0.0:
            raise ValueError("maximum_kkt_violation must be nonnegative")
        if maximum_ratio < 0.0:
            raise ValueError("maximum_kkt_ratio must be nonnegative")
        violations = _kkt_violations(gradient, self.bound_statuses)
        expected_maximum = max(violations)
        expected_ratio = max(
            violation / bound
            for violation, bound in zip(
                violations,
                tolerance_bounds,
                strict=True,
            )
        )
        if maximum_violation != expected_maximum:
            raise ValueError("maximum_kkt_violation is inconsistent")
        if maximum_ratio != expected_ratio:
            raise ValueError("maximum_kkt_ratio is inconsistent")
        if maximum_ratio > 1.0:
            raise ValueError("maximum_kkt_ratio must not exceed one")

        iterations = _positive_int(self.iteration_count, "iteration_count")
        changes = _nonnegative_int(
            self.active_set_change_count,
            "active_set_change_count",
        )
        solves = _nonnegative_int(self.linear_solve_count, "linear_solve_count")
        maximum_iterations = u32(
            self.maximum_iterations,
            "maximum_iterations",
            allow_zero=False,
        )
        if iterations > maximum_iterations:
            raise ValueError("iteration_count must not exceed maximum_iterations")
        if solves > iterations:
            raise ValueError("linear_solve_count must not exceed iteration_count")
        identifier = text(self.backend_identifier, "backend_identifier")
        version = text(self.backend_version, "backend_version")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "squared_loss", squared_loss)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "kkt_tolerance_bounds", tolerance_bounds)
        object.__setattr__(self, "maximum_kkt_violation", maximum_violation)
        object.__setattr__(self, "maximum_kkt_ratio", maximum_ratio)
        object.__setattr__(self, "iteration_count", iterations)
        object.__setattr__(self, "active_set_change_count", changes)
        object.__setattr__(self, "linear_solve_count", solves)
        object.__setattr__(self, "maximum_iterations", maximum_iterations)
        object.__setattr__(self, "backend_identifier", identifier)
        object.__setattr__(self, "backend_version", version)


@final
@dataclass(frozen=True, slots=True)
class _BoxQuadraticBackendResult:
    """Private candidate handoff from one dense backend to the final template."""

    coordinates: tuple[float, ...]
    iteration_count: int
    active_set_change_count: int
    linear_solve_count: int

    def __post_init__(self) -> None:
        coordinates = finite_vector(self.coordinates, "coordinates")
        iterations = _positive_int(self.iteration_count, "iteration_count")
        changes = _nonnegative_int(
            self.active_set_change_count,
            "active_set_change_count",
        )
        solves = _nonnegative_int(self.linear_solve_count, "linear_solve_count")
        if solves > iterations:
            raise ValueError("linear_solve_count must not exceed iteration_count")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "iteration_count", iterations)
        object.__setattr__(self, "active_set_change_count", changes)
        object.__setattr__(self, "linear_solve_count", solves)


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

        return BoxQuadraticMinimum(
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
