"""NumPy dense active-set box minimization for positive quadratics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import numpy as np

from ....numerics.numpy.typing import Float64Array
from ...spaces.parameters import ParameterSpace
from ..box import (
    BoxQuadraticBackendResult,
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimizer,
    _gradient_and_row_scales,
    _kkt_tolerance_bound,
    _kkt_violations,
)
from ..constrained import ConstrainedQuadraticTolerance
from ..quadratic import PositiveDefiniteQuadraticLoss


@final
class NumpyBoxQuadraticMinimizer(BoxQuadraticMinimizer):
    """Apply a bounded deterministic primal active-set method."""

    __slots__ = ()

    @property
    def backend_identifier(self) -> str:
        """Return the trusted active-set backend declaration."""
        return "projectkoios.active-set.numpy"

    @property
    def backend_version(self) -> str:
        """Return the algorithm and imported NumPy version declaration."""
        return f"1/numpy-{np.__version__}"

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> BoxQuadraticBackendResult:
        """Return a candidate from the deterministic primal active-set method.

        Parameters
        ----------
        loss
            Validated dense positive-definite quadratic loss.
        bounds
            Validated finite box with the same positive dimension.

        Returns
        -------
        BoxQuadraticBackendResult
            Candidate coordinates and bounded execution counts.

        Raises
        ------
        TypeError
            If the validated dense matrix does not retain its tuple representation.
        FloatingPointError
            If clipped points, face solves, blocking steps, or updates become invalid.
        RuntimeError
            If a face solve fails, the iteration stagnates, or the limit is reached.
        """
        matrix_object = loss.matrix
        if type(matrix_object) is not tuple:
            raise TypeError("dense quadratic matrix must be a tuple of tuples")
        matrix = np.asarray(matrix_object, dtype=np.float64)
        center = np.asarray(loss.center, dtype=np.float64)
        lower = np.asarray(
            tuple(bound.lower for bound in bounds.bounds),
            dtype=np.float64,
        )
        upper = np.asarray(
            tuple(bound.upper for bound in bounds.bounds),
            dtype=np.float64,
        )
        point = np.clip(center, lower, upper)
        if not np.isfinite(point).all():
            raise FloatingPointError("initial clipped point must be finite")
        statuses = _initial_statuses(point, lower, upper)
        active_set_change_count = 0
        linear_solve_count = 0

        for iteration_count in range(1, self.maximum_iterations + 1):
            evidence = _active_set_kkt_evidence(
                matrix_object,
                loss.center,
                point,
                statuses,
                self.tolerance,
            )
            if not evidence.free_violations:
                if not evidence.active_violations:
                    return BoxQuadraticBackendResult(
                        coordinates=evidence.coordinates,
                        iteration_count=iteration_count,
                        active_set_change_count=active_set_change_count,
                        linear_solve_count=linear_solve_count,
                    )
                # Smallest-index release is the deterministic anti-ambiguity rule.
                statuses[evidence.active_violations[0]] = BoxQuadraticBoundStatus.FREE
                active_set_change_count += 1
                continue

            direction = _solve_free_direction(
                matrix,
                evidence.gradient,
                evidence.free_indices,
            )
            linear_solve_count += 1
            step = _blocking_step(
                point,
                direction,
                evidence.free_indices,
                lower,
                upper,
            )
            previous = point.copy()
            active_set_change_count += _advance_free_coordinates(
                point,
                statuses,
                direction,
                evidence.free_indices,
                lower,
                upper,
                step,
                self.tolerance,
            )
            if np.array_equal(point, previous):
                raise RuntimeError("dense active-set solve stagnated before optimality")

        raise RuntimeError("dense active-set solve reached maximum_iterations")


@dataclass(frozen=True, slots=True)
class _ActiveSetKktEvidence:
    """Hold one face's immutable KKT decision evidence.

    Parameters
    ----------
    coordinates
        Current feasible point in dependency-free tuple form.
    gradient
        Full gradient at ``coordinates``.
    violations
        Per-coordinate KKT violations under the current statuses.
    tolerance_bounds
        Per-coordinate row-local acceptance bounds.
    free_indices
        Current free-coordinate indices in ascending order.
    free_violations
        Nonstationary free-coordinate indices in ascending order.
    active_violations
        Invalid lower/upper sign-condition indices in ascending order.
    """

    coordinates: tuple[float, ...]
    gradient: tuple[float, ...]
    violations: tuple[float, ...]
    tolerance_bounds: tuple[float, ...]
    free_indices: tuple[int, ...]
    free_violations: tuple[int, ...]
    active_violations: tuple[int, ...]


def _active_set_kkt_evidence(
    matrix: tuple[tuple[float, ...], ...],
    center: tuple[float, ...],
    point: Float64Array,
    statuses: list[BoxQuadraticBoundStatus],
    tolerance: ConstrainedQuadraticTolerance,
    /,
) -> _ActiveSetKktEvidence:
    """Reconstruct one face's KKT evidence and deterministic index sets.

    Parameters
    ----------
    matrix, center
        Validated dense quadratic matrix and unconstrained center.
    point
        Current finite feasible binary64 point.
    statuses
        Current bound status for every coordinate.
    tolerance
        Row-local KKT acceptance policy.

    Returns
    -------
    _ActiveSetKktEvidence
        Immutable arithmetic and decision evidence in coordinate order.

    Raises
    ------
    FloatingPointError
        If gradient or tolerance-bound arithmetic becomes nonfinite.
    """
    coordinates = tuple(float(value) for value in point)
    gradient, row_scales = _gradient_and_row_scales(matrix, center, coordinates)
    violations = _kkt_violations(gradient, tuple(statuses))
    tolerance_bounds = tuple(
        _kkt_tolerance_bound(tolerance, scale) for scale in row_scales
    )
    free_indices = tuple(
        index
        for index, status in enumerate(statuses)
        if status is BoxQuadraticBoundStatus.FREE
    )
    free_violations = tuple(
        index for index in free_indices if violations[index] > tolerance_bounds[index]
    )
    active_violations = tuple(
        index
        for index, status in enumerate(statuses)
        if status
        in (
            BoxQuadraticBoundStatus.LOWER,
            BoxQuadraticBoundStatus.UPPER,
        )
        and violations[index] > tolerance_bounds[index]
    )
    return _ActiveSetKktEvidence(
        coordinates=coordinates,
        gradient=gradient,
        violations=violations,
        tolerance_bounds=tolerance_bounds,
        free_indices=free_indices,
        free_violations=free_violations,
        active_violations=active_violations,
    )


def _solve_free_direction(
    matrix: Float64Array,
    gradient: tuple[float, ...],
    free_indices: tuple[int, ...],
    /,
) -> Float64Array:
    """Solve the exact Newton direction on the current free face.

    Parameters
    ----------
    matrix
        Dense positive-definite binary64 quadratic matrix.
    gradient
        Full dependency-free gradient at the current point.
    free_indices
        Nonempty current free-coordinate indices in ascending order.

    Returns
    -------
    numpy.ndarray
        Finite binary64 direction indexed by ``free_indices``.

    Raises
    ------
    FloatingPointError
        If solve arithmetic or the returned direction becomes nonfinite.
    RuntimeError
        If NumPy cannot solve the positive-definite principal system.
    """
    free = np.asarray(free_indices, dtype=np.intp)
    half_gradient = np.asarray(
        tuple(0.5 * gradient[index] for index in free_indices),
        dtype=np.float64,
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            direction = -np.linalg.solve(
                matrix[np.ix_(free, free)],
                half_gradient,
            )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("dense active-set face solve failed") from error
    except FloatingPointError as error:
        raise FloatingPointError(
            "dense active-set face solve arithmetic must be finite"
        ) from error
    if not np.isfinite(direction).all():
        raise FloatingPointError("active-set direction must be finite")
    return direction


def _blocking_step(
    point: Float64Array,
    direction: Float64Array,
    free_indices: tuple[int, ...],
    lower: Float64Array,
    upper: Float64Array,
    /,
) -> float:
    """Return the largest unit-bounded step that remains inside the box.

    Parameters
    ----------
    point
        Current finite feasible binary64 point.
    direction
        Finite free-face direction indexed by ``free_indices``.
    free_indices
        Current free-coordinate indices in ascending order.
    lower, upper
        Binary64 coordinate bounds.

    Returns
    -------
    float
        Finite step in the closed interval from zero to one.

    Raises
    ------
    FloatingPointError
        If a required blocking ratio is nonfinite or negative.
    """
    step = 1.0
    for offset, index in enumerate(free_indices):
        component = float(direction[offset])
        candidate_step: float | None = None
        if component > 0.0 and point[index] + component > upper[index]:
            candidate_step = float((upper[index] - point[index]) / component)
        elif component < 0.0 and point[index] + component < lower[index]:
            candidate_step = float((lower[index] - point[index]) / component)
        if candidate_step is not None:
            if not math.isfinite(candidate_step) or candidate_step < 0.0:
                raise FloatingPointError(
                    "active-set blocking step must be finite and nonnegative"
                )
            if candidate_step < step:
                step = candidate_step
    return step


def _advance_free_coordinates(
    point: Float64Array,
    statuses: list[BoxQuadraticBoundStatus],
    direction: Float64Array,
    free_indices: tuple[int, ...],
    lower: Float64Array,
    upper: Float64Array,
    step: float,
    tolerance: ConstrainedQuadraticTolerance,
    /,
) -> int:
    """Apply one face step, snap endpoints, and return activation count.

    Parameters
    ----------
    point
        Mutable current binary64 point, updated in place.
    statuses
        Mutable bound statuses, updated in place for snapped coordinates.
    direction
        Finite free-face direction indexed by ``free_indices``.
    free_indices
        Current free-coordinate indices in ascending order.
    lower, upper
        Binary64 coordinate bounds.
    step
        Finite nonnegative feasible step no greater than one.
    tolerance
        Endpoint-snapping acceptance policy.

    Returns
    -------
    int
        Number of coordinates newly activated in this update.

    Raises
    ------
    FloatingPointError
        If point-update arithmetic or the resulting point becomes nonfinite.
    """
    free = np.asarray(free_indices, dtype=np.intp)
    try:
        with np.errstate(over="raise", invalid="raise"):
            point[free] = point[free] + step * direction
    except FloatingPointError as error:
        raise FloatingPointError("active-set point update must be finite") from error
    if not np.isfinite(point).all():
        raise FloatingPointError("active-set point must be finite")

    activation_count = 0
    for index in free_indices:
        snapped_value, snapped_status = _snap_free_coordinate(
            float(point[index]),
            float(lower[index]),
            float(upper[index]),
            tolerance,
        )
        if snapped_status is not None:
            point[index] = snapped_value
            statuses[index] = snapped_status
            activation_count += 1
    return activation_count


def _snap_free_coordinate(
    value: float,
    lower: float,
    upper: float,
    tolerance: ConstrainedQuadraticTolerance,
    /,
) -> tuple[float, BoxQuadraticBoundStatus | None]:
    """Apply deterministic exact-hit and tolerance endpoint snapping.

    Parameters
    ----------
    value
        Finite updated value for a formerly free coordinate.
    lower, upper
        Finite ordered endpoints for that coordinate.
    tolerance
        Scale-aware endpoint acceptance policy.

    Returns
    -------
    tuple[float, BoxQuadraticBoundStatus or None]
        Snapped endpoint and new status, or the original interior value and no status.

    Notes
    -----
    Exact endpoint hits precede tolerance comparisons. For a strictly interior value
    accepted against both endpoints, the lower endpoint has deterministic precedence.
    """
    if value <= lower:
        return lower, BoxQuadraticBoundStatus.LOWER
    if value >= upper:
        return upper, BoxQuadraticBoundStatus.UPPER
    if tolerance.accepts(value, lower):
        return lower, BoxQuadraticBoundStatus.LOWER
    if tolerance.accepts(value, upper):
        return upper, BoxQuadraticBoundStatus.UPPER
    return value, None


def _initial_statuses(
    point: Float64Array,
    lower: Float64Array,
    upper: Float64Array,
    /,
) -> list[BoxQuadraticBoundStatus]:
    """Classify the clipped initial point in ascending coordinate order.

    Parameters
    ----------
    point
        Finite binary64 point already clipped into the box.
    lower, upper
        Finite ordered binary64 coordinate bounds.

    Returns
    -------
    list[BoxQuadraticBoundStatus]
        Mutable status state for the active-set iteration.
    """
    statuses: list[BoxQuadraticBoundStatus] = []
    for value, lower_value, upper_value in zip(
        point,
        lower,
        upper,
        strict=True,
    ):
        if lower_value == upper_value:
            status = BoxQuadraticBoundStatus.FIXED
        elif value == lower_value:
            status = BoxQuadraticBoundStatus.LOWER
        elif value == upper_value:
            status = BoxQuadraticBoundStatus.UPPER
        else:
            status = BoxQuadraticBoundStatus.FREE
        statuses.append(status)
    return statuses


__all__ = ["NumpyBoxQuadraticMinimizer"]
