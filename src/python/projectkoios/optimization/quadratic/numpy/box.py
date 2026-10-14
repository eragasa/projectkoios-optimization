"""NumPy dense active-set box minimization for positive quadratics."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from ....numerics.numpy.typing import Float64Array
from ...spaces.parameters import ParameterSpace
from ..box import (
    BoxQuadraticBoundStatus,
    BoxQuadraticMinimizer,
    _BoxQuadraticBackendResult,
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
    ) -> _BoxQuadraticBackendResult:
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
            coordinates = tuple(float(value) for value in point)
            gradient, row_scales = _gradient_and_row_scales(
                matrix_object,
                loss.center,
                coordinates,
            )
            violations = _kkt_violations(gradient, tuple(statuses))
            tolerance_bounds = tuple(
                _kkt_tolerance_bound(self.tolerance, scale) for scale in row_scales
            )
            free_violations = tuple(
                index
                for index, status in enumerate(statuses)
                if status is BoxQuadraticBoundStatus.FREE
                and violations[index] > tolerance_bounds[index]
            )

            if not free_violations:
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
                if not active_violations:
                    return _BoxQuadraticBackendResult(
                        coordinates=coordinates,
                        iteration_count=iteration_count,
                        active_set_change_count=active_set_change_count,
                        linear_solve_count=linear_solve_count,
                    )
                # Smallest-index release is the deterministic anti-ambiguity rule.
                statuses[active_violations[0]] = BoxQuadraticBoundStatus.FREE
                active_set_change_count += 1
                continue

            free_indices = tuple(
                index
                for index, status in enumerate(statuses)
                if status is BoxQuadraticBoundStatus.FREE
            )
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
            linear_solve_count += 1
            if not np.isfinite(direction).all():
                raise FloatingPointError("active-set direction must be finite")

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

            previous = point.copy()
            try:
                with np.errstate(over="raise", invalid="raise"):
                    point[free] = point[free] + step * direction
            except FloatingPointError as error:
                raise FloatingPointError(
                    "active-set point update must be finite"
                ) from error
            if not np.isfinite(point).all():
                raise FloatingPointError("active-set point must be finite")

            for index in free_indices:
                snapped_value, snapped_status = _snap_free_coordinate(
                    float(point[index]),
                    float(lower[index]),
                    float(upper[index]),
                    self.tolerance,
                )
                if snapped_status is not None:
                    point[index] = snapped_value
                    statuses[index] = snapped_status
                    active_set_change_count += 1

            if np.array_equal(point, previous):
                raise RuntimeError("dense active-set solve stagnated before optimality")

        raise RuntimeError("dense active-set solve reached maximum_iterations")


def _snap_free_coordinate(
    value: float,
    lower: float,
    upper: float,
    tolerance: ConstrainedQuadraticTolerance,
    /,
) -> tuple[float, BoxQuadraticBoundStatus | None]:
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
