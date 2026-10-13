"""Dependency-free contracts for real and complex linear least squares."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Complex
from typing import final

from ..validation.validation import finite_float

type LinearLeastSquaresMatrix = tuple[tuple[complex, ...], ...]


@final
@dataclass(frozen=True, slots=True)
class LinearLeastSquaresTolerance:
    """Numerical acceptance tolerance for least-squares backend results."""

    absolute: float = 1.0e-12
    relative: float = 1.0e-10

    def __post_init__(self) -> None:
        absolute = finite_float(self.absolute, "absolute")
        relative = finite_float(self.relative, "relative")
        if absolute < 0.0:
            raise ValueError("absolute must be nonnegative")
        if relative < 0.0:
            raise ValueError("relative must be nonnegative")
        if absolute == 0.0 and relative == 0.0:
            raise ValueError("at least one tolerance must be positive")
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)

    def accepts(self, left: complex, right: complex, /) -> bool:
        """Return whether two finite scalars satisfy the scale-aware tolerance."""
        normalized_left = _finite_complex(left, "left")
        normalized_right = _finite_complex(right, "right")
        difference = abs(normalized_left - normalized_right)
        bound = self.absolute + self.relative * max(
            abs(normalized_left),
            abs(normalized_right),
        )
        return (
            math.isfinite(difference) and math.isfinite(bound) and difference <= bound
        )

    def accepts_zero(self, value: complex, scale: float, /) -> bool:
        """Return whether a finite residual is zero at a declared finite scale."""
        normalized_value = _finite_complex(value, "value")
        normalized_scale = finite_float(scale, "scale")
        if normalized_scale < 0.0:
            raise ValueError("scale must be nonnegative")
        bound = self.absolute + self.relative * normalized_scale
        return math.isfinite(bound) and abs(normalized_value) <= bound


DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE = LinearLeastSquaresTolerance()


@final
@dataclass(frozen=True, slots=True)
class LinearLeastSquaresResult:
    """Retain one validated linear least-squares solution and diagnostics."""

    coefficients: LinearLeastSquaresMatrix
    fitted_values: LinearLeastSquaresMatrix
    residuals: LinearLeastSquaresMatrix
    residual_norms: tuple[float, ...]
    weighted_residual_norms: tuple[float, ...]
    residual_frobenius_norm: float
    weighted_residual_frobenius_norm: float
    maximum_row_residual_norm: float
    rank: int
    singular_values: tuple[float, ...]
    full_rank_condition_number: float | None

    def __post_init__(self) -> None:
        coefficients = _finite_matrix(self.coefficients, "coefficients")
        fitted_values = _finite_matrix(self.fitted_values, "fitted_values")
        residuals = _finite_matrix(self.residuals, "residuals")
        coefficient_columns = len(coefficients[0])
        if len(fitted_values[0]) != coefficient_columns:
            raise ValueError(
                "coefficients and fitted_values must have one target-column count"
            )
        if _matrix_shape(residuals) != _matrix_shape(fitted_values):
            raise ValueError("residuals must match fitted_values shape")

        residual_norms = _nonnegative_vector(
            self.residual_norms,
            "residual_norms",
        )
        weighted_residual_norms = _nonnegative_vector(
            self.weighted_residual_norms,
            "weighted_residual_norms",
        )
        if len(residual_norms) != coefficient_columns:
            raise ValueError("one residual_norm is required per target column")
        if len(weighted_residual_norms) != coefficient_columns:
            raise ValueError("one weighted_residual_norm is required per target column")

        residual_frobenius_norm = _nonnegative_float(
            self.residual_frobenius_norm,
            "residual_frobenius_norm",
        )
        weighted_residual_frobenius_norm = _nonnegative_float(
            self.weighted_residual_frobenius_norm,
            "weighted_residual_frobenius_norm",
        )
        maximum_row_residual_norm = _nonnegative_float(
            self.maximum_row_residual_norm,
            "maximum_row_residual_norm",
        )

        if type(self.rank) is not int:
            raise TypeError("rank must be a built-in int")
        maximum_rank = min(len(fitted_values), len(coefficients))
        if self.rank < 0 or self.rank > maximum_rank:
            raise ValueError("rank must lie within the matrix rank bound")
        singular_values = _nonnegative_vector(
            self.singular_values,
            "singular_values",
        )
        if len(singular_values) != maximum_rank:
            raise ValueError("singular_values must match the matrix rank bound")
        if any(
            left < right
            for left, right in zip(
                singular_values,
                singular_values[1:],
                strict=False,
            )
        ):
            raise ValueError("singular_values must be nonincreasing")

        condition = self.full_rank_condition_number
        has_full_column_rank = self.rank == len(coefficients)
        if has_full_column_rank:
            condition = _nonnegative_float(
                condition,
                "full_rank_condition_number",
            )
            if condition < 1.0:
                raise ValueError("full_rank_condition_number must be at least one")
        elif condition is not None:
            raise ValueError(
                "full_rank_condition_number is defined only at full column rank"
            )

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "fitted_values", fitted_values)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "residual_norms", residual_norms)
        object.__setattr__(
            self,
            "weighted_residual_norms",
            weighted_residual_norms,
        )
        object.__setattr__(
            self,
            "residual_frobenius_norm",
            residual_frobenius_norm,
        )
        object.__setattr__(
            self,
            "weighted_residual_frobenius_norm",
            weighted_residual_frobenius_norm,
        )
        object.__setattr__(
            self,
            "maximum_row_residual_norm",
            maximum_row_residual_norm,
        )
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "full_rank_condition_number", condition)


class LinearLeastSquaresSolver(ABC):
    """Nominal SDK boundary for ordinary and positive-weight least squares."""

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        *,
        tolerance: LinearLeastSquaresTolerance = (
            DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE
        ),
    ) -> None:
        if type(tolerance) is not LinearLeastSquaresTolerance:
            raise TypeError("tolerance must be LinearLeastSquaresTolerance")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> LinearLeastSquaresTolerance:
        """Return the backend-result acceptance tolerance."""
        return self._tolerance

    @final
    def solve(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        /,
        *,
        weights: tuple[float, ...] | None = None,
        rank_tolerance: float | None = None,
    ) -> LinearLeastSquaresResult:
        """Minimize a real or complex ordinary or positive-weight residual."""
        design = _finite_matrix(design_matrix, "design_matrix")
        targets = _finite_matrix(target_matrix, "target_matrix")
        if len(targets) != len(design):
            raise ValueError("target_matrix row count must match design_matrix")
        normalized_weights = _weights(weights, len(design))
        normalized_rank_tolerance = _rank_tolerance(
            rank_tolerance,
            len(design),
            len(design[0]),
        )

        result = self._solve_validated(
            design,
            targets,
            normalized_weights,
            normalized_rank_tolerance,
        )
        if type(result) is not LinearLeastSquaresResult:
            raise TypeError("backend must return LinearLeastSquaresResult")
        if len(result.coefficients) != len(design[0]):
            raise ValueError("backend coefficient row count is inconsistent")
        if len(result.coefficients[0]) != len(targets[0]):
            raise ValueError("backend coefficient column count is inconsistent")
        if _matrix_shape(result.fitted_values) != _matrix_shape(targets):
            raise ValueError("backend fitted_values shape is inconsistent")

        expected_fitted = _matrix_product(design, result.coefficients)
        expected_residuals = _matrix_difference(expected_fitted, targets)
        _require_matrix_agreement(
            result.fitted_values,
            expected_fitted,
            self.tolerance,
            "backend fitted_values are inconsistent",
        )
        _require_matrix_agreement(
            result.residuals,
            expected_residuals,
            self.tolerance,
            "backend residuals are inconsistent",
        )

        row_weights = normalized_weights or (1.0,) * len(design)
        residual_norms = _column_norms(expected_residuals)
        weighted_residual_norms = _column_norms(expected_residuals, row_weights)
        residual_frobenius_norm = _frobenius_norm(expected_residuals)
        weighted_residual_frobenius_norm = _frobenius_norm(
            expected_residuals,
            row_weights,
        )
        maximum_row_residual_norm = _maximum_row_norm(expected_residuals)
        _require_real_vector_agreement(
            result.residual_norms,
            residual_norms,
            self.tolerance,
            "backend residual_norms are inconsistent",
        )
        _require_real_vector_agreement(
            result.weighted_residual_norms,
            weighted_residual_norms,
            self.tolerance,
            "backend weighted_residual_norms are inconsistent",
        )
        for actual, expected, message in (
            (
                result.residual_frobenius_norm,
                residual_frobenius_norm,
                "backend residual_frobenius_norm is inconsistent",
            ),
            (
                result.weighted_residual_frobenius_norm,
                weighted_residual_frobenius_norm,
                "backend weighted_residual_frobenius_norm is inconsistent",
            ),
            (
                result.maximum_row_residual_norm,
                maximum_row_residual_norm,
                "backend maximum_row_residual_norm is inconsistent",
            ),
        ):
            if not self.tolerance.accepts(complex(actual), complex(expected)):
                raise ValueError(message)

        _require_spectral_metadata(
            design,
            row_weights,
            result,
            normalized_rank_tolerance,
            self.tolerance,
        )
        if not _has_discarded_positive_singular_direction(
            result.rank,
            result.singular_values,
        ):
            _require_stationarity(
                design,
                expected_residuals,
                row_weights,
                self.tolerance,
            )
        expected_condition = _condition_number(
            result.rank,
            len(design[0]),
            result.singular_values,
        )
        if expected_condition is None:
            if result.full_rank_condition_number is not None:
                raise ValueError("backend full-rank condition is inconsistent")
        elif result.full_rank_condition_number is None or not self.tolerance.accepts(
            complex(result.full_rank_condition_number),
            complex(expected_condition),
        ):
            raise ValueError("backend full-rank condition is inconsistent")
        return result

    @abstractmethod
    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        """Solve validated inputs with one concrete backend."""
        raise NotImplementedError


def _finite_complex(value: object, name: str, /) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError(f"{name} must be a real or complex number excluding bool")
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be representable as finite complex128"
        ) from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be representable as finite complex128")
    return result


def _finite_matrix(value: object, name: str, /) -> LinearLeastSquaresMatrix:
    if type(value) is not tuple or not value:
        raise TypeError(f"{name} must be a nonempty tuple of nonempty tuples")
    rows: list[tuple[complex, ...]] = []
    column_count: int | None = None
    for row in value:
        if type(row) is not tuple or not row:
            raise TypeError(f"{name} must be a nonempty tuple of nonempty tuples")
        normalized = tuple(_finite_complex(entry, name) for entry in row)
        if column_count is None:
            column_count = len(normalized)
        elif len(normalized) != column_count:
            raise ValueError(f"{name} rows must have one common length")
        rows.append(normalized)
    return tuple(rows)


def _matrix_shape(matrix: LinearLeastSquaresMatrix, /) -> tuple[int, int]:
    return (len(matrix), len(matrix[0]))


def _weights(value: object, row_count: int, /) -> tuple[float, ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise TypeError("weights must be None or an exact tuple")
    if len(value) != row_count:
        raise ValueError("weights must contain one value per design row")
    weights = tuple(finite_float(entry, "weight") for entry in value)
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("weights must be strictly positive")
    return weights


def _rank_tolerance(
    value: object,
    row_count: int,
    column_count: int,
    /,
) -> float:
    if value is None:
        return sys.float_info.epsilon * max(row_count, column_count)
    tolerance = finite_float(value, "rank_tolerance")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("rank_tolerance must be at least zero and less than one")
    return tolerance


def _nonnegative_float(value: object, name: str, /) -> float:
    result = finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonnegative_vector(value: object, name: str, /) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{name} must be a nonempty tuple")
    return tuple(_nonnegative_float(entry, name) for entry in value)


def _finite_complex_sum(values: list[complex], /) -> complex:
    try:
        real = math.fsum(value.real for value in values)
        imaginary = math.fsum(value.imag for value in values)
    except (OverflowError, ValueError) as error:
        raise ValueError("complex sum must be finite") from error
    if not math.isfinite(real) or not math.isfinite(imaginary):
        raise ValueError("complex sum must be finite")
    return complex(real, imaginary)


def _matrix_product(
    left: LinearLeastSquaresMatrix,
    right: LinearLeastSquaresMatrix,
    /,
) -> LinearLeastSquaresMatrix:
    if len(left[0]) != len(right):
        raise ValueError("matrix product dimensions must agree")
    return tuple(
        tuple(
            _finite_complex_sum(
                [left[row][inner] * right[inner][column] for inner in range(len(right))]
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matrix_difference(
    left: LinearLeastSquaresMatrix,
    right: LinearLeastSquaresMatrix,
    /,
) -> LinearLeastSquaresMatrix:
    return tuple(
        tuple(
            _finite_complex(left_value - right_value, "matrix difference")
            for left_value, right_value in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _require_matrix_agreement(
    actual: LinearLeastSquaresMatrix,
    expected: LinearLeastSquaresMatrix,
    tolerance: LinearLeastSquaresTolerance,
    message: str,
    /,
) -> None:
    if _matrix_shape(actual) != _matrix_shape(expected):
        raise ValueError(message)
    if any(
        not tolerance.accepts(actual_value, expected_value)
        for actual_row, expected_row in zip(actual, expected, strict=True)
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True)
    ):
        raise ValueError(message)


def _column_norms(
    residuals: LinearLeastSquaresMatrix,
    weights: tuple[float, ...] | None = None,
    /,
) -> tuple[float, ...]:
    row_weights = weights or (1.0,) * len(residuals)
    return tuple(
        math.hypot(
            *(
                math.sqrt(row_weights[row]) * abs(residuals[row][column])
                for row in range(len(residuals))
            )
        )
        for column in range(len(residuals[0]))
    )


def _frobenius_norm(
    residuals: LinearLeastSquaresMatrix,
    weights: tuple[float, ...] | None = None,
    /,
) -> float:
    row_weights = weights or (1.0,) * len(residuals)
    return math.hypot(
        *(
            math.sqrt(row_weights[row]) * abs(value)
            for row, values in enumerate(residuals)
            for value in values
        )
    )


def _maximum_row_norm(residuals: LinearLeastSquaresMatrix, /) -> float:
    return max(math.hypot(*(abs(value) for value in row)) for row in residuals)


def _require_real_vector_agreement(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    tolerance: LinearLeastSquaresTolerance,
    message: str,
    /,
) -> None:
    if len(actual) != len(expected) or any(
        not tolerance.accepts(complex(left), complex(right))
        for left, right in zip(actual, expected, strict=True)
    ):
        raise ValueError(message)


def _require_spectral_metadata(
    design: LinearLeastSquaresMatrix,
    weights: tuple[float, ...],
    result: LinearLeastSquaresResult,
    rank_tolerance: float,
    tolerance: LinearLeastSquaresTolerance,
    /,
) -> None:
    weighted_design_norm = _frobenius_norm(design, weights)
    singular_value_norm = math.hypot(*result.singular_values)
    if not math.isfinite(weighted_design_norm) or not math.isfinite(
        singular_value_norm
    ):
        raise ValueError("weighted-design spectral diagnostics must be finite")
    spectral_scale = max(weighted_design_norm, singular_value_norm)
    spectral_relative_tolerance = max(
        tolerance.relative,
        16.0 * sys.float_info.epsilon * max(len(design), len(design[0])),
    )
    if spectral_scale == 0.0:
        spectral_agrees = weighted_design_norm == singular_value_norm
    else:
        spectral_agrees = (
            abs(weighted_design_norm - singular_value_norm)
            <= spectral_relative_tolerance * spectral_scale
        )
    if not spectral_agrees:
        raise ValueError("backend singular_values are inconsistent with the design")
    largest = result.singular_values[0]
    cutoff = rank_tolerance * largest
    expected_rank = sum(value > cutoff for value in result.singular_values)
    if result.rank != expected_rank:
        raise ValueError("backend rank is inconsistent with singular_values")


def _has_discarded_positive_singular_direction(
    rank: int,
    singular_values: tuple[float, ...],
    /,
) -> bool:
    return any(value > 0.0 for value in singular_values[rank:])


def _require_stationarity(
    design: LinearLeastSquaresMatrix,
    residuals: LinearLeastSquaresMatrix,
    weights: tuple[float, ...],
    tolerance: LinearLeastSquaresTolerance,
    /,
) -> None:
    square_root_weights = tuple(math.sqrt(weight) for weight in weights)
    for coefficient in range(len(design[0])):
        weighted_design = tuple(
            _finite_complex(
                square_root_weights[row] * design[row][coefficient],
                "weighted design entry",
            )
            for row in range(len(design))
        )
        design_scale = max(abs(value) for value in weighted_design)
        for target in range(len(residuals[0])):
            weighted_residuals = tuple(
                _finite_complex(
                    square_root_weights[row] * residuals[row][target],
                    "weighted residual entry",
                )
                for row in range(len(design))
            )
            residual_scale = max(abs(value) for value in weighted_residuals)
            if design_scale == 0.0 or residual_scale == 0.0:
                continue
            terms = [
                (design_value / design_scale).conjugate()
                * (residual_value / residual_scale)
                for design_value, residual_value in zip(
                    weighted_design,
                    weighted_residuals,
                    strict=True,
                )
            ]
            scaled_stationarity = _finite_complex_sum(terms)
            scaled_term_sum = math.fsum(abs(term) for term in terms)
            scaled_absolute = _scaled_absolute_tolerance(
                tolerance.absolute,
                design_scale,
                residual_scale,
            )
            bound = scaled_absolute + tolerance.relative * scaled_term_sum
            if abs(scaled_stationarity) > bound:
                raise ValueError("backend coefficients do not satisfy stationarity")


def _scaled_absolute_tolerance(
    absolute: float,
    first_scale: float,
    second_scale: float,
    /,
) -> float:
    if absolute == 0.0:
        return 0.0
    exponent = math.log(absolute) - math.log(first_scale) - math.log(second_scale)
    maximum_exponent = math.log(sys.float_info.max)
    minimum_exponent = math.log(math.ulp(0.0))
    if exponent >= maximum_exponent:
        return math.inf
    if exponent < minimum_exponent:
        return 0.0
    return math.exp(exponent)


def _condition_number(
    rank: int,
    coefficient_count: int,
    singular_values: tuple[float, ...],
    /,
) -> float | None:
    if rank != coefficient_count:
        return None
    smallest = singular_values[-1]
    if smallest <= 0.0:
        raise ValueError("full-rank singular values must be positive")
    condition = singular_values[0] / smallest
    if not math.isfinite(condition):
        raise ValueError("full-rank condition number must be finite")
    return condition


__all__ = [
    "DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE",
    "LinearLeastSquaresMatrix",
    "LinearLeastSquaresResult",
    "LinearLeastSquaresSolver",
    "LinearLeastSquaresTolerance",
]
