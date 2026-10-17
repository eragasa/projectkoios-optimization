"""Contracts for complex affine residual quadratics with real coordinates."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Complex
from typing import final

from projectkoios.numerics.typing import FiniteComplexMatrix
from projectkoios.numerics.validation import finite_float

from ..quadratic.quadratic import PositiveDefiniteQuadraticLoss

type ComplexAffineResidualMatrix = FiniteComplexMatrix
type ComplexAffineResidualVector = tuple[complex, ...]


@final
@dataclass(frozen=True, slots=True)
class ComplexAffineResidualQuadraticTolerance:
    """Numerical tolerance for relational validation of constructed quadratics."""

    absolute: float = 1.0e-12
    relative: float = 1.0e-10

    def __post_init__(self) -> None:
        """Normalize finite nonnegative tolerance components."""
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

    def accepts(self, left: float, right: float, /) -> bool:
        """Return whether two finite real values satisfy this tolerance."""
        normalized_left = finite_float(left, "left")
        normalized_right = finite_float(right, "right")
        difference = abs(normalized_left - normalized_right)
        bound = self.absolute + self.relative * max(
            abs(normalized_left),
            abs(normalized_right),
        )
        return (
            math.isfinite(difference) and math.isfinite(bound) and difference <= bound
        )


DEFAULT_COMPLEX_AFFINE_RESIDUAL_QUADRATIC_TOLERANCE = (
    ComplexAffineResidualQuadraticTolerance()
)


class ComplexAffineResidualQuadraticConstructor(ABC):
    r"""Nominal constructor for complex observations and real coordinates.

    The represented objective is
    ``sum(weights[i] * abs(design[i] @ x - targets[i]) ** 2)`` for real ``x``.
    ``None`` selects unit objective weights.
    """

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        *,
        tolerance: ComplexAffineResidualQuadraticTolerance = (
            DEFAULT_COMPLEX_AFFINE_RESIDUAL_QUADRATIC_TOLERANCE
        ),
    ) -> None:
        if type(tolerance) is not ComplexAffineResidualQuadraticTolerance:
            raise TypeError("tolerance must be ComplexAffineResidualQuadraticTolerance")
        self._tolerance = tolerance

    @property
    def tolerance(self) -> ComplexAffineResidualQuadraticTolerance:
        """Return the backend-result relational validation tolerance."""
        return self._tolerance

    @final
    def construct(
        self,
        design_matrix: ComplexAffineResidualMatrix,
        target_vector: ComplexAffineResidualVector,
        /,
        *,
        weights: tuple[float, ...] | None = None,
    ) -> PositiveDefiniteQuadraticLoss:
        """Construct a dense completed-square loss for real coordinates."""
        design = _finite_matrix(design_matrix, "design_matrix")
        targets = _finite_vector(target_vector, "target_vector")
        if len(targets) != len(design):
            raise ValueError("target_vector length must match design_matrix rows")
        normalized_weights = _weights(weights, len(design))
        row_weights = normalized_weights or (1.0,) * len(design)

        result = self._construct_validated(design, targets, normalized_weights)
        if type(result) is not PositiveDefiniteQuadraticLoss:
            raise TypeError("backend must return PositiveDefiniteQuadraticLoss")
        if result.sparse:
            raise ValueError("backend must return a dense quadratic")
        dimension = len(design[0])
        if result.dimension != dimension:
            raise ValueError("backend quadratic dimension is inconsistent")
        if type(result.matrix) is not tuple:
            raise TypeError("backend dense matrix must be a tuple of tuples")

        expected_matrix = _weighted_normal_matrix(design, row_weights)
        actual_matrix = result.matrix
        if any(
            not self.tolerance.accepts(actual, expected)
            for actual_row, expected_row in zip(
                actual_matrix,
                expected_matrix,
                strict=True,
            )
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ):
            raise ValueError("backend quadratic matrix is inconsistent")

        linear = _weighted_linear_vector(design, targets, row_weights)
        for row, expected in enumerate(linear):
            actual = _finite_sum(
                tuple(
                    expected_matrix[row][column] * result.center[column]
                    for column in range(dimension)
                ),
                "stationarity value",
            )
            if not self.tolerance.accepts(actual, expected):
                raise ValueError("backend quadratic center is not stationary")

        expected_minimum = _weighted_squared_loss(
            design,
            targets,
            row_weights,
            result.center,
        )
        if not self.tolerance.accepts(
            result.minimum_squared_loss,
            expected_minimum,
        ):
            raise ValueError("backend quadratic minimum is inconsistent")
        return result

    @abstractmethod
    def _construct_validated(
        self,
        design_matrix: ComplexAffineResidualMatrix,
        target_vector: ComplexAffineResidualVector,
        weights: tuple[float, ...] | None,
        /,
    ) -> PositiveDefiniteQuadraticLoss:
        """Construct from normalized finite inputs with one concrete backend."""
        raise NotImplementedError


def _finite_complex(value: object, name: str, /) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError(f"{name} must contain real or complex numbers excluding bool")
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} entries must be finite complex128") from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} entries must be finite complex128")
    return result


def _finite_matrix(value: object, name: str, /) -> ComplexAffineResidualMatrix:
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


def _finite_vector(value: object, name: str, /) -> ComplexAffineResidualVector:
    if type(value) is not tuple or not value:
        raise TypeError(f"{name} must be a nonempty tuple")
    return tuple(_finite_complex(entry, name) for entry in value)


def _weights(value: object, row_count: int, /) -> tuple[float, ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise TypeError("weights must be None or an exact tuple")
    if len(value) != row_count:
        raise ValueError("weights must contain one value per observation")
    weights = tuple(finite_float(entry, "weight") for entry in value)
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("weights must be strictly positive")
    return weights


def _scaled_complex(value: complex, weight: float, /) -> complex:
    scale = math.sqrt(weight)
    real = scale * value.real
    imaginary = scale * value.imag
    if not math.isfinite(real) or not math.isfinite(imaginary):
        raise ValueError("weighted complex value is nonfinite")
    return complex(real, imaginary)


def _weighted_normal_matrix(
    design: ComplexAffineResidualMatrix,
    weights: tuple[float, ...],
    /,
) -> tuple[tuple[float, ...], ...]:
    dimension = len(design[0])
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    scaled = tuple(
        tuple(_scaled_complex(value, weight) for value in row)
        for row, weight in zip(design, weights, strict=True)
    )
    for first in range(dimension):
        for second in range(first, dimension):
            products = tuple(
                _finite_sum(
                    (
                        row[first].real * row[second].real,
                        row[first].imag * row[second].imag,
                    ),
                    "normal matrix observation",
                )
                for row in scaled
            )
            value = _finite_sum(products, "normal matrix entry")
            matrix[first][second] = value
            matrix[second][first] = value
    return tuple(tuple(row) for row in matrix)


def _weighted_linear_vector(
    design: ComplexAffineResidualMatrix,
    targets: ComplexAffineResidualVector,
    weights: tuple[float, ...],
    /,
) -> tuple[float, ...]:
    dimension = len(design[0])
    scaled_design = tuple(
        tuple(_scaled_complex(value, weight) for value in row)
        for row, weight in zip(design, weights, strict=True)
    )
    scaled_targets = tuple(
        _scaled_complex(target, weight)
        for target, weight in zip(targets, weights, strict=True)
    )
    return tuple(
        _finite_sum(
            tuple(
                _finite_sum(
                    (
                        row[column].real * target.real,
                        row[column].imag * target.imag,
                    ),
                    "linear observation",
                )
                for row, target in zip(
                    scaled_design,
                    scaled_targets,
                    strict=True,
                )
            ),
            "linear entry",
        )
        for column in range(dimension)
    )


def _weighted_squared_loss(
    design: ComplexAffineResidualMatrix,
    targets: ComplexAffineResidualVector,
    weights: tuple[float, ...],
    coordinates: tuple[float, ...],
    /,
) -> float:
    scaled_components: list[float] = []
    for row, target, weight in zip(design, targets, weights, strict=True):
        real = _finite_sum(
            tuple(
                value.real * coordinate
                for value, coordinate in zip(row, coordinates, strict=True)
            )
            + (-target.real,),
            "residual real component",
        )
        imaginary = _finite_sum(
            tuple(
                value.imag * coordinate
                for value, coordinate in zip(row, coordinates, strict=True)
            )
            + (-target.imag,),
            "residual imaginary component",
        )
        scale = math.sqrt(weight)
        scaled_real = scale * real
        scaled_imaginary = scale * imaginary
        if not math.isfinite(scaled_real) or not math.isfinite(scaled_imaginary):
            raise ValueError("weighted residual is nonfinite")
        scaled_components.extend((scaled_real, scaled_imaginary))
    norm = math.hypot(*scaled_components)
    squared = norm * norm
    if not math.isfinite(squared):
        raise ValueError("weighted squared loss is nonfinite")
    return squared


def _finite_sum(values: tuple[float, ...], name: str, /) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} is nonfinite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is nonfinite")
    return result


__all__ = [
    "ComplexAffineResidualMatrix",
    "ComplexAffineResidualQuadraticConstructor",
    "ComplexAffineResidualQuadraticTolerance",
    "ComplexAffineResidualVector",
    "DEFAULT_COMPLEX_AFFINE_RESIDUAL_QUADRATIC_TOLERANCE",
]
