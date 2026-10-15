"""Dependency-free Frobenius power and norm for finite complex matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

type FiniteComplexMatrix = tuple[tuple[complex, ...], ...]


def _positive_integer(value: object, name: str, /) -> int:
    """Validate a positive exact built-in integer.

    Parameters
    ----------
    value
        Candidate value.
    name
        Public field name used in failure messages.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in integer.
    ValueError
        If ``value`` is not positive.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(value: object, name: str, /) -> float:
    """Validate a finite nonnegative exact built-in float.

    Parameters
    ----------
    value
        Candidate value.
    name
        Public field name used in failure messages.

    Returns
    -------
    float
        The validated float.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in float.
    ValueError
        If ``value`` is nonfinite or negative.
    """
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _matrix(value: object, name: str, /) -> FiniteComplexMatrix:
    """Validate a nonempty rectangular finite complex tuple matrix.

    Parameters
    ----------
    value
        Candidate matrix.
    name
        Public field name used in failure messages.

    Returns
    -------
    FiniteComplexMatrix
        Validated immutable matrix in original row-major order.

    Raises
    ------
    TypeError
        If the container, rows, or entries do not use exact public runtime types.
    ValueError
        If the matrix is empty, ragged, or contains a nonfinite entry.
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    column_count: int | None = None
    normalized: list[tuple[complex, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError(f"{name} rows must be exact tuples")
        if not row:
            raise ValueError(f"{name} rows must be nonempty")
        if column_count is None:
            column_count = len(row)
        elif len(row) != column_count:
            raise ValueError(f"{name} must be rectangular")
        normalized_row: list[complex] = []
        for entry in row:
            if type(entry) is not complex:
                raise TypeError(f"{name} entries must be exact built-in complex values")
            if not math.isfinite(entry.real) or not math.isfinite(entry.imag):
                raise ValueError(f"{name} entries must be finite")
            normalized_row.append(entry)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _power(matrix: FiniteComplexMatrix, /) -> float:
    """Compute a deterministic binary64 Frobenius power.

    Parameters
    ----------
    matrix
        Validated finite complex matrix.

    Returns
    -------
    float
        Row-major ``math.fsum`` of squared entry magnitudes.

    Raises
    ------
    FloatingPointError
        If an individual squared magnitude or the accumulated raw power becomes
        nonfinite.
    """
    terms: list[float] = []
    # A scaled norm could preserve the square root for larger inputs, but it would hide
    # overflow of the public raw-power value. Each term therefore fails independently.
    for row in matrix:
        for entry in row:
            magnitude = abs(entry)
            power = magnitude * magnitude
            if not math.isfinite(power):
                raise FloatingPointError("Frobenius power term became nonfinite")
            terms.append(power)
    # Preserve row-major term order and use the standard library's reproducible,
    # compensated-style summation rather than backend-dependent array reduction.
    try:
        result = math.fsum(terms)
    except OverflowError as error:
        raise FloatingPointError("Frobenius power became nonfinite") from error
    if not math.isfinite(result):
        raise FloatingPointError("Frobenius power became nonfinite")
    return result


@final
@dataclass(frozen=True, slots=True)
class ComplexMatrixFrobenius:
    """Record finite Frobenius power and its nonnegative square root.

    Parameters
    ----------
    row_count, column_count
        Positive matrix dimensions.
    power
        Raw squared Frobenius norm ``sum(abs(entry)**2)``.
    norm
        Exact binary64 ``sqrt(power)``.
    """

    row_count: int
    column_count: int
    power: float
    norm: float

    def __post_init__(self) -> None:
        """Validate dimensions, finite values, and the square-root relation.

        Raises
        ------
        TypeError
            If a field violates its exact public runtime type.
        ValueError
            If dimensions are nonpositive, values are nonfinite or negative, or
            ``norm`` does not equal the binary64 square root of ``power``.
        """
        row_count = _positive_integer(self.row_count, "row_count")
        column_count = _positive_integer(self.column_count, "column_count")
        power = _nonnegative_float(self.power, "power")
        norm = _nonnegative_float(self.norm, "norm")
        if norm != math.sqrt(power):
            raise ValueError("norm must equal sqrt(power)")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "column_count", column_count)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "norm", norm)


def complex_matrix_frobenius(
    matrix: FiniteComplexMatrix,
    /,
) -> ComplexMatrixFrobenius:
    """Compute Frobenius power and norm for a finite complex matrix.

    Parameters
    ----------
    matrix
        Nonempty rectangular matrix represented by exact built-in tuples and finite
        exact built-in complex entries.

    Returns
    -------
    ComplexMatrixFrobenius
        Matrix dimensions, raw Frobenius power, and its binary64 square root.

    Raises
    ------
    TypeError
        If containers or entries do not use exact public runtime types.
    ValueError
        If the matrix is empty, ragged, or contains a nonfinite entry.
    FloatingPointError
        If a finite entry's squared magnitude or the accumulated raw power overflows or
        otherwise becomes nonfinite.

    Notes
    -----
    The capability deliberately requires a finite raw power. It therefore rejects some
    matrices whose norm could be represented by a scaled norm algorithm but whose raw
    squared Frobenius norm cannot be represented in binary64.
    """
    normalized = _matrix(matrix, "matrix")
    power = _power(normalized)
    return ComplexMatrixFrobenius(
        len(normalized),
        len(normalized[0]),
        power,
        math.sqrt(power),
    )


__all__ = [
    "ComplexMatrixFrobenius",
    "FiniteComplexMatrix",
    "complex_matrix_frobenius",
]
