"""Dependency-free Frobenius power and norm for finite complex matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

from projectkoios.numerics.typing import FiniteComplexMatrix
from projectkoios.numerics.validation import (
    exact_nonnegative_float,
    exact_positive_integer,
    validate_finite_complex_matrix,
)


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
        exact_positive_integer(self.row_count, "row_count")
        exact_positive_integer(self.column_count, "column_count")
        exact_nonnegative_float(self.power, "power")
        exact_nonnegative_float(self.norm, "norm")
        if self.norm != math.sqrt(self.power):
            raise ValueError("norm must equal sqrt(power)")


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
    normalized = validate_finite_complex_matrix(matrix, "matrix")
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
