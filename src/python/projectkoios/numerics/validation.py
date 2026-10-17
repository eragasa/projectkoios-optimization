"""Validate exact built-in numerical values and finite arithmetic results.

Notes
-----
These dependency-free primitives own software representation checks only. They assign no
optimization, physical, or scientific meaning to validated values.
"""

from __future__ import annotations

import math
from numbers import Real

from projectkoios.numerics.typing import FiniteComplexMatrix

MAX_U32 = 4_294_967_295
MAX_U64 = 18_446_744_073_709_551_615


def _validate_finite_complex_row(
    row: tuple[complex, ...],
    name: str,
    /,
    *,
    expected_columns: int | None,
    column_error: str,
    arithmetic_result: bool,
) -> int:
    """Validate one exact nonempty finite complex matrix row.

    Parameters
    ----------
    row
        Candidate row.
    name
        Matrix field name included in failure messages.
    expected_columns
        Required row length, or ``None`` for the first row.
    column_error
        Failure message for an inconsistent row length.
    arithmetic_result
        Whether nonfinite entries represent failed arithmetic.

    Returns
    -------
    int
        Validated row length.

    Raises
    ------
    TypeError
        If the row or an entry has the wrong exact runtime type.
    ValueError
        If the row is empty, has the wrong length, or a public entry is nonfinite.
    FloatingPointError
        If an arithmetic-result entry is nonfinite.
    """
    if type(row) is not tuple:
        raise TypeError(f"{name} rows must be exact tuples")
    if not row:
        raise ValueError(f"{name} rows must be nonempty")
    if expected_columns is not None and len(row) != expected_columns:
        raise ValueError(column_error)
    for entry in row:
        if type(entry) is not complex:
            raise TypeError(f"{name} entries must be exact built-in complex values")
        if not math.isfinite(entry.real) or not math.isfinite(entry.imag):
            if arithmetic_result:
                raise FloatingPointError(f"{name} entries must be finite")
            raise ValueError(f"{name} entries must be finite")
    return len(row)


def validate_finite_complex_matrix(
    value: FiniteComplexMatrix,
    name: str,
    /,
    *,
    row_count: int | None = None,
    column_count: int | None = None,
    arithmetic_result: bool = False,
) -> FiniteComplexMatrix:
    """Validate an exact nonempty finite complex tuple matrix.

    Parameters
    ----------
    value
        Candidate immutable matrix.
    name
        Field name included in failure messages.
    row_count, column_count
        Optional required dimensions.
    arithmetic_result
        Whether nonfinite entries represent failed numerical arithmetic.

    Returns
    -------
    FiniteComplexMatrix
        Original validated matrix.

    Raises
    ------
    TypeError
        If containers or entries do not use exact runtime types.
    ValueError
        If shape is empty, ragged, inconsistent, or a public value is nonfinite.
    FloatingPointError
        If an arithmetic result contains a nonfinite entry.
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    if row_count is not None and len(value) != row_count:
        raise ValueError(f"{name} row count is inconsistent")
    inferred_columns: int | None = None
    for row in value:
        required_columns = column_count
        column_error = f"{name} column count is inconsistent"
        if required_columns is None:
            required_columns = inferred_columns
            column_error = f"{name} must be rectangular"
        validated_columns = _validate_finite_complex_row(
            row,
            name,
            expected_columns=required_columns,
            column_error=column_error,
            arithmetic_result=arithmetic_result,
        )
        if inferred_columns is None:
            inferred_columns = validated_columns
    return value


def finite_float(value: object, name: str, /) -> float:
    """Normalize a non-Boolean real value to finite built-in binary64.

    Parameters
    ----------
    value
        Candidate real numeric value.
    name
        Field name included in failure messages.

    Returns
    -------
    float
        Finite built-in binary64 representation.

    Raises
    ------
    TypeError
        If the value is Boolean or is not a real number.
    ValueError
        If conversion fails, overflows, or produces a nonfinite result.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number excluding bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if type(result) is not float or not math.isfinite(result):
        raise ValueError(f"{name} must be representable as a finite float")
    return result


def integer(value: object, name: str, /) -> int:
    """Validate an exact built-in integer while excluding Boolean values.

    Parameters
    ----------
    value
        Candidate integer.
    name
        Field name included in failure messages.

    Returns
    -------
    int
        Original validated integer.

    Raises
    ------
    TypeError
        If the value is not exactly a built-in integer.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer excluding bool")
    return value


def u32(value: object, name: str, /, *, allow_zero: bool) -> int:
    """Validate an exact integer in a requested unsigned 32-bit interval.

    Parameters
    ----------
    value
        Candidate integer.
    name
        Field name included in failure messages.
    allow_zero
        Whether zero belongs to the accepted interval.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    TypeError
        If the value is not exactly a built-in integer.
    ValueError
        If the value lies outside the requested interval.
    """
    result = integer(value, name)
    minimum = 0 if allow_zero else 1
    if not minimum <= result <= MAX_U32:
        interval = "unsigned u32" if allow_zero else "positive u32"
        raise ValueError(f"{name} must be in the {interval} range")
    return result


def u64(value: object, name: str, /) -> int:
    """Validate an exact integer in the unsigned 64-bit interval.

    Parameters
    ----------
    value
        Candidate integer.
    name
        Field name included in failure messages.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    TypeError
        If the value is not exactly a built-in integer.
    ValueError
        If the value lies outside the unsigned 64-bit interval.
    """
    result = integer(value, name)
    if not 0 <= result <= MAX_U64:
        raise ValueError(f"{name} must be in the unsigned u64 range")
    return result


def exact_positive_integer(value: int, name: str, /) -> int:
    """Validate a positive exact built-in integer.

    Parameters
    ----------
    value
        Candidate integer.
    name
        Field name included in failure messages.

    Returns
    -------
    int
        Original validated integer.

    Raises
    ------
    TypeError
        If the value is not an exact built-in integer.
    ValueError
        If the integer is not positive.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def exact_finite_float(value: float, name: str, /) -> float:
    """Validate a finite exact built-in float without coercion.

    Parameters
    ----------
    value
        Candidate binary64 value.
    name
        Field name included in failure messages.

    Returns
    -------
    float
        Original validated float.

    Raises
    ------
    TypeError
        If the value is not an exact built-in float.
    ValueError
        If the value is nonfinite.
    """
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_finite_float_tuple(
    value: tuple[float, ...],
    name: str,
    /,
) -> tuple[float, ...]:
    """Validate a nonempty exact tuple of exact finite built-in floats.

    Parameters
    ----------
    value
        Candidate immutable float vector.
    name
        Field name included in failure messages.

    Returns
    -------
    tuple[float, ...]
        Original validated tuple without copying.

    Raises
    ------
    TypeError
        If the container or an entry has the wrong exact runtime type.
    ValueError
        If the tuple is empty or an entry is nonfinite.
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    for index, entry in enumerate(value):
        exact_finite_float(entry, f"{name}[{index}]")
    return value


def exact_nonnegative_float(value: float, name: str, /) -> float:
    """Validate a finite nonnegative exact built-in float.

    Parameters
    ----------
    value
        Candidate binary64 value.
    name
        Field name included in failure messages.

    Returns
    -------
    float
        Original validated float.

    Raises
    ------
    TypeError
        If the value is not an exact built-in float.
    ValueError
        If the value is nonfinite or negative.
    """
    result = exact_finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def exact_relative_float(value: float, name: str, /) -> float:
    """Validate an exact built-in float in the half-open interval ``[0, 1)``.

    Parameters
    ----------
    value
        Candidate relative numerical tolerance.
    name
        Field name included in failure messages.

    Returns
    -------
    float
        Original validated float.

    Raises
    ------
    TypeError
        If the value is not an exact built-in float.
    ValueError
        If the value is nonfinite or lies outside ``[0, 1)``.
    """
    result = exact_nonnegative_float(value, name)
    if result >= 1.0:
        raise ValueError(f"{name} must be less than 1.0")
    return result


def finite_float_result(value: float, message: str, /) -> float:
    """Return one finite built-in float arithmetic result.

    Parameters
    ----------
    value
        Candidate arithmetic result.
    message
        Failure message describing the inducing numerical operation.

    Returns
    -------
    float
        Original finite built-in value.

    Raises
    ------
    TypeError
        If the value is not an exact built-in float.
    FloatingPointError
        If the value is NaN or infinite.
    """
    if type(value) is not float:
        raise TypeError("arithmetic result must be an exact built-in float")
    if not math.isfinite(value):
        raise FloatingPointError(message)
    return value


def finite_complex_result(value: complex, message: str, /) -> complex:
    """Return one finite built-in complex arithmetic result.

    Parameters
    ----------
    value
        Candidate arithmetic result.
    message
        Failure message describing the inducing numerical operation.

    Returns
    -------
    complex
        Original finite built-in value.

    Raises
    ------
    TypeError
        If the value is not an exact built-in complex value.
    FloatingPointError
        If either component is NaN or infinite.
    """
    if type(value) is not complex:
        raise TypeError("arithmetic result must be an exact built-in complex value")
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError(message)
    return value


__all__ = [
    "MAX_U32",
    "MAX_U64",
    "exact_finite_float",
    "exact_nonnegative_float",
    "exact_positive_integer",
    "exact_relative_float",
    "validate_finite_complex_matrix",
    "finite_complex_result",
    "finite_float",
    "finite_float_result",
    "integer",
    "u32",
    "u64",
    "validate_finite_float_tuple",
]
