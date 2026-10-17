from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import pytest

from projectkoios.numerics.typing import FiniteComplexMatrix
from projectkoios.numerics.validation import (
    exact_finite_float,
    exact_nonnegative_float,
    exact_positive_integer,
    exact_relative_float,
    finite_complex_result,
    finite_float,
    finite_float_result,
    integer,
    u32,
    u64,
    validate_finite_complex_matrix,
    validate_finite_float_tuple,
)


@pytest.fixture
def finite_complex_row_matrix() -> FiniteComplexMatrix:
    """Return a named finite rectangular matrix fixture."""
    real_unit = complex(1.0, 0.0)
    imaginary_unit = complex(0.0, 1.0)
    return ((real_unit, imaginary_unit),)


@pytest.fixture
def nonfinite_complex_matrix() -> FiniteComplexMatrix:
    """Return a named matrix fixture containing positive infinity."""
    return ((complex(math.inf, 0.0),),)


def test_finite_complex_matrix_validation_preserves_exact_immutable_input(
    finite_complex_row_matrix: FiniteComplexMatrix,
) -> None:
    validated = validate_finite_complex_matrix(
        finite_complex_row_matrix,
        "matrix",
    )

    assert validated is finite_complex_row_matrix


def test_finite_complex_matrix_validation_rejects_required_column_mismatch(
    finite_complex_row_matrix: FiniteComplexMatrix,
) -> None:
    actual_column_count = len(finite_complex_row_matrix[0])
    inconsistent_column_count = actual_column_count + 1

    with pytest.raises(ValueError, match="column count"):
        validate_finite_complex_matrix(
            finite_complex_row_matrix,
            "matrix",
            column_count=inconsistent_column_count,
        )


def test_finite_complex_matrix_validation_classifies_nonfinite_public_input(
    nonfinite_complex_matrix: FiniteComplexMatrix,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_finite_complex_matrix(nonfinite_complex_matrix, "matrix")


def test_finite_complex_matrix_validation_classifies_nonfinite_arithmetic(
    nonfinite_complex_matrix: FiniteComplexMatrix,
) -> None:
    with pytest.raises(FloatingPointError, match="finite"):
        validate_finite_complex_matrix(
            nonfinite_complex_matrix,
            "matrix",
            arithmetic_result=True,
        )


def test_finite_float_normalizes_a_rational_value() -> None:
    rational_value = Fraction(1, 2)

    assert finite_float(rational_value, "value") == float(rational_value)


def test_integer_validates_an_exact_builtin_integer() -> None:
    candidate_count = 3

    assert integer(candidate_count, "count") == candidate_count


def test_unsigned_integer_validators_accept_their_lower_bounds() -> None:
    minimum_positive_u32 = 1
    minimum_u64 = 0

    assert u32(minimum_positive_u32, "count", allow_zero=False) == minimum_positive_u32
    assert u64(minimum_u64, "seed") == minimum_u64


def test_numeric_boundaries_reject_boolean_and_out_of_range_values() -> None:
    minimum_positive_u32 = 1
    minimum_u64 = 0

    with pytest.raises(TypeError):
        finite_float(True, "value")
    with pytest.raises(TypeError):
        integer(True, "count")
    with pytest.raises(ValueError):
        u32(minimum_positive_u32 - 1, "count", allow_zero=False)
    with pytest.raises(ValueError):
        u64(minimum_u64 - 1, "seed")


@pytest.mark.parametrize(
    "value", [pytest.param(1, id="one"), pytest.param(7, id="seven")]
)
def test_exact_positive_integer_returns_original_value(value: int) -> None:
    assert exact_positive_integer(value, "value") == value


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        pytest.param(0, ValueError, "positive", id="zero"),
        pytest.param(-1, ValueError, "positive", id="negative"),
        pytest.param(True, TypeError, "exact built-in integer", id="boolean"),
    ],
)
def test_exact_positive_integer_rejects_invalid_values(
    value: int,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        exact_positive_integer(value, "value")


@pytest.mark.parametrize(
    "value",
    [pytest.param(-2.0, id="negative"), pytest.param(0.0, id="zero")],
)
def test_exact_finite_float_returns_original_value(value: float) -> None:
    assert exact_finite_float(value, "value") == value


def test_exact_finite_float_rejects_wrong_type_and_nonfinite_value() -> None:
    with pytest.raises(TypeError, match="exact built-in float"):
        exact_finite_float(cast(float, 1), "value")
    with pytest.raises(ValueError, match="finite"):
        exact_finite_float(math.inf, "value")


def test_finite_float_tuple_validation_preserves_canonical_input() -> None:
    values = (-1.0, 0.0, 2.0)

    assert validate_finite_float_tuple(values, "values") is values


def test_finite_float_tuple_validation_rejects_coercion_and_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="exact built-in float"):
        validate_finite_float_tuple(cast(tuple[float, ...], (1,)), "values")
    with pytest.raises(ValueError, match="finite"):
        validate_finite_float_tuple((math.inf,), "values")


@pytest.mark.parametrize(
    "value",
    [pytest.param(0.0, id="zero"), pytest.param(2.0, id="positive")],
)
def test_exact_nonnegative_float_returns_original_value(value: float) -> None:
    assert exact_nonnegative_float(value, "value") == value


def test_exact_nonnegative_float_rejects_wrong_exact_type() -> None:
    with pytest.raises(TypeError, match="exact built-in float"):
        exact_nonnegative_float(cast(float, 1), "value")


@pytest.mark.parametrize(
    "value",
    [pytest.param(-1.0, id="negative"), pytest.param(math.inf, id="infinite")],
)
def test_exact_nonnegative_float_rejects_invalid_values(value: float) -> None:
    match = "finite" if not math.isfinite(value) else "nonnegative"
    with pytest.raises(ValueError, match=match):
        exact_nonnegative_float(value, "value")


@pytest.mark.parametrize(
    "value",
    [pytest.param(0.0, id="zero"), pytest.param(0.5, id="interior")],
)
def test_exact_relative_float_returns_original_value(value: float) -> None:
    assert exact_relative_float(value, "value") == value


def test_exact_relative_float_rejects_upper_bound() -> None:
    with pytest.raises(ValueError, match="less than 1.0"):
        exact_relative_float(1.0, "value")


def test_finite_result_validators_return_original_values() -> None:
    assert finite_float_result(1.0, "nonfinite") == 1.0
    assert finite_complex_result(1.0 + 2.0j, "nonfinite") == 1.0 + 2.0j


def test_finite_result_validators_reject_nonfinite_values() -> None:
    with pytest.raises(FloatingPointError, match="nonfinite"):
        finite_float_result(math.inf, "nonfinite")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        finite_complex_result(complex(math.nan, 0.0), "nonfinite")
