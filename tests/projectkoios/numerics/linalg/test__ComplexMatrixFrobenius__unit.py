"""Unit tests for dependency-free complex-matrix Frobenius values."""

from __future__ import annotations

from typing import Any, cast

import pytest

import projectkoios.numerics.linalg.frobenius as frobenius
from projectkoios.numerics.linalg.frobenius import (
    ComplexMatrixFrobenius,
    complex_matrix_frobenius,
)

pytestmark = pytest.mark.unit


def test_does_not_reexport_shared_matrix_representation() -> None:
    assert "FiniteComplexMatrix" not in frobenius.__all__
    assert not hasattr(frobenius, "FiniteComplexMatrix")


def test_evaluates_rectangular_complex_matrix_with_stable_power_sum() -> None:
    result = complex_matrix_frobenius(
        (
            (3.0 + 4.0j, 0.0 + 0.0j, 1.0 - 1.0j),
            (-2.0 + 0.0j, 0.0 + 1.0j, 2.0 - 2.0j),
        )
    )

    assert result == ComplexMatrixFrobenius(2, 3, 40.0, 40.0**0.5)


def test_zero_matrix_has_exact_zero_power_and_norm() -> None:
    result = complex_matrix_frobenius(((0.0 + 0.0j, 0.0 + 0.0j),))

    assert result.power == 0.0
    assert result.norm == 0.0


@pytest.mark.parametrize(
    "matrix, error, match",
    [
        ([], TypeError, "exact tuple"),
        ((), ValueError, "nonempty"),
        (((1.0 + 0.0j,), []), TypeError, "rows must be exact tuples"),
        (((1.0 + 0.0j,), ()), ValueError, "rows must be nonempty"),
        (
            ((1.0 + 0.0j,), (1.0 + 0.0j, 2.0 + 0.0j)),
            ValueError,
            "rectangular",
        ),
        (((1.0,),), TypeError, "exact built-in complex"),
        (((complex(float("nan"), 0.0),),), ValueError, "finite"),
        (((complex(0.0, float("inf")),),), ValueError, "finite"),
    ],
)
def test_rejects_malformed_matrix(
    matrix: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        complex_matrix_frobenius(cast(Any, matrix))


def test_finite_input_power_overflow_fails_closed() -> None:
    with pytest.raises(FloatingPointError, match="power term became nonfinite"):
        complex_matrix_frobenius(((complex(1.0e308, 0.0),),))


def test_finite_power_sum_overflow_fails_closed() -> None:
    value = complex(1.0e154, 0.0)

    with pytest.raises(FloatingPointError, match="power became nonfinite"):
        complex_matrix_frobenius(((value, value),))


@pytest.mark.parametrize(
    "arguments, error, match",
    [
        ((True, 1, 1.0, 1.0), TypeError, "row_count"),
        ((1, 0, 1.0, 1.0), ValueError, "column_count"),
        ((1, 1, float("inf"), 1.0), ValueError, "power must be finite"),
        ((1, 1, -1.0, 1.0), ValueError, "power must be nonnegative"),
        ((1, 1, 4.0, 3.0), ValueError, "sqrt"),
    ],
)
def test_result_rejects_malformed_or_inconsistent_fields(
    arguments: tuple[object, object, object, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ComplexMatrixFrobenius(*cast(tuple[int, int, float, float], arguments))
