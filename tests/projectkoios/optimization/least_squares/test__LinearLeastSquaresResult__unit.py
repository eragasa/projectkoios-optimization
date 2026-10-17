"""Unit tests for linear least-squares tolerance and result records."""

import pytest

from projectkoios.optimization.least_squares.linear import (
    DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE,
    LinearLeastSquaresResult,
    LinearLeastSquaresTolerance,
)

pytestmark = pytest.mark.unit


def _exact_full_rank_result() -> LinearLeastSquaresResult:
    return LinearLeastSquaresResult(
        coefficients=((1.0 + 0.0j,), (2.0 + 0.0j,)),
        fitted_values=((1.0 + 0.0j,), (2.0 + 0.0j,), (3.0 + 0.0j,)),
        residuals=((0.0 + 0.0j,), (0.0 + 0.0j,), (0.0 + 0.0j,)),
        residual_norms=(0.0,),
        weighted_residual_norms=(0.0,),
        residual_frobenius_norm=0.0,
        weighted_residual_frobenius_norm=0.0,
        maximum_row_residual_norm=0.0,
        rank=2,
        singular_values=(2.0, 1.0),
        full_rank_condition_number=2.0,
    )


def test_tolerance_is_scale_aware_and_rejects_invalid_values() -> None:
    tolerance = LinearLeastSquaresTolerance(absolute=1.0e-6, relative=1.0e-3)

    assert tolerance.accepts(1000.0 + 2.0j, 1000.5 + 2.0j)
    assert not tolerance.accepts(1000.0 + 2.0j, 1002.0 + 2.0j)
    assert tolerance.accepts_zero(1.0e-5 + 0.0j, 1.0)
    assert LinearLeastSquaresTolerance() == DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE

    with pytest.raises(TypeError):
        LinearLeastSquaresTolerance(absolute=True)
    with pytest.raises(ValueError):
        LinearLeastSquaresTolerance(relative=-1.0)
    with pytest.raises(ValueError):
        LinearLeastSquaresTolerance(absolute=0.0, relative=0.0)


def test_full_rank_result_retains_canonical_numeric_state() -> None:
    result = _exact_full_rank_result()

    assert result.coefficients == ((1.0 + 0.0j,), (2.0 + 0.0j,))
    assert result.rank == 2
    assert result.full_rank_condition_number == 2.0


def test_constructor_rejects_inconsistent_target_column_count() -> None:
    with pytest.raises(ValueError):
        LinearLeastSquaresResult(
            coefficients=((1.0 + 0.0j, 2.0 + 0.0j),),
            fitted_values=((1.0 + 0.0j,),),
            residuals=((0.0 + 0.0j,),),
            residual_norms=(0.0,),
            weighted_residual_norms=(0.0,),
            residual_frobenius_norm=0.0,
            weighted_residual_frobenius_norm=0.0,
            maximum_row_residual_norm=0.0,
            rank=1,
            singular_values=(1.0,),
            full_rank_condition_number=1.0,
        )


def test_constructor_rejects_rank_above_matrix_bound() -> None:
    with pytest.raises(ValueError):
        LinearLeastSquaresResult(
            coefficients=((1.0 + 0.0j,), (2.0 + 0.0j,)),
            fitted_values=((1.0 + 0.0j,), (2.0 + 0.0j,), (3.0 + 0.0j,)),
            residuals=((0.0 + 0.0j,), (0.0 + 0.0j,), (0.0 + 0.0j,)),
            residual_norms=(0.0,),
            weighted_residual_norms=(0.0,),
            residual_frobenius_norm=0.0,
            weighted_residual_frobenius_norm=0.0,
            maximum_row_residual_norm=0.0,
            rank=3,
            singular_values=(2.0, 1.0),
            full_rank_condition_number=2.0,
        )


def test_rank_deficient_result_forbids_full_rank_condition_number() -> None:
    with pytest.raises(ValueError):
        LinearLeastSquaresResult(
            coefficients=((1.0 + 0.0j,), (2.0 + 0.0j,)),
            fitted_values=((1.0 + 0.0j,), (2.0 + 0.0j,), (3.0 + 0.0j,)),
            residuals=((0.0 + 0.0j,), (0.0 + 0.0j,), (0.0 + 0.0j,)),
            residual_norms=(0.0,),
            weighted_residual_norms=(0.0,),
            residual_frobenius_norm=0.0,
            weighted_residual_frobenius_norm=0.0,
            maximum_row_residual_norm=0.0,
            rank=1,
            singular_values=(2.0, 1.0),
            full_rank_condition_number=2.0,
        )
