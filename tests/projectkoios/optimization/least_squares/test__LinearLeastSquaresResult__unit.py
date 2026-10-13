import math

import pytest

from projectkoios.optimization.least_squares.linear import (
    DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE,
    LinearLeastSquaresResult,
    LinearLeastSquaresTolerance,
    _scaled_absolute_tolerance,
)

pytestmark = pytest.mark.unit


def _result(**changes: object) -> LinearLeastSquaresResult:
    values: dict[str, object] = {
        "coefficients": ((1.0 + 0.0j,), (2.0 + 0.0j,)),
        "fitted_values": ((1.0 + 0.0j,), (2.0 + 0.0j,), (3.0 + 0.0j,)),
        "residuals": ((0.0 + 0.0j,), (0.0 + 0.0j,), (0.0 + 0.0j,)),
        "residual_norms": (0.0,),
        "weighted_residual_norms": (0.0,),
        "residual_frobenius_norm": 0.0,
        "weighted_residual_frobenius_norm": 0.0,
        "maximum_row_residual_norm": 0.0,
        "rank": 2,
        "singular_values": (2.0, 1.0),
        "full_rank_condition_number": 2.0,
    }
    values.update(changes)
    return LinearLeastSquaresResult(**values)  # type: ignore[arg-type]


def test_tolerance_is_scale_aware_and_rejects_invalid_values() -> None:
    tolerance = LinearLeastSquaresTolerance(absolute=1.0e-6, relative=1.0e-3)

    assert tolerance.accepts(1000.0 + 2.0j, 1000.5 + 2.0j)
    assert not tolerance.accepts(1000.0 + 2.0j, 1002.0 + 2.0j)
    assert tolerance.accepts_zero(1.0e-5 + 0.0j, 1.0)
    assert LinearLeastSquaresTolerance() == DEFAULT_LINEAR_LEAST_SQUARES_TOLERANCE

    with pytest.raises(TypeError, match="absolute"):
        LinearLeastSquaresTolerance(absolute=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        LinearLeastSquaresTolerance(relative=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        LinearLeastSquaresTolerance(absolute=0.0, relative=0.0)


def test_scaled_absolute_tolerance_retains_smallest_subnormal_boundary() -> None:
    smallest = math.ulp(0.0)

    assert _scaled_absolute_tolerance(smallest, 1.0, 1.0) == smallest


def test_result_normalizes_numeric_state() -> None:
    result = _result()

    assert result.coefficients == ((1.0 + 0.0j,), (2.0 + 0.0j,))
    assert result.rank == 2
    assert result.full_rank_condition_number == 2.0


def test_result_enforces_shapes_rank_singular_values_and_condition() -> None:
    with pytest.raises(ValueError, match="target-column count"):
        _result(coefficients=((1.0 + 0.0j, 2.0 + 0.0j),))
    with pytest.raises(ValueError, match="rank bound"):
        _result(rank=3)
    with pytest.raises(ValueError, match="nonincreasing"):
        _result(singular_values=(1.0, 2.0))
    with pytest.raises(ValueError, match="only at full column rank"):
        _result(rank=1, full_rank_condition_number=2.0)
    with pytest.raises(TypeError, match="full_rank_condition_number"):
        _result(full_rank_condition_number=None)
