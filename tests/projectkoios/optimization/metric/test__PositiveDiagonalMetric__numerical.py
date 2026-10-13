import pytest

from projectkoios.optimization.metric.metric import PositiveDiagonalMetric

pytestmark = pytest.mark.numerical


def test_preserves_representable_compensated_scales() -> None:
    primal = PositiveDiagonalMetric((1.0e300,))
    dual = PositiveDiagonalMetric((1.0e-300,))
    large_dual = PositiveDiagonalMetric((1.0e308,))

    assert primal.squared_distance((1.0e-200,), (0.0,)) == pytest.approx(
        1.0e-100, rel=5.0e-15
    )
    assert dual.squared_dual_norm((1.0e-200,)) == pytest.approx(1.0e-100, rel=5.0e-15)
    assert large_dual.squared_dual_norm((1.0e308,)) == pytest.approx(
        1.0e308, rel=5.0e-15
    )


def test_rescales_subnormal_squares_and_rejects_overflow() -> None:
    primal = PositiveDiagonalMetric((1.0e300,))
    dual = PositiveDiagonalMetric((1.0e-300,))

    assert primal.squared_distance((1.0e-161,), (0.0,)) == pytest.approx(
        1.0e-22, rel=5.0e-15
    )
    assert dual.squared_dual_norm((1.0e-161,)) == pytest.approx(1.0e-22, rel=5.0e-15)
    with pytest.raises(ValueError, match="nonfinite"):
        PositiveDiagonalMetric((1.0e308,)).squared_distance((1.0e308,), (0.0,))
