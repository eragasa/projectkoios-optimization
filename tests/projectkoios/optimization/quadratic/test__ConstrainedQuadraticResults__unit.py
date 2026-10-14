import pytest

from projectkoios.optimization.quadratic.constrained import (
    DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE,
    ConstrainedQuadraticTolerance,
    QuadraticSublevelSupportResult,
    RectangleQuadraticCandidate,
    RectangleQuadraticMinimum,
)

pytestmark = pytest.mark.unit


def test_constrained_tolerance_is_scale_aware_and_rejects_invalid_values() -> None:
    tolerance = ConstrainedQuadraticTolerance(absolute=1.0e-6, relative=1.0e-3)

    assert tolerance.accepts(1000.0, 1000.5)
    assert not tolerance.accepts(1000.0, 1002.0)
    assert ConstrainedQuadraticTolerance() == DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE

    with pytest.raises(TypeError, match="absolute"):
        ConstrainedQuadraticTolerance(absolute=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absolute"):
        ConstrainedQuadraticTolerance(absolute=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        ConstrainedQuadraticTolerance(absolute=0.0, relative=0.0)


def test_rectangle_minimum_normalizes_finite_state() -> None:
    result = RectangleQuadraticMinimum(
        (1, -2),
        0,
        RectangleQuadraticCandidate.COORDINATE_0_LOWER,
    )

    assert result.coordinates == (1.0, -2.0)
    assert result.squared_loss == 0.0

    with pytest.raises(ValueError, match="exactly two"):
        RectangleQuadraticMinimum(
            (1.0,),  # type: ignore[arg-type]
            0.0,
            RectangleQuadraticCandidate.CLIPPED_CENTER,
        )
    with pytest.raises(TypeError, match="candidate"):
        RectangleQuadraticMinimum((1.0, 2.0), 0.0, "edge")  # type: ignore[arg-type]


def test_support_result_requires_matching_nonzero_direction() -> None:
    result = QuadraticSublevelSupportResult(
        (1, 2),
        (0, 1),
        2,
        3,
    )

    assert result.point == (1.0, 2.0)
    assert result.direction == (0.0, 1.0)
    assert result.support_value == 2.0
    assert result.squared_loss == 3.0

    with pytest.raises(ValueError, match="dimensions"):
        QuadraticSublevelSupportResult((1.0,), (1.0, 0.0), 1.0, 0.0)
    with pytest.raises(ValueError, match="nonzero"):
        QuadraticSublevelSupportResult((1.0, 2.0), (0.0, 0.0), 0.0, 0.0)
