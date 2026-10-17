import pytest

from projectkoios.optimization.selection.finite import FiniteMinimum, first_minimum

pytestmark = pytest.mark.unit


def test_first_minimum_preserves_first_exact_tie() -> None:
    result = first_minimum((3.0, -1.0, 2.0, -1.0))

    assert result == FiniteMinimum(1, -1.0)


def test_finite_minimum_validates_index_and_value() -> None:
    assert FiniteMinimum(0, 1.0) == FiniteMinimum(0, 1.0)
    with pytest.raises(TypeError, match="exact built-in float"):
        FiniteMinimum(0, 1)

    with pytest.raises(TypeError, match="index"):
        FiniteMinimum(True, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        FiniteMinimum(-1, 1.0)
    with pytest.raises(ValueError, match="finite"):
        FiniteMinimum(0, float("nan"))
