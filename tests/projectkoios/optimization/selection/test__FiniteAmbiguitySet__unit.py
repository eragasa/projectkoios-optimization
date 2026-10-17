import pytest

from projectkoios.optimization.selection.finite import (
    FiniteAmbiguitySet,
    FiniteMinimum,
    absolute_ambiguity_set,
)

pytestmark = pytest.mark.unit


def test_zero_tolerance_retains_all_exact_minima_in_order() -> None:
    result = absolute_ambiguity_set((2.0, 1.0, 3.0, 1.0), 0.0)

    assert result.minimum == FiniteMinimum(1, 1.0)
    assert result.indices == (1, 3)


def test_nonzero_tolerance_retains_complete_ordered_set() -> None:
    result = absolute_ambiguity_set((1.25, 1.0, 1.125, 2.0), 0.125)

    assert result.indices == (1, 2)
    assert result.tolerance == 0.125


def test_ambiguity_record_requires_ordered_indices_containing_minimum() -> None:
    minimum = FiniteMinimum(1, 0.0)
    canonical = FiniteAmbiguitySet(minimum, 0.0, (1,))
    assert type(canonical.tolerance) is float
    with pytest.raises(TypeError, match="exact built-in float"):
        FiniteAmbiguitySet(minimum, 0, (1,))

    with pytest.raises(ValueError, match="strictly increasing"):
        FiniteAmbiguitySet(minimum, 0.0, (1, 0))
    with pytest.raises(ValueError, match="contain"):
        FiniteAmbiguitySet(minimum, 0.0, (0, 2))
    with pytest.raises(TypeError, match="nonempty tuple"):
        FiniteAmbiguitySet(minimum, 0.0, ())
    with pytest.raises(ValueError, match="nonnegative"):
        FiniteAmbiguitySet(minimum, -1.0, (1,))
