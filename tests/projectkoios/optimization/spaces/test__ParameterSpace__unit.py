import pytest

from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


def test_parameter_bound_requires_canonical_frozen_construction() -> None:
    bound = ParameterBound("x", -1.0, 2.0)

    assert bound.lower == -1.0
    assert bound.upper == 2.0
    assert type(bound.lower) is float
    assert type(bound.upper) is float
    with pytest.raises(TypeError, match="exact built-in float"):
        ParameterBound("x", -1, 2.0)


def test_parameter_space_supports_empty_fixed_and_bounded_coordinates() -> None:
    empty = ParameterSpace(())
    space = ParameterSpace(
        (ParameterBound("fixed", 2.0, 2.0), ParameterBound("x", -1.0, 3.0))
    )

    assert empty.dimension == 0
    assert empty.admits(()) is True
    assert space.names == ("fixed", "x")
    assert space.admits((2.0, 0.5)) is True
    assert space.admits((2.0, 4.0)) is False


def test_parameter_space_rejects_duplicate_names_and_invalid_widths() -> None:
    with pytest.raises(ValueError, match="unique"):
        ParameterSpace((ParameterBound("x", 0.0, 1.0), ParameterBound("x", 1.0, 2.0)))
    with pytest.raises(ValueError, match="width"):
        ParameterBound("x", -1.0e308, 1.0e308)
