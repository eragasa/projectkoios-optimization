import pytest

from projectkoios.optimization.selection.finite import (
    absolute_ambiguity_set,
    first_minimum,
    indices_at_or_below,
)

pytestmark = pytest.mark.unit


def test_ceiling_filter_is_inclusive_ordered_and_can_be_empty() -> None:
    values = (0.5, 0.25, 0.75, 0.25)

    assert indices_at_or_below(values, 0.25) == (1, 3)
    assert indices_at_or_below(values, 0.1) == ()


def test_completed_m3_finite_family_regression() -> None:
    minima = (
        0.09692887541289989,
        0.149994481487845,
        0.20984544126705884,
        0.20015475304768268,
        0.1332808001578272,
        0.09450689673607807,
        0.13328080015782712,
        0.20015475304768265,
        0.20984544126705887,
    )

    assert first_minimum(minima).index == 5
    assert indices_at_or_below(minima, 0.31**2) == (5,)
    assert absolute_ambiguity_set(minima, 0.003).indices == (0, 5)


@pytest.mark.parametrize(
    ("operation", "arguments", "error"),
    (
        (first_minimum, ((),), ValueError),
        (first_minimum, ((True,),), TypeError),
        (first_minimum, ((float("inf"),),), ValueError),
        (indices_at_or_below, ((1.0,), True), TypeError),
        (absolute_ambiguity_set, ((1.0,), True), TypeError),
        (absolute_ambiguity_set, ((1.0,), -1.0), ValueError),
    ),
)
def test_selection_rejects_invalid_values(
    operation: object,
    arguments: tuple[object, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        operation(*arguments)  # type: ignore[operator]
