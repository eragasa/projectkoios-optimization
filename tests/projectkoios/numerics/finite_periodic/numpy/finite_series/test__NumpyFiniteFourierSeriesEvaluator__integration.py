"""Integration tests for NumPy finite Fourier-series evaluation."""

from __future__ import annotations

import math

import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.finite_series import (
    NumpyFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.integration


def _evaluate(
    dimension: int,
    coordinates: tuple[tuple[float, ...], ...],
    integer_modes: tuple[tuple[int, ...], ...],
    coefficients: tuple[complex, ...],
) -> tuple[complex, ...]:
    result = NumpyFiniteFourierSeriesEvaluator().evaluate(
        FiniteFourierSeriesEvaluationDefinition(
            dimension,
            coordinates,
            integer_modes,
            coefficients,
        )
    )
    assert result.dimension == dimension
    assert result.point_count == len(coordinates)
    assert result.term_count == len(integer_modes)
    assert all(type(value) is complex for value in result.values)
    return result.values


def test_constant_and_positive_exponent_orientation() -> None:
    values = _evaluate(
        1,
        ((0.0,), (0.25,), (0.5,), (0.75,)),
        ((0,), (1,)),
        (2.0 - 0.5j, 1.0 + 0.0j),
    )

    expected = (
        3.0 - 0.5j,
        2.0 + 0.5j,
        1.0 - 0.5j,
        2.0 - 1.5j,
    )
    assert values == pytest.approx(expected, abs=2.0e-15, rel=0.0)


def test_negative_and_duplicate_modes_remain_separate_terms() -> None:
    values = _evaluate(
        1,
        ((0.125,),),
        ((1,), (-1,), (1,)),
        (1.0 + 2.0j, 1.0 - 2.0j, -0.5 + 0.25j),
    )
    phase = complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
    expected = (
        (1.0 + 2.0j) * phase
        + (1.0 - 2.0j) * phase.conjugate()
        + (-0.5 + 0.25j) * phase,
    )

    assert values == pytest.approx(expected, abs=3.0e-15, rel=0.0)


def test_two_and_three_dimensional_values() -> None:
    two_dimensional = _evaluate(
        2,
        ((0.25, 0.5),),
        ((1, 0), (0, 1), (1, -1)),
        (1.0 + 0.0j, 2.0 + 0.0j, 0.0 + 1.0j),
    )
    three_dimensional = _evaluate(
        3,
        ((0.5, 0.25, 0.0),),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        (1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j),
    )

    assert two_dimensional == pytest.approx((-1.0 + 1.0j,), abs=3.0e-15, rel=0.0)
    assert three_dimensional == pytest.approx((0.0 + 1.0j,), abs=3.0e-15, rel=0.0)


def test_conjugate_paired_terms_produce_known_real_values() -> None:
    values = _evaluate(
        1,
        ((0.0,), (0.25,), (0.5,)),
        ((1,), (-1,)),
        (1.5 + 0.0j, 1.5 + 0.0j),
    )

    assert values == pytest.approx(
        (3.0 + 0.0j, 0.0 + 0.0j, -3.0 + 0.0j), abs=2.0e-15, rel=0.0
    )


def test_declared_term_order_controls_scalar_accumulation_order() -> None:
    first = _evaluate(
        1,
        ((0.0,),),
        ((0,), (0,), (0,)),
        (1.0e16 + 0.0j, 1.0 + 0.0j, -1.0e16 + 0.0j),
    )
    second = _evaluate(
        1,
        ((0.0,),),
        ((0,), (0,), (0,)),
        (1.0e16 + 0.0j, -1.0e16 + 0.0j, 1.0 + 0.0j),
    )

    assert first == (0.0 + 0.0j,)
    assert second == (1.0 + 0.0j,)


def test_declared_component_order_controls_inner_product_accumulation() -> None:
    # Sequential binary64 accumulation loses 0.25 after adding it to 1e16, then
    # cancellation with -1e16 leaves zero. Reassociating the first and third
    # components would instead leave 0.25 and rotate the result to approximately 1j.
    # This fixture therefore prevents a dot product or pairwise reduction from silently
    # replacing the reviewed component order.
    values = _evaluate(
        3,
        ((1.0e16, 0.25, -1.0e16),),
        ((1, 1, 1),),
        (1.0 + 0.0j,),
    )

    assert values == (1.0 + 0.0j,)


def test_consumer_shaped_symmetric_mode_fixture() -> None:
    values = _evaluate(
        2,
        ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5)),
        ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
        (1.0 + 0.0j, -0.5 + 0.0j, -0.5 + 0.0j, -0.5 + 0.0j, -0.5 + 0.0j),
    )

    assert values == pytest.approx(
        (-1.0 + 0.0j, 1.0 + 0.0j, 3.0 + 0.0j), abs=3.0e-15, rel=0.0
    )


def test_nonfinite_intermediate_fails_closed() -> None:
    definition = FiniteFourierSeriesEvaluationDefinition(
        1,
        ((float.fromhex("0x1.fffffffffffffp+1023"),),),
        ((2,),),
        (1.0 + 0.0j,),
    )

    with pytest.raises(FloatingPointError):
        NumpyFiniteFourierSeriesEvaluator().evaluate(definition)
