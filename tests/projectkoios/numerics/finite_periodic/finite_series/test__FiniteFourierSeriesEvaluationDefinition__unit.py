"""Unit tests for finite Fourier-series evaluation definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
)

pytestmark = pytest.mark.unit


def test_definition_retains_valid_dependency_free_inputs() -> None:
    definition = FiniteFourierSeriesEvaluationDefinition(
        2,
        ((0.0, 0.25), (0.5, -0.25)),
        ((0, 0), (1, -2)),
        (1.0 + 0.0j, -0.5 + 0.25j),
    )

    assert definition.dimension == 2
    assert definition.point_count == 2
    assert definition.term_count == 2
    assert definition.coordinates == ((0.0, 0.25), (0.5, -0.25))
    assert definition.integer_modes == ((0, 0), (1, -2))
    assert definition.coefficients == (1.0 + 0.0j, -0.5 + 0.25j)
    with pytest.raises(FrozenInstanceError):
        definition.dimension = 3  # type: ignore[misc]


def test_definition_accepts_empty_points_and_terms_with_explicit_dimension() -> None:
    definition = FiniteFourierSeriesEvaluationDefinition(3, (), (), ())

    assert definition.dimension == 3
    assert definition.point_count == 0
    assert definition.term_count == 0


@pytest.mark.parametrize(
    ("dimension", "error"),
    (
        (True, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ),
)
def test_definition_rejects_invalid_dimension(
    dimension: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="dimension"):
        FiniteFourierSeriesEvaluationDefinition(
            cast(Any, dimension),
            (),
            (),
            (),
        )


@pytest.mark.parametrize(
    ("coordinates", "error", "message"),
    (
        ([], TypeError, "coordinates"),
        (([],), TypeError, "coordinate rows"),
        (((0.0,),), ValueError, "dimension"),
        (((0, 0.0),), TypeError, "floats"),
        (((float("nan"), 0.0),), ValueError, "finite"),
        (((float("inf"), 0.0),), ValueError, "finite"),
    ),
)
def test_definition_rejects_invalid_coordinates(
    coordinates: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            cast(Any, coordinates),
            (),
            (),
        )


def test_definition_requires_integer_modes_to_be_a_tuple() -> None:
    with pytest.raises(TypeError, match="integer_modes"):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            cast(Any, []),
            (),
        )


def test_definition_requires_each_integer_mode_to_be_a_tuple() -> None:
    list_row: list[object] = []
    integer_modes_with_list_row = (list_row,)

    with pytest.raises(TypeError, match="mode rows"):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            cast(Any, integer_modes_with_list_row),
            (0.0 + 0.0j,),
        )


def test_definition_requires_integer_mode_width_to_match_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            ((0,),),
            (0.0 + 0.0j,),
        )


@pytest.mark.parametrize("component", (True, 1.0), ids=("boolean", "float"))
def test_definition_requires_exact_integer_mode_components(
    component: object,
) -> None:
    with pytest.raises(TypeError, match="integers"):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            ((cast(Any, component), 0),),
            (0.0 + 0.0j,),
        )


@pytest.mark.parametrize(
    "component",
    ((2**53) + 1, -(2**53) - 1),
    ids=("above-positive-limit", "below-negative-limit"),
)
def test_definition_rejects_integer_modes_outside_exact_binary64_range(
    component: int,
) -> None:
    with pytest.raises(ValueError, match="2\\*\\*53"):
        FiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            ((component, 0),),
            (0.0 + 0.0j,),
        )


def test_definition_accepts_integer_modes_at_exact_binary64_endpoints() -> None:
    definition = FiniteFourierSeriesEvaluationDefinition(
        2,
        (),
        ((-(2**53), 2**53),),
        (0.0 + 0.0j,),
    )

    assert definition.integer_modes == ((-(2**53), 2**53),)


@pytest.mark.parametrize(
    ("coefficients", "error", "message"),
    (
        ([], TypeError, "coefficients"),
        ((1.0,), TypeError, "complex"),
        ((complex(float("nan"), 0.0),), ValueError, "finite"),
        ((complex(0.0, float("inf")),), ValueError, "finite"),
    ),
)
def test_definition_rejects_invalid_coefficients(
    coefficients: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        FiniteFourierSeriesEvaluationDefinition(
            1,
            (),
            ((0,),),
            cast(Any, coefficients),
        )


def test_definition_rejects_term_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per integer mode"):
        FiniteFourierSeriesEvaluationDefinition(
            1,
            (),
            ((0,), (1,)),
            (1.0 + 0.0j,),
        )
