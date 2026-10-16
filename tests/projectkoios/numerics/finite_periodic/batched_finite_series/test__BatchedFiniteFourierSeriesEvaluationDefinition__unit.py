"""Unit tests for batched finite Fourier-series definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
)

pytestmark = pytest.mark.unit


def test_definition_records_shared_grid_and_channel_counts() -> None:
    definition = BatchedFiniteFourierSeriesEvaluationDefinition(
        2,
        ((0.0, 0.5), (0.25, -0.25)),
        ((0, 0), (1, -1), (-2, 3)),
        (
            (1.0 + 0.0j, 2.0 - 1.0j, -0.5 + 0.25j),
            (0.0 + 1.0j, 3.0 + 0.0j, 4.0 - 2.0j),
        ),
    )

    assert definition.dimension == 2
    assert definition.point_count == 2
    assert definition.term_count == 3
    assert definition.channel_count == 2
    assert definition.coefficient_channels[1][2] == 4.0 - 2.0j


def test_definition_permits_each_empty_collection_unambiguously() -> None:
    empty_points = BatchedFiniteFourierSeriesEvaluationDefinition(
        2, (), ((1, -1),), ((2.0 + 0.0j,),)
    )
    empty_modes = BatchedFiniteFourierSeriesEvaluationDefinition(
        2, ((0.0, 0.0),), (), ((), ())
    )
    empty_channels = BatchedFiniteFourierSeriesEvaluationDefinition(
        2, ((0.0, 0.0),), ((1, -1),), ()
    )

    assert (
        empty_points.point_count,
        empty_points.term_count,
        empty_points.channel_count,
    ) == (
        0,
        1,
        1,
    )
    assert (
        empty_modes.point_count,
        empty_modes.term_count,
        empty_modes.channel_count,
    ) == (
        1,
        0,
        2,
    )
    assert (
        empty_channels.point_count,
        empty_channels.term_count,
        empty_channels.channel_count,
    ) == (1, 1, 0)


@pytest.mark.parametrize("dimension", (True, 1.0, 0, -1))
def test_definition_rejects_invalid_dimension(dimension: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BatchedFiniteFourierSeriesEvaluationDefinition(
            dimension,  # type: ignore[arg-type]
            (),
            (),
            (),
        )


@pytest.mark.parametrize(
    ("coordinates", "integer_modes", "coefficient_channels"),
    (
        ([(0.0,)], ((0,),), ((1.0 + 0.0j,),)),
        (((0.0,),), [(0,)], ((1.0 + 0.0j,),)),
        (((0.0,),), ((0,),), [(1.0 + 0.0j,)]),
        (((0.0,),), ((0,),), ([1.0 + 0.0j],)),
        (((0,),), ((0,),), ((1.0 + 0.0j,),)),
        (((0.0,),), ((0.0,),), ((1.0 + 0.0j,),)),
        (((0.0,),), ((0,),), ((1.0,),)),
    ),
)
def test_definition_rejects_noncanonical_container_or_scalar_types(
    coordinates: object,
    integer_modes: object,
    coefficient_channels: object,
) -> None:
    with pytest.raises(TypeError):
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            coordinates,  # type: ignore[arg-type]
            integer_modes,  # type: ignore[arg-type]
            coefficient_channels,  # type: ignore[arg-type]
        )


def test_definition_rejects_channel_width_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per integer mode"):
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.0,),),
            ((0,), (1,)),
            ((1.0 + 0.0j,),),
        )


@pytest.mark.parametrize(
    ("coordinates", "coefficients"),
    (
        (((float("nan"),),), ((1.0 + 0.0j,),)),
        (((0.0,),), ((complex(float("inf"), 0.0),),)),
    ),
)
def test_definition_rejects_nonfinite_values(
    coordinates: tuple[tuple[float, ...], ...],
    coefficients: tuple[tuple[complex, ...], ...],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            coordinates,
            ((0,),),
            coefficients,
        )


def test_definition_rejects_modes_outside_exact_binary64_range() -> None:
    with pytest.raises(ValueError, match=r"2\*\*53"):
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.0,),),
            ((2**53 + 1,),),
            ((1.0 + 0.0j,),),
        )


def test_definition_is_frozen() -> None:
    definition = BatchedFiniteFourierSeriesEvaluationDefinition(1, (), (), ())

    with pytest.raises(FrozenInstanceError):
        definition.dimension = 2  # type: ignore[misc]
