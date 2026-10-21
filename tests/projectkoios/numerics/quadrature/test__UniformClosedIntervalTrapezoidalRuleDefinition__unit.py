"""Unit tests for uniform closed-interval trapezoidal definitions."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)

pytestmark = pytest.mark.unit


def test_definition_retains_interval_spacing_and_convention_identity() -> None:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(-2.0, 3.0, 11)

    assert definition.interval == (-2.0, 3.0)
    assert definition.interval_length == 5.0
    assert definition.spacing == 0.5
    assert definition.point_count == 11
    assert definition.interior_point_count == 9
    assert definition.rule_identifier == (
        "projectkoios.numerics.quadrature.uniform_closed_interval_trapezoidal"
    )
    assert definition.rule_version == 1
    assert definition.node_convention == "endpoint-inclusive-uniform-closed-interval"
    assert definition.full_weight_convention == "composite-trapezoidal"
    assert definition.interior_restriction_convention == (
        "homogeneous-dirichlet-interior-nodes"
    )
    assert definition.mass_matrix_convention == "diagonal-lumped-diag-weights"


def test_two_point_definition_has_no_interior_nodes() -> None:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(0.0, 1.0, 2)

    assert definition.spacing == 1.0
    assert definition.interior_point_count == 0


@pytest.mark.parametrize(
    ("left", "right", "point_count", "error", "message"),
    [
        (cast(float, True), 1.0, 3, TypeError, "left_endpoint"),
        (0.0, cast(float, False), 3, TypeError, "right_endpoint"),
        (cast(float, 0), 1.0, 3, TypeError, "left_endpoint"),
        (0.0, cast(float, 1), 3, TypeError, "right_endpoint"),
        (math.nan, 1.0, 3, ValueError, "left_endpoint"),
        (0.0, math.inf, 3, ValueError, "right_endpoint"),
        (0.0, 0.0, 3, ValueError, "greater"),
        (1.0, 0.0, 3, ValueError, "greater"),
        (0.0, 1.0, cast(int, True), TypeError, "point_count"),
        (0.0, 1.0, 1, ValueError, "at least two"),
    ],
)
def test_definition_rejects_invalid_public_inputs(
    left: float,
    right: float,
    point_count: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        UniformClosedIntervalTrapezoidalRuleDefinition(
            left,
            right,
            point_count,
        )


def test_definition_rejects_nonfinite_represented_interval_length() -> None:
    with pytest.raises(ValueError, match="interval length"):
        UniformClosedIntervalTrapezoidalRuleDefinition(
            -float.fromhex("0x1.fffffffffffffp+1023"),
            float.fromhex("0x1.fffffffffffffp+1023"),
            3,
        )


def test_definition_is_frozen() -> None:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(0.0, 1.0, 5)

    with pytest.raises(FrozenInstanceError):
        definition.point_count = 7  # type: ignore[misc]
