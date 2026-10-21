"""Dependency-free definition for uniform closed-interval trapezoidal quadrature.

The capability defines one endpoint-inclusive rule and its homogeneous-Dirichlet
interior restriction. It assigns no physical units, finite-element basis semantics,
scientific interpretation, or integration-error acceptance policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import final

from projectkoios.numerics.validation import (
    exact_finite_float,
    exact_positive_integer,
)

_RULE_IDENTIFIER = (
    "projectkoios.numerics.quadrature.uniform_closed_interval_trapezoidal"
)
_RULE_VERSION = 1
_NODE_CONVENTION = "endpoint-inclusive-uniform-closed-interval"
_FULL_WEIGHT_CONVENTION = "composite-trapezoidal"
_INTERIOR_RESTRICTION_CONVENTION = "homogeneous-dirichlet-interior-nodes"
_MASS_MATRIX_CONVENTION = "diagonal-lumped-diag-weights"


@final
@dataclass(frozen=True, slots=True)
class UniformClosedIntervalTrapezoidalRuleDefinition:
    """Define one unit-free endpoint-inclusive uniform trapezoidal rule.

    Parameters
    ----------
    left_endpoint, right_endpoint
        Exact finite built-in floats defining the ordered closed interval. The right
        endpoint must be strictly greater than the left endpoint, and the represented
        interval length and spacing must remain finite and positive.
    point_count
        Exact built-in integer number of endpoint-inclusive nodes. At least two points
        are required for the full rule. The homogeneous-Dirichlet interior restriction
        additionally requires at least three points.

    Notes
    -----
    The full nodes are ordered by ``x_i = left_endpoint + i * spacing`` for
    ``i = 0, ..., point_count - 1`` with
    ``spacing = (right_endpoint - left_endpoint) / (point_count - 1)``. The numerical
    implementation retains weights, not coordinates; consumers may correlate the rule
    with their own unit-bearing coordinate representation.
    """

    left_endpoint: float
    right_endpoint: float
    point_count: int
    rule_identifier: str = field(default=_RULE_IDENTIFIER, init=False)
    rule_version: int = field(default=_RULE_VERSION, init=False)
    node_convention: str = field(default=_NODE_CONVENTION, init=False)
    full_weight_convention: str = field(
        default=_FULL_WEIGHT_CONVENTION,
        init=False,
    )
    interior_restriction_convention: str = field(
        default=_INTERIOR_RESTRICTION_CONVENTION,
        init=False,
    )
    mass_matrix_convention: str = field(
        default=_MASS_MATRIX_CONVENTION,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate exact scalar types and representable interval arithmetic.

        Raises
        ------
        TypeError
            If an endpoint is not an exact built-in float or the point count is not an
            exact built-in integer. Boolean values are rejected by these exact-type
            boundaries.
        ValueError
            If an endpoint is nonfinite, the interval is reversed or degenerate, fewer
            than two points are requested, or the interval length, spacing, or endpoint
            half-weight is not representable as a finite positive binary64 value.
        """
        left = exact_finite_float(self.left_endpoint, "left_endpoint")
        right = exact_finite_float(self.right_endpoint, "right_endpoint")
        point_count = exact_positive_integer(self.point_count, "point_count")
        if right <= left:
            raise ValueError("right_endpoint must be greater than left_endpoint")
        if point_count < 2:
            raise ValueError("point_count must be at least two")
        length = right - left
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("interval length must be finite and positive")
        spacing = length / (point_count - 1)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing must be finite and positive")
        if 0.5 * spacing <= 0.0:
            raise ValueError("endpoint half-weight must be representable and positive")

    @property
    def interval(self) -> tuple[float, float]:
        """Return the ordered closed interval as exact retained endpoints."""
        return (self.left_endpoint, self.right_endpoint)

    @property
    def interval_length(self) -> float:
        """Return the finite positive represented interval length."""
        return self.right_endpoint - self.left_endpoint

    @property
    def spacing(self) -> float:
        """Return the finite positive uniform full-grid spacing."""
        return self.interval_length / (self.point_count - 1)

    @property
    def interior_point_count(self) -> int:
        """Return the number of full-grid nodes strictly inside the interval."""
        return max(0, self.point_count - 2)


__all__ = ["UniformClosedIntervalTrapezoidalRuleDefinition"]
