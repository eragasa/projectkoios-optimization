"""Independent numerical identities for uniform one-dimensional quadrature."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyUniformClosedIntervalTrapezoidalRule,
    NumpyUniformClosedIntervalTrapezoidalRuleResult,
)
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)

pytestmark = pytest.mark.numerical


def _calculate(
    left: float,
    right: float,
    point_count: int,
) -> NumpyUniformClosedIntervalTrapezoidalRuleResult:
    return NumpyUniformClosedIntervalTrapezoidalRule().calculate(
        UniformClosedIntervalTrapezoidalRuleDefinition(
            left,
            right,
            point_count,
        )
    )


def test_full_weights_sum_to_interval_length() -> None:
    result = _calculate(-2.0, 3.0, 11)

    assert float(np.sum(result.full_weights)) == pytest.approx(5.0, abs=1.0e-15)


def test_rule_integrates_constant_and_linear_functions_independently() -> None:
    left = -1.5
    right = 2.5
    result = _calculate(left, right, 9)
    nodes = np.linspace(left, right, 9, dtype=np.float64)

    represented_constant = float(result.full_weights @ np.ones(9, dtype=np.float64))
    represented_linear = float(result.full_weights @ nodes)
    exact_constant = right - left
    exact_linear = 0.5 * (right**2 - left**2)

    assert represented_constant == pytest.approx(exact_constant, abs=1.0e-15)
    assert represented_linear == pytest.approx(exact_linear, abs=2.0e-15)


def test_dirichlet_restriction_reconstructs_full_zero_endpoint_form() -> None:
    result = _calculate(0.0, 2.0, 5)
    interior = result.restrict_homogeneous_dirichlet()
    left = np.array([0.0, 1.0, -2.0, 3.0, 0.0], dtype=np.float64)
    right = np.array([0.0, 4.0, 5.0, -1.0, 0.0], dtype=np.float64)

    full_form = float(np.sum(result.full_weights * left * right))
    restricted_form = float(
        np.sum(
            interior.interior_weights
            * left[interior.full_grid_indices]
            * right[interior.full_grid_indices]
        )
    )

    assert interior.full_grid_indices.tolist() == [1, 2, 3]
    assert interior.interior_weights.tolist() == [0.5, 0.5, 0.5]
    assert restricted_form == full_form


def test_diagonal_mass_matrix_matches_independent_weighted_bilinear_form() -> None:
    result = _calculate(-1.0, 1.0, 5)
    left = np.array([2.0, -1.0, 3.0, 0.5, -2.0], dtype=np.float64)
    right = np.array([1.0, 4.0, -2.0, 3.0, 0.25], dtype=np.float64)
    dense = result.full_mass_matrix.to_dense()

    matrix_form = float(left @ dense @ right)
    independent_form = float(
        sum(
            weight * first * second
            for weight, first, second in zip(
                result.full_weights.tolist(),
                left.tolist(),
                right.tolist(),
                strict=True,
            )
        )
    )

    assert matrix_form == pytest.approx(independent_form, abs=1.0e-15)
    assert np.count_nonzero(dense - np.diag(np.diag(dense))) == 0


def test_interior_mass_matrix_is_spacing_times_identity() -> None:
    result = _calculate(0.0, 1.0, 6)
    interior = result.restrict_homogeneous_dirichlet()
    dense = interior.mass_matrix.to_dense()

    expected = 0.2 * np.eye(4, dtype=np.float64)
    assert np.array_equal(dense, expected)
