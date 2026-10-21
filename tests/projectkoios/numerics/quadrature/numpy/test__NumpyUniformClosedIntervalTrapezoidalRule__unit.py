"""Unit tests for immutable NumPy quadrature records and boundaries."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyDiagonalQuadratureMassMatrix,
    NumpyHomogeneousDirichletInteriorQuadrature1D,
    NumpyUniformClosedIntervalTrapezoidalRule,
    NumpyUniformClosedIntervalTrapezoidalRuleResult,
)
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)

pytestmark = pytest.mark.unit


def _result(
    point_count: int = 5,
) -> NumpyUniformClosedIntervalTrapezoidalRuleResult:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(
        0.0,
        1.0,
        point_count,
    )
    return NumpyUniformClosedIntervalTrapezoidalRule().calculate(definition)


def test_calculated_result_retains_rule_identity_and_conventions() -> None:
    result = _result()
    interior = result.restrict_homogeneous_dirichlet()

    assert result.interval == (0.0, 1.0)
    assert result.point_count == 5
    assert result.spacing == 0.25
    assert result.rule_identifier == (
        "projectkoios.numerics.quadrature.uniform_closed_interval_trapezoidal"
    )
    assert result.rule_version == 1
    assert interior.definition is result.definition
    assert interior.interval == result.interval
    assert interior.point_count == result.point_count
    assert interior.spacing == result.spacing
    assert interior.rule_identifier == result.rule_identifier
    assert interior.rule_version == result.rule_version
    assert interior.restriction_convention == ("homogeneous-dirichlet-interior-nodes")


def test_full_result_defensively_copies_weights() -> None:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(0.0, 1.0, 5)
    source = np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64)

    result = NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(
        definition,
        0.25,
        source,
    )
    source[0] = 9.0

    assert result.full_weights.tolist() == [0.125, 0.25, 0.25, 0.25, 0.125]
    assert not result.full_weights.flags.writeable
    with pytest.raises(ValueError):
        result.full_weights.flags.writeable = True


def test_interior_restriction_defensively_copies_correlated_arrays() -> None:
    full = _result()
    indices = np.array([1, 2, 3], dtype=np.int64)
    weights = np.full(3, 0.25, dtype=np.float64)

    interior = NumpyHomogeneousDirichletInteriorQuadrature1D.from_arrays(
        full,
        indices,
        weights,
    )
    indices[0] = 0
    weights[0] = 7.0

    assert interior.full_grid_indices.tolist() == [1, 2, 3]
    assert interior.interior_weights.tolist() == [0.25, 0.25, 0.25]
    assert not interior.full_grid_indices.flags.writeable
    assert not interior.interior_weights.flags.writeable
    with pytest.raises(ValueError):
        interior.full_grid_indices.flags.writeable = True
    with pytest.raises(ValueError):
        interior.interior_weights.flags.writeable = True


def test_diagonal_mass_matrix_defensively_copies_and_expands_immutably() -> None:
    source = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    mass = NumpyDiagonalQuadratureMassMatrix.from_array(source)
    source[1] = 8.0
    dense = mass.to_dense()

    assert mass.dimension == 3
    assert mass.convention == "diagonal-lumped-diag-weights"
    assert mass.diagonal.tolist() == [0.25, 0.5, 0.25]
    assert np.array_equal(dense, np.diag([0.25, 0.5, 0.25]))
    assert not mass.diagonal.flags.writeable
    assert not dense.flags.writeable
    with pytest.raises(ValueError):
        dense.flags.writeable = True


def test_two_point_rule_rejects_interior_restriction() -> None:
    result = _result(2)

    assert result.full_weights.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="at least three"):
        result.restrict_homogeneous_dirichlet()


def test_calculator_requires_nominal_definition() -> None:
    with pytest.raises(TypeError, match="definition"):
        NumpyUniformClosedIntervalTrapezoidalRule().calculate(
            cast(UniformClosedIntervalTrapezoidalRuleDefinition, object())
        )


def test_result_rejects_incorrect_spacing_or_weights() -> None:
    definition = UniformClosedIntervalTrapezoidalRuleDefinition(0.0, 1.0, 5)
    correct = np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64)

    with pytest.raises(ValueError, match="spacing"):
        NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(
            definition,
            0.5,
            correct,
        )
    with pytest.raises(TypeError, match="float64"):
        NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(
            definition,
            0.25,
            correct.astype(np.float32),
        )
    incorrect = correct.copy()
    incorrect[2] = 0.125
    with pytest.raises(ValueError, match="trapezoidal"):
        NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(
            definition,
            0.25,
            incorrect,
        )


def test_interior_result_rejects_incorrect_indices_or_weights() -> None:
    full = _result()

    with pytest.raises(ValueError, match="indices"):
        NumpyHomogeneousDirichletInteriorQuadrature1D.from_arrays(
            full,
            np.array([0, 1, 2], dtype=np.int64),
            np.full(3, 0.25, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="indexed"):
        NumpyHomogeneousDirichletInteriorQuadrature1D.from_arrays(
            full,
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.25, 0.5, 0.25], dtype=np.float64),
        )


def test_mass_matrix_rejects_nonpositive_or_nondiagonal_storage() -> None:
    with pytest.raises(ValueError, match="positive"):
        NumpyDiagonalQuadratureMassMatrix.from_array(
            np.array([0.5, 0.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        NumpyDiagonalQuadratureMassMatrix.from_array(np.eye(2, dtype=np.float64))
