import pytest

from projectkoios.optimization.affine.affine import (
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)
from projectkoios.optimization.affine.quadratic import (
    weighted_design,
    weighted_normal_matrix,
)

pytestmark = pytest.mark.unit


def test_weighted_design_and_normal_matrix_are_hand_derived() -> None:
    definition = WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0, 2.0), 3.0, 4.0),
            WeightedAffineResidualTerm((2.0, -1.0), -2.0, 9.0),
        )
    )

    design, target = weighted_design(definition)
    matrix = weighted_normal_matrix(definition)

    assert design == ((2.0, 4.0), (6.0, -3.0))
    assert target == (6.0, -6.0)
    assert matrix == ((40.0, -10.0), (-10.0, 25.0))
    assert matrix[0][1] == matrix[1][0]


def test_quadratic_functions_require_the_exact_definition_type() -> None:
    with pytest.raises(TypeError, match="definition"):
        weighted_design(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="definition"):
        weighted_normal_matrix(object())  # type: ignore[arg-type]
