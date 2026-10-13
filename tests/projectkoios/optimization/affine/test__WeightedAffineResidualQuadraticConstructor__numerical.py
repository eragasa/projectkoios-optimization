import math
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.affine.affine import (  # noqa: E402
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)
from projectkoios.optimization.affine.quadratic import (  # noqa: E402
    WeightedAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.numerical


def test_rejects_exact_dependent_columns_after_gram_rounding() -> None:
    rows = (
        (16.0, -16.0, 0.0),
        (-13.0, 11.0, -2.0),
        (8.0, 10.0, 18.0),
        (4.0, -7.0, -3.0),
    )
    weights = (0.3, math.pi, 1.3, 0.1)
    definition = WeightedAffineResidualDefinition(
        tuple(
            WeightedAffineResidualTerm(row, 0.0, weight)
            for row, weight in zip(rows, weights, strict=True)
        )
    )

    with pytest.raises(ValueError, match="numerical full column rank"):
        WeightedAffineResidualQuadraticConstructor().execute(definition)


def test_rejects_numerically_rank_deficient_design() -> None:
    definition = WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0, 1.0), 0.0, 1.0),
            WeightedAffineResidualTerm((1.0, 1.0 + 1.0e-15), 0.0, 1.0),
        )
    )

    with pytest.raises(ValueError, match="numerical full column rank"):
        WeightedAffineResidualQuadraticConstructor().execute(definition)


def test_preserves_representable_compensated_scales() -> None:
    definition = WeightedAffineResidualDefinition(
        (WeightedAffineResidualTerm((1.0e-200,), 2.0e-200, 1.0e300),)
    )

    quadratic = WeightedAffineResidualQuadraticConstructor().execute(definition)

    assert quadratic.center == pytest.approx((2.0,), rel=5.0e-15, abs=0.0)
    assert quadratic.matrix[0][0] == pytest.approx(1.0e-100, rel=5.0e-15)
    assert quadratic.squared_loss((3.0,)) == pytest.approx(1.0e-100, rel=5.0e-15)
