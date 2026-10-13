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

pytestmark = pytest.mark.integration


def definition() -> WeightedAffineResidualDefinition:
    return WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0, 0.0), 1.0, 2.0),
            WeightedAffineResidualTerm((0.0, 1.0), -2.0, 3.0),
            WeightedAffineResidualTerm((1.0, 1.0), -1.0, 0.5),
        )
    )


def test_constructor_builds_the_hand_derived_dense_quadratic() -> None:
    residuals = definition()
    quadratic = WeightedAffineResidualQuadraticConstructor().execute(residuals)

    assert quadratic.center == pytest.approx((1.0, -2.0), rel=0.0, abs=2.0e-15)
    assert quadratic.matrix == ((2.5, 0.5), (0.5, 3.5))
    assert quadratic.minimum_squared_loss == pytest.approx(0.0, abs=2.0e-29)
    assert quadratic.squared_loss((3.0, -1.0)) == pytest.approx(
        residuals.squared_loss((3.0, -1.0)), rel=0.0, abs=4.0e-15
    )


def test_constructor_preserves_a_nonzero_least_squares_minimum() -> None:
    residuals = WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0,), 0.0, 1.0),
            WeightedAffineResidualTerm((1.0,), 2.0, 1.0),
        )
    )

    quadratic = WeightedAffineResidualQuadraticConstructor().execute(residuals)

    assert quadratic.center == pytest.approx((1.0,), rel=0.0, abs=3.0e-16)
    assert quadratic.matrix == ((2.0,),)
    assert quadratic.minimum_squared_loss == pytest.approx(2.0, abs=1.0e-30)
    assert quadratic.squared_loss((3.0,)) == pytest.approx(
        residuals.squared_loss((3.0,)), abs=2.0e-15
    )


def test_constructor_builds_the_same_sparse_function() -> None:
    pytest.importorskip("scipy")
    residuals = definition()

    quadratic = WeightedAffineResidualQuadraticConstructor().execute(
        residuals,
        sparse=True,
    )

    assert quadratic.sparse is True
    assert quadratic.squared_loss((3.0, -1.0)) == pytest.approx(
        residuals.squared_loss((3.0, -1.0)), rel=0.0, abs=4.0e-15
    )
    assert quadratic.root_loss((3.0, -1.0)) == pytest.approx(
        math.sqrt(15.5), rel=0.0, abs=1.0e-15
    )


def test_constructor_rejects_invalid_action_inputs() -> None:
    constructor = WeightedAffineResidualQuadraticConstructor()

    with pytest.raises(TypeError, match="definition"):
        constructor.execute(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sparse"):
        constructor.execute(definition(), sparse=1)  # type: ignore[arg-type]
