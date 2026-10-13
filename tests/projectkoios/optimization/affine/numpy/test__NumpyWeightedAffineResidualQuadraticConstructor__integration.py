from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.affine.affine import (  # noqa: E402
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)
from projectkoios.optimization.affine.numpy.affine import (  # noqa: E402
    NumpyWeightedAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.integration


def test_numpy_constructor_returns_dense_completed_square() -> None:
    definition = WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0,), 0.0, 1.0),
            WeightedAffineResidualTerm((1.0,), 2.0, 1.0),
        )
    )

    quadratic = NumpyWeightedAffineResidualQuadraticConstructor().execute(definition)

    assert quadratic.sparse is False
    assert quadratic.center == pytest.approx((1.0,), abs=3.0e-16)
    assert quadratic.matrix == ((2.0,),)
    assert quadratic.minimum_squared_loss == pytest.approx(2.0, abs=1.0e-30)
