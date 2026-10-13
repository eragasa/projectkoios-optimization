from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("scipy.linalg")
pytest.importorskip("scipy.sparse")

from projectkoios.optimization.affine.affine import (  # noqa: E402
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)
from projectkoios.optimization.affine.scipy.affine import (  # noqa: E402
    ScipyWeightedAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.integration


def test_scipy_constructor_returns_sparse_completed_square() -> None:
    definition = WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0,), 0.0, 1.0),
            WeightedAffineResidualTerm((1.0,), 2.0, 1.0),
        )
    )

    quadratic = ScipyWeightedAffineResidualQuadraticConstructor().execute(definition)

    assert quadratic.sparse is True
    assert quadratic.center == pytest.approx((1.0,), abs=3.0e-16)
    assert float(quadratic.matrix[0, 0]) == 2.0
    assert quadratic.minimum_squared_loss == pytest.approx(2.0, abs=1.0e-30)
