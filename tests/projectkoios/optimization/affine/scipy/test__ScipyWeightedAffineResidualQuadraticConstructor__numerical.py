import math
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("scipy.linalg")

from projectkoios.optimization.affine.affine import (  # noqa: E402
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)
from projectkoios.optimization.affine.scipy.affine import (  # noqa: E402
    ScipyWeightedAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.numerical


def test_scipy_rejects_exact_dependent_columns_before_gram_construction() -> None:
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
        ScipyWeightedAffineResidualQuadraticConstructor().execute(definition)
