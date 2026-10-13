import random
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

pytestmark = pytest.mark.property


def test_generated_full_rank_systems_match_direct_residual_evaluation() -> None:
    generator = random.Random(20261011)
    for dimension in range(1, 7):
        terms = [
            WeightedAffineResidualTerm(
                tuple(1.0 if row == column else 0.0 for column in range(dimension)),
                generator.uniform(-2.0, 2.0),
                generator.uniform(0.25, 3.0),
            )
            for row in range(dimension)
        ]
        terms.extend(
            WeightedAffineResidualTerm(
                tuple(generator.uniform(-2.0, 2.0) for _ in range(dimension)),
                generator.uniform(-2.0, 2.0),
                generator.uniform(0.25, 3.0),
            )
            for _ in range(3)
        )
        definition = WeightedAffineResidualDefinition(tuple(terms))
        loss = WeightedAffineResidualQuadraticConstructor().execute(definition)

        for _ in range(4):
            point = tuple(generator.uniform(-3.0, 3.0) for _ in range(dimension))
            assert loss.squared_loss(point) == pytest.approx(
                definition.squared_loss(point),
                rel=5.0e-13,
                abs=5.0e-13,
            )
