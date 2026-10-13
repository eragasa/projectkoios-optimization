import math
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)

pytestmark = pytest.mark.integration


def test_dense_facade_exposes_completed_square_geometry() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (1.0, -1.0),
        ((2.0, 0.5), (0.5, 3.0)),
        0.25,
    )

    assert loss.center == (1.0, -1.0)
    assert loss.matrix == ((2.0, 0.5), (0.5, 3.0))
    assert loss.dimension == 2
    assert loss.minimum_squared_loss == 0.25
    assert loss.sparse is False
    assert loss.squared_loss((2.0, 1.0)) == pytest.approx(16.25)
    assert loss.root_loss((2.0, 1.0)) == pytest.approx(math.sqrt(16.25))


def test_facade_rejects_non_boolean_backend_selection() -> None:
    with pytest.raises(TypeError, match="sparse"):
        PositiveDefiniteQuadraticLoss(
            (0.0,),
            ((1.0,),),
            0.0,
            sparse=1,  # type: ignore[arg-type]
        )
