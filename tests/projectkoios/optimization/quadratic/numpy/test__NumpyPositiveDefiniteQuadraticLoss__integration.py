import math
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.quadratic.numpy.quadratic import (  # noqa: E402
    NumpyPositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)

pytestmark = pytest.mark.integration


def test_dense_quadratic_uses_numpy_backend_and_retains_tuple_state() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        center=(1, -2),
        matrix=((4, 1), (1, 3)),
        minimum_squared_loss=0.25,
    )

    assert not loss.sparse
    assert loss.center == (1.0, -2.0)
    assert loss.matrix == ((4.0, 1.0), (1.0, 3.0))
    assert loss.dimension == 2
    assert loss.squared_loss((3.0, -1.0)) == 23.25
    assert loss.root_loss((1.0, -2.0)) == 0.5


def test_direct_numpy_quadratic_retains_canonical_tuple_storage() -> None:
    center = (0.0,)
    matrix = ((2.0,),)

    loss = NumpyPositiveDefiniteQuadraticLoss(center, matrix, 0.0)

    assert loss.center is center
    assert loss.matrix is matrix


def test_direct_numpy_quadratic_rejects_noncanonical_storage() -> None:
    with pytest.raises(TypeError, match="exact built-in float"):
        NumpyPositiveDefiniteQuadraticLoss((0,), ((1.0,),), 0.0)  # type: ignore[arg-type]


def test_direct_numpy_quadratic_rejects_non_positive_definite_matrix() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        NumpyPositiveDefiniteQuadraticLoss(
            center=(0.0, 0.0),
            matrix=((1.0, 2.0), (2.0, 1.0)),
            minimum_squared_loss=0.0,
        )


def test_dense_quadratic_rejects_invalid_requests_and_overflow() -> None:
    with pytest.raises(TypeError, match="sparse"):
        PositiveDefiniteQuadraticLoss(
            (0.0,),
            ((1.0,),),
            0.0,
            sparse=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly symmetric"):
        PositiveDefiniteQuadraticLoss((0.0, 0.0), ((1.0, 0.0), (0.5, 1.0)), 0.0)

    loss = PositiveDefiniteQuadraticLoss((0.0,), ((1.0,),), 0.0)
    with pytest.raises(ValueError, match="finite"):
        loss.squared_loss((1.0e308,))
    assert math.isfinite(loss.squared_loss((1.0e150,)))
