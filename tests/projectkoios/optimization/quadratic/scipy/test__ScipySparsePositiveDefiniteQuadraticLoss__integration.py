from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
sp: Any = pytest.importorskip("scipy.sparse")
pytest.importorskip("scipy.sparse.linalg")

from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.quadratic.scipy.quadratic import (  # noqa: E402
    ScipySparsePositiveDefiniteQuadraticLoss,
)

pytestmark = pytest.mark.integration


def test_sparse_flag_selects_scipy_backend() -> None:
    matrix = sp.csr_array(np.asarray(((4.0, 1.0), (1.0, 3.0))))
    loss = PositiveDefiniteQuadraticLoss(
        center=(1.0, -2.0),
        matrix=matrix,
        minimum_squared_loss=0.25,
        sparse=True,
    )

    assert loss.sparse
    assert loss.center == (1.0, -2.0)
    assert sp.issparse(loss.matrix)
    assert loss.dimension == 2
    assert loss.squared_loss((3.0, -1.0)) == pytest.approx(23.25)
    assert loss.root_loss((1.0, -2.0)) == 0.5


def test_dense_and_sparse_backends_agree_on_bounded_case() -> None:
    matrix = ((3.0, 0.5, 0.0), (0.5, 2.0, -0.25), (0.0, -0.25, 1.0))
    center = (0.5, -1.0, 2.0)
    point = (1.5, 1.0, 0.0)
    dense = PositiveDefiniteQuadraticLoss(center, matrix, 0.125)
    sparse = PositiveDefiniteQuadraticLoss(center, matrix, 0.125, sparse=True)

    assert sparse.squared_loss(point) == pytest.approx(dense.squared_loss(point))


def test_sparse_backend_rejects_indefinite_or_asymmetric_matrices() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        ScipySparsePositiveDefiniteQuadraticLoss(
            (0.0, 0.0),
            sp.csr_array(np.asarray(((1.0, 2.0), (2.0, 1.0)))),
            0.0,
        )
    with pytest.raises(ValueError, match="exactly symmetric"):
        ScipySparsePositiveDefiniteQuadraticLoss(
            (0.0, 0.0),
            sp.csr_array(np.asarray(((1.0, 0.0), (0.5, 1.0)))),
            0.0,
        )


def test_sparse_matrix_property_is_a_defensive_copy() -> None:
    loss = ScipySparsePositiveDefiniteQuadraticLoss(
        (0.0, 0.0),
        sp.eye(2, format="csr"),
        0.0,
    )
    exposed = loss.matrix
    exposed[0, 0] = 9.0

    assert loss.squared_loss((1.0, 0.0)) == 1.0
