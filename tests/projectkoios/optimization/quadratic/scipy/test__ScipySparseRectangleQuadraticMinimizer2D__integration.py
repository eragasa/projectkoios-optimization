from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
sp: Any = pytest.importorskip("scipy.sparse")

from projectkoios.optimization.quadratic.constrained import (  # noqa: E402
    RectangleQuadraticCandidate,
)
from projectkoios.optimization.quadratic.factory import (  # noqa: E402
    RectangleQuadraticMinimizerFactory,
    RectangleQuadraticMinimizerMethod,
)
from projectkoios.optimization.quadratic.numpy.constrained import (  # noqa: E402
    NumpyRectangleQuadraticMinimizer2D,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.quadratic.scipy.constrained import (  # noqa: E402
    ScipySparseRectangleQuadraticMinimizer2D,
)
from projectkoios.optimization.spaces.parameters import (  # noqa: E402
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.integration


def _bounds() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterBound("x", 0.0, 2.0),
            ParameterBound("y", 0.0, 2.0),
        )
    )


def test_factory_constructs_exact_sparse_rectangle_solver() -> None:
    solver = RectangleQuadraticMinimizerFactory.create(
        RectangleQuadraticMinimizerMethod.SCIPY_SPARSE
    )

    assert type(solver) is ScipySparseRectangleQuadraticMinimizer2D
    assert ScipySparseRectangleQuadraticMinimizer2D.__final__ is True


def test_dense_and_sparse_rectangle_results_agree() -> None:
    matrix = ((2.0, 1.0), (1.0, 2.0))
    dense_loss = PositiveDefiniteQuadraticLoss((-1.0, 0.5), matrix, 0.0)
    sparse_loss = PositiveDefiniteQuadraticLoss(
        (-1.0, 0.5),
        sp.csr_array(np.asarray(matrix)),
        0.0,
        sparse=True,
    )

    dense = NumpyRectangleQuadraticMinimizer2D().solve(dense_loss, _bounds())
    sparse = ScipySparseRectangleQuadraticMinimizer2D().solve(
        sparse_loss,
        _bounds(),
    )

    assert sparse.coordinates == pytest.approx(dense.coordinates)
    assert sparse.squared_loss == pytest.approx(dense.squared_loss)
    assert sparse.candidate is dense.candidate
    assert sparse.candidate is RectangleQuadraticCandidate.COORDINATE_0_LOWER


def test_dense_and_sparse_preserve_exact_duplicate_edge_tie_order() -> None:
    matrix = ((2.0, 1.0), (1.0, 2.0))
    dense_loss = PositiveDefiniteQuadraticLoss((-1.0, 0.5), matrix, 0.0)
    sparse_loss = PositiveDefiniteQuadraticLoss(
        (-1.0, 0.5),
        sp.csr_array(np.asarray(matrix)),
        0.0,
        sparse=True,
    )
    fixed_coordinate_bounds = ParameterSpace(
        (
            ParameterBound("x", 0.0, 0.0),
            ParameterBound("y", -2.0, 2.0),
        )
    )

    dense = NumpyRectangleQuadraticMinimizer2D().solve(
        dense_loss,
        fixed_coordinate_bounds,
    )
    sparse = ScipySparseRectangleQuadraticMinimizer2D().solve(
        sparse_loss,
        fixed_coordinate_bounds,
    )

    assert dense.coordinates == sparse.coordinates == (0.0, 0.0)
    assert dense.squared_loss == sparse.squared_loss == 1.5
    assert (
        dense.candidate
        is sparse.candidate
        is RectangleQuadraticCandidate.COORDINATE_0_LOWER
    )


def test_sparse_rectangle_solver_does_not_densify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (-1.0, 0.5),
        sp.csr_array(np.asarray(((2.0, 1.0), (1.0, 2.0)))),
        0.0,
        sparse=True,
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("sparse matrix was densified")

    monkeypatch.setattr(sp.csr_array, "toarray", fail)
    monkeypatch.setattr(sp.csr_array, "todense", fail)

    result = ScipySparseRectangleQuadraticMinimizer2D().solve(loss, _bounds())

    assert result.coordinates == pytest.approx((0.0, 0.0))


def test_sparse_rectangle_rejects_dense_loss() -> None:
    dense_loss = PositiveDefiniteQuadraticLoss(
        (0.0, 0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        0.0,
    )

    with pytest.raises(ValueError, match="requires a sparse loss"):
        ScipySparseRectangleQuadraticMinimizer2D().solve(dense_loss, _bounds())
