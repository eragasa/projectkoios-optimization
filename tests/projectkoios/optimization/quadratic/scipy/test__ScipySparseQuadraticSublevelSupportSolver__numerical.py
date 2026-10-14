from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
sp: Any = pytest.importorskip("scipy.sparse")
spla: Any = pytest.importorskip("scipy.sparse.linalg")

from projectkoios.optimization.quadratic.factory import (  # noqa: E402
    QuadraticSublevelSupportMethod,
    QuadraticSublevelSupportSolverFactory,
)
from projectkoios.optimization.quadratic.numpy.constrained import (  # noqa: E402
    NumpyQuadraticSublevelSupportSolver,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.quadratic.scipy.constrained import (  # noqa: E402
    ScipySparseQuadraticSublevelSupportSolver,
)

pytestmark = pytest.mark.numerical


def _losses() -> tuple[PositiveDefiniteQuadraticLoss, PositiveDefiniteQuadraticLoss]:
    matrix = (
        (3.0, 0.5, 0.0),
        (0.5, 2.0, -0.25),
        (0.0, -0.25, 1.0),
    )
    center = (0.5, -1.0, 2.0)
    return (
        PositiveDefiniteQuadraticLoss(center, matrix, 0.125),
        PositiveDefiniteQuadraticLoss(
            center,
            sp.csr_array(np.asarray(matrix)),
            0.125,
            sparse=True,
        ),
    )


def test_factory_constructs_exact_sparse_support_solver() -> None:
    solver = QuadraticSublevelSupportSolverFactory.create(
        QuadraticSublevelSupportMethod.SCIPY_SPARSE
    )

    assert type(solver) is ScipySparseQuadraticSublevelSupportSolver
    assert ScipySparseQuadraticSublevelSupportSolver.__final__ is True


def test_dense_and_sparse_support_results_agree() -> None:
    dense_loss, sparse_loss = _losses()
    direction = (1.0, -2.0, 0.5)
    ceiling = 2.125

    dense = NumpyQuadraticSublevelSupportSolver().solve(
        dense_loss,
        ceiling,
        direction,
    )
    sparse = ScipySparseQuadraticSublevelSupportSolver().solve(
        sparse_loss,
        ceiling,
        direction,
    )

    assert sparse.point == pytest.approx(dense.point, rel=1.0e-10, abs=1.0e-12)
    assert sparse.support_value == pytest.approx(
        dense.support_value,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert sparse.squared_loss == pytest.approx(ceiling, rel=1.0e-10, abs=1.0e-12)


def test_sparse_support_solver_does_not_densify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, sparse_loss = _losses()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("sparse matrix was densified")

    monkeypatch.setattr(sp.csr_array, "toarray", fail)
    monkeypatch.setattr(sp.csr_array, "todense", fail)

    result = ScipySparseQuadraticSublevelSupportSolver().solve(
        sparse_loss,
        1.125,
        (1.0, 0.0, 0.0),
    )

    assert result.squared_loss == pytest.approx(1.125)


def test_sparse_backend_solve_failure_is_translated_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, sparse_loss = _losses()
    failure = RuntimeError("forced sparse solve failure")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(spla, "spsolve", fail)

    with pytest.raises(
        ValueError, match="sparse quadratic support solve failed"
    ) as caught:
        ScipySparseQuadraticSublevelSupportSolver().solve(
            sparse_loss,
            1.125,
            (1.0, 0.0, 0.0),
        )

    assert caught.value.__cause__ is failure


def test_numpy_support_rejects_sparse_loss() -> None:
    _, sparse_loss = _losses()

    with pytest.raises(ValueError, match="requires a dense loss"):
        NumpyQuadraticSublevelSupportSolver().solve(
            sparse_loss,
            1.125,
            (1.0, 0.0, 0.0),
        )
