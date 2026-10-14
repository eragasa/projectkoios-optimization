import math
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

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

pytestmark = pytest.mark.numerical


def _loss() -> PositiveDefiniteQuadraticLoss:
    return PositiveDefiniteQuadraticLoss(
        (1.0, -1.0),
        ((2.0, 0.0), (0.0, 8.0)),
        0.25,
    )


def test_factory_constructs_exact_numpy_support_solver() -> None:
    solver = QuadraticSublevelSupportSolverFactory.create(
        QuadraticSublevelSupportMethod.NUMPY_DENSE
    )

    assert type(solver) is NumpyQuadraticSublevelSupportSolver


def test_axis_support_lies_on_boundary_and_satisfies_kkt() -> None:
    loss = _loss()
    ceiling = 4.25
    direction = (1.0, 0.0)

    result = NumpyQuadraticSublevelSupportSolver().solve(
        loss,
        ceiling,
        direction,
    )

    assert result.point == pytest.approx((1.0 + math.sqrt(2.0), -1.0))
    assert result.squared_loss == pytest.approx(ceiling)
    assert result.support_value == pytest.approx(result.point[0])

    matrix = np.asarray(loss.matrix)
    delta = np.asarray(result.point) - np.asarray(loss.center)
    budget = ceiling - loss.minimum_squared_loss
    solved = np.linalg.solve(matrix, np.asarray(direction))
    alpha = math.sqrt(budget / float(np.asarray(direction) @ solved))
    multiplier = 1.0 / (2.0 * alpha)
    stationarity = -np.asarray(direction) + 2.0 * multiplier * (matrix @ delta)
    assert np.linalg.norm(stationarity, ord=np.inf) < 1.0e-12


def test_zero_budget_returns_exact_center() -> None:
    loss = _loss()

    result = NumpyQuadraticSublevelSupportSolver().solve(
        loss,
        loss.minimum_squared_loss,
        (1.0, -2.0),
    )

    assert result.point == loss.center
    assert result.squared_loss == loss.minimum_squared_loss


def test_positive_direction_scaling_preserves_point_and_scales_support() -> None:
    solver = NumpyQuadraticSublevelSupportSolver()
    first = solver.solve(_loss(), 2.25, (1.0, 2.0))
    second = solver.solve(_loss(), 2.25, (3.0, 6.0))

    assert second.point == pytest.approx(first.point)
    assert second.support_value == pytest.approx(3.0 * first.support_value)


def test_support_dominates_deterministic_sampled_boundary() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (0.25, -0.5),
        ((3.0, 0.5), (0.5, 1.5)),
        0.125,
    )
    ceiling = 2.125
    direction = np.asarray((0.75, -1.25))
    result = NumpyQuadraticSublevelSupportSolver().solve(
        loss,
        ceiling,
        (float(direction[0]), float(direction[1])),
    )

    matrix = np.asarray(loss.matrix)
    cholesky = np.linalg.cholesky(matrix)
    radius = math.sqrt(ceiling - loss.minimum_squared_loss)
    for angle in np.linspace(0.0, 2.0 * math.pi, 721, endpoint=False):
        unit = np.asarray((math.cos(float(angle)), math.sin(float(angle))))
        delta = radius * np.linalg.solve(cholesky.T, unit)
        sample = np.asarray(loss.center) + delta
        assert float(direction @ sample) <= result.support_value + 1.0e-10


def test_dense_backend_solve_failure_is_translated_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = np.linalg.LinAlgError("forced dense solve failure")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(np.linalg, "solve", fail)

    with pytest.raises(
        ValueError, match="dense quadratic support solve failed"
    ) as caught:
        NumpyQuadraticSublevelSupportSolver().solve(
            _loss(),
            1.25,
            (1.0, 0.0),
        )

    assert caught.value.__cause__ is failure


def test_completed_m3_support_regression() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (0.0, 1.0),
        ((1.0, 0.0), (0.0, 0.21612337988275399)),
        0.0,
    )
    ceiling = 0.03**2

    result = NumpyQuadraticSublevelSupportSolver().solve(
        loss,
        ceiling,
        (0.0, -1.0),
    )

    expected_y = 1.0 - math.sqrt(ceiling / 0.21612337988275399)
    assert result.point == pytest.approx((0.0, expected_y))
    assert result.squared_loss == pytest.approx(ceiling)
