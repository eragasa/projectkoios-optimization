from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.quadratic.constrained import (  # noqa: E402
    QuadraticSublevelSupportResult,
    QuadraticSublevelSupportSolver,
)
from projectkoios.optimization.quadratic.numpy.constrained import (  # noqa: E402
    NumpyQuadraticSublevelSupportSolver,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)

pytestmark = pytest.mark.integration


class _ChangedDirectionSupportSolver(QuadraticSublevelSupportSolver):
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        return QuadraticSublevelSupportResult(
            loss.center,
            tuple(-value for value in direction),
            0.0,
            squared_loss_ceiling,
        )


class _OffBoundarySupportSolver(QuadraticSublevelSupportSolver):
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult:
        point = loss.center
        return QuadraticSublevelSupportResult(
            point,
            direction,
            sum(left * right for left, right in zip(direction, point, strict=True)),
            loss.minimum_squared_loss,
        )


def _loss() -> PositiveDefiniteQuadraticLoss:
    return PositiveDefiniteQuadraticLoss(
        (1.0, -1.0),
        ((2.0, 0.0), (0.0, 8.0)),
        0.25,
    )


def test_support_solver_is_nominal_and_builtin_is_final() -> None:
    assert QuadraticSublevelSupportSolver.__abstractmethods__
    assert issubclass(
        NumpyQuadraticSublevelSupportSolver,
        QuadraticSublevelSupportSolver,
    )
    assert NumpyQuadraticSublevelSupportSolver.__final__ is True


def test_final_template_rejects_inconsistent_external_results() -> None:
    with pytest.raises(ValueError, match="normalized direction"):
        _ChangedDirectionSupportSolver().solve(_loss(), 1.25, (1.0, 0.0))
    with pytest.raises(ValueError, match="requested boundary"):
        _OffBoundarySupportSolver().solve(_loss(), 1.25, (1.0, 0.0))


def test_support_solver_rejects_empty_zero_and_mismatched_requests() -> None:
    solver = NumpyQuadraticSublevelSupportSolver()

    with pytest.raises(ValueError, match="empty"):
        solver.solve(_loss(), 0.2, (1.0, 0.0))
    with pytest.raises(ValueError, match="nonzero"):
        solver.solve(_loss(), 1.0, (0.0, 0.0))
    with pytest.raises(ValueError, match="dimension"):
        solver.solve(_loss(), 1.0, (1.0,))
    with pytest.raises(TypeError, match="squared_loss_ceiling"):
        solver.solve(_loss(), True, (1.0, 0.0))  # type: ignore[arg-type]
