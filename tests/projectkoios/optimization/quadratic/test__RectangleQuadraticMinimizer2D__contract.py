from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.quadratic.constrained import (  # noqa: E402
    RectangleQuadraticCandidate,
    RectangleQuadraticMinimizer2D,
    RectangleQuadraticMinimum,
)
from projectkoios.optimization.quadratic.numpy.constrained import (  # noqa: E402
    NumpyRectangleQuadraticMinimizer2D,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.spaces.parameters import (  # noqa: E402
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.integration


class _InconsistentRectangleSolver(RectangleQuadraticMinimizer2D):
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        return RectangleQuadraticMinimum(
            (bounds.bounds[0].lower, bounds.bounds[1].lower),
            999.0,
            RectangleQuadraticCandidate.COORDINATE_0_LOWER,
        )


class _OutsideRectangleSolver(RectangleQuadraticMinimizer2D):
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum:
        return RectangleQuadraticMinimum(
            (bounds.bounds[0].lower - 1.0, bounds.bounds[1].lower),
            0.0,
            RectangleQuadraticCandidate.COORDINATE_0_LOWER,
        )


def _loss() -> PositiveDefiniteQuadraticLoss:
    return PositiveDefiniteQuadraticLoss(
        (0.5, 0.5),
        ((2.0, 0.5), (0.5, 1.0)),
        0.25,
    )


def _bounds() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterBound("x", 0.0, 1.0),
            ParameterBound("y", 0.0, 1.0),
        )
    )


def test_rectangle_solver_is_nominal_and_builtin_is_final() -> None:
    assert RectangleQuadraticMinimizer2D.__abstractmethods__
    assert issubclass(
        NumpyRectangleQuadraticMinimizer2D,
        RectangleQuadraticMinimizer2D,
    )
    assert NumpyRectangleQuadraticMinimizer2D.__final__ is True


def test_final_template_rejects_inconsistent_external_results() -> None:
    with pytest.raises(ValueError, match="squared_loss"):
        _InconsistentRectangleSolver().solve(_loss(), _bounds())
    with pytest.raises(ValueError, match="inside bounds"):
        _OutsideRectangleSolver().solve(_loss(), _bounds())


def test_rectangle_solver_rejects_wrong_public_inputs() -> None:
    solver = NumpyRectangleQuadraticMinimizer2D()

    with pytest.raises(TypeError, match="loss"):
        solver.solve(object(), _bounds())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bounds"):
        solver.solve(_loss(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimension two"):
        solver.solve(
            PositiveDefiniteQuadraticLoss((0.0,), ((1.0,),), 0.0),
            _bounds(),
        )
