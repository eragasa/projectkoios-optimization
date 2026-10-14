from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

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
from projectkoios.optimization.spaces.parameters import (  # noqa: E402
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.integration


def _space(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterBound("x", *x_bounds),
            ParameterBound("y", *y_bounds),
        )
    )


def test_factory_constructs_exact_numpy_rectangle_solver() -> None:
    solver = RectangleQuadraticMinimizerFactory.create(
        RectangleQuadraticMinimizerMethod.NUMPY_DENSE
    )

    assert type(solver) is NumpyRectangleQuadraticMinimizer2D


def test_interior_center_is_the_first_candidate() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (0.25, 0.75),
        ((2.0, 0.5), (0.5, 3.0)),
        0.125,
    )

    result = NumpyRectangleQuadraticMinimizer2D().solve(
        loss,
        _space((0.0, 1.0), (0.0, 1.0)),
    )

    assert result.coordinates == (0.25, 0.75)
    assert result.squared_loss == 0.125
    assert result.candidate is RectangleQuadraticCandidate.CLIPPED_CENTER


@pytest.mark.parametrize(
    ("center", "expected_coordinates", "expected_candidate"),
    (
        (
            (-1.0, 1.5),
            (0.0, 0.75),
            RectangleQuadraticCandidate.COORDINATE_0_LOWER,
        ),
        (
            (3.0, 0.5),
            (2.0, 1.25),
            RectangleQuadraticCandidate.COORDINATE_0_UPPER,
        ),
        (
            (1.5, -1.0),
            (0.75, 0.0),
            RectangleQuadraticCandidate.COORDINATE_1_LOWER,
        ),
        (
            (0.5, 3.0),
            (1.25, 2.0),
            RectangleQuadraticCandidate.COORDINATE_1_UPPER,
        ),
    ),
)
def test_each_correlated_edge_orientation_finds_its_conditional_minimum(
    center: tuple[float, float],
    expected_coordinates: tuple[float, float],
    expected_candidate: RectangleQuadraticCandidate,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        center,
        ((2.0, 1.5), (1.5, 2.0)),
        0.0,
    )

    result = NumpyRectangleQuadraticMinimizer2D().solve(
        loss,
        _space((0.0, 2.0), (0.0, 2.0)),
    )

    assert result.coordinates == expected_coordinates
    assert result.squared_loss == pytest.approx(0.875)
    assert result.candidate is expected_candidate


def test_corner_and_center_on_boundary_retain_clipped_center() -> None:
    solver = NumpyRectangleQuadraticMinimizer2D()
    corner = solver.solve(
        PositiveDefiniteQuadraticLoss(
            (-1.0, -2.0),
            ((1.0, 0.0), (0.0, 1.0)),
            0.0,
        ),
        _space((0.0, 1.0), (0.0, 1.0)),
    )
    boundary = solver.solve(
        PositiveDefiniteQuadraticLoss(
            (0.0, 0.5),
            ((1.0, 0.0), (0.0, 1.0)),
            0.25,
        ),
        _space((0.0, 1.0), (0.0, 1.0)),
    )

    assert corner.coordinates == (0.0, 0.0)
    assert corner.candidate is RectangleQuadraticCandidate.CLIPPED_CENTER
    assert boundary.coordinates == (0.0, 0.5)
    assert boundary.candidate is RectangleQuadraticCandidate.CLIPPED_CENTER


def test_degenerate_line_and_singleton_have_explicit_tie_order() -> None:
    solver = NumpyRectangleQuadraticMinimizer2D()
    loss = PositiveDefiniteQuadraticLoss(
        (-1.0, 0.5),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )

    line = solver.solve(loss, _space((0.0, 0.0), (-2.0, 2.0)))
    singleton = solver.solve(loss, _space((2.0, 2.0), (3.0, 3.0)))

    assert line.coordinates == (0.0, 0.0)
    assert line.candidate is RectangleQuadraticCandidate.COORDINATE_0_LOWER
    assert singleton.coordinates == (2.0, 3.0)
    assert singleton.candidate is RectangleQuadraticCandidate.CLIPPED_CENTER


def test_completed_m3_numeric_rectangle_regression() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (0.0, 0.7501452176838512),
        ((1.0, 0.0), (0.0, 0.216123379882754)),
        0.09450689673607807,
    )
    bounds = _space((-0.25, 0.25), (0.4, 1.2))

    result = NumpyRectangleQuadraticMinimizer2D().solve(loss, bounds)

    assert result.coordinates == pytest.approx((0.0, 0.7501452176838512))
    assert result.squared_loss == pytest.approx(0.09450689673607807)
    assert result.candidate is RectangleQuadraticCandidate.CLIPPED_CENTER
