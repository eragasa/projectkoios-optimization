from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
optimize: Any = pytest.importorskip("scipy.optimize")

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

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize(
    ("center", "matrix", "bounds"),
    (
        (
            (-3.0, 1.2),
            ((8.0, 3.0), (3.0, 2.0)),
            ((-0.5, 2.0), (-2.0, 0.25)),
        ),
        (
            (2.5, -3.0),
            ((1.0, -0.9), (-0.9, 1.5)),
            ((-2.0, 0.5), (-0.25, 2.0)),
        ),
        (
            (0.2, 3.0),
            ((25.0, 4.0), (4.0, 1.0)),
            ((-1.0, 1.0), (-2.0, 0.4)),
        ),
        (
            (-1.7, -2.2),
            ((0.5, 0.3), (0.3, 4.0)),
            ((-0.5, 1.2), (-1.0, 0.75)),
        ),
        (
            (4.0, 4.0),
            ((3.0, -1.25), (-1.25, 0.75)),
            ((-2.0, 1.0), (-3.0, 0.5)),
        ),
    ),
)
def test_rectangle_minimum_agrees_with_independent_bounded_minimizer(
    center: tuple[float, float],
    matrix: tuple[tuple[float, float], tuple[float, float]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    loss = PositiveDefiniteQuadraticLoss(center, matrix, 0.125)
    space = ParameterSpace(
        (
            ParameterBound("x", *bounds[0]),
            ParameterBound("y", *bounds[1]),
        )
    )
    actual = NumpyRectangleQuadraticMinimizer2D().solve(loss, space)

    matrix_array = np.asarray(matrix)
    center_array = np.asarray(center)

    def objective(point: Any) -> float:
        delta = np.asarray(point) - center_array
        return 0.125 + float(delta @ matrix_array @ delta)

    lower = np.asarray((bounds[0][0], bounds[1][0]))
    upper = np.asarray((bounds[0][1], bounds[1][1]))
    starts = (
        np.clip(center_array, lower, upper),
        lower,
        upper,
        np.asarray((lower[0], upper[1])),
        np.asarray((upper[0], lower[1])),
        0.5 * (lower + upper),
    )
    references = tuple(
        optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 2000},
        )
        for start in starts
    )
    assert all(reference.success for reference in references)
    reference = min(references, key=lambda item: float(item.fun))

    assert actual.coordinates == pytest.approx(
        tuple(float(value) for value in reference.x),
        rel=1.0e-7,
        abs=1.0e-7,
    )
    assert actual.squared_loss == pytest.approx(
        float(reference.fun),
        rel=1.0e-10,
        abs=1.0e-12,
    )
