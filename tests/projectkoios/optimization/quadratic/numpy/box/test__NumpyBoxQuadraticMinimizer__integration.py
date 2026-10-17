"""Integration tests for dense active-set box minimization."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Float64Array
from projectkoios.optimization.quadratic.box import BoxQuadraticBoundStatus
from projectkoios.optimization.quadratic.numpy.box import NumpyBoxQuadraticMinimizer
from projectkoios.optimization.quadratic.numpy.constrained import (
    NumpyRectangleQuadraticMinimizer2D,
)
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.integration

type _DenseSolve = Callable[[Float64Array, Float64Array], Float64Array]


def _space(bounds: tuple[tuple[float, float], ...]) -> ParameterSpace:
    return ParameterSpace(
        tuple(
            ParameterBound(f"x_{index}", lower, upper)
            for index, (lower, upper) in enumerate(bounds)
        )
    )


@pytest.mark.parametrize(
    ("center", "bounds", "coordinate", "status"),
    (
        ((0.25,), ((-1.0, 1.0),), (0.25,), BoxQuadraticBoundStatus.FREE),
        ((2.0,), ((-1.0, 1.0),), (1.0,), BoxQuadraticBoundStatus.UPPER),
        ((-2.0,), ((-1.0, 1.0),), (-1.0,), BoxQuadraticBoundStatus.LOWER),
        ((3.0,), ((0.5, 0.5),), (0.5,), BoxQuadraticBoundStatus.FIXED),
    ),
)
def test_one_dimensional_and_fixed_coordinate_behavior(
    center: tuple[float],
    bounds: tuple[tuple[float, float]],
    coordinate: tuple[float],
    status: BoxQuadraticBoundStatus,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(center, ((2.0,),), 0.125)
    result = NumpyBoxQuadraticMinimizer().solve(loss, _space(bounds))

    assert result.coordinates == coordinate
    assert result.bound_statuses == (status,)
    assert result.squared_loss == loss.squared_loss(coordinate)
    assert 1 <= result.iteration_count <= result.maximum_iterations
    assert result.maximum_kkt_ratio <= 1.0


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
    ),
)
def test_two_dimensional_result_matches_existing_exact_rectangle_capability(
    center: tuple[float, float],
    matrix: tuple[tuple[float, float], tuple[float, float]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    loss = PositiveDefiniteQuadraticLoss(center, matrix, 0.125)
    space = _space(bounds)

    box_result = NumpyBoxQuadraticMinimizer().solve(loss, space)
    rectangle_result = NumpyRectangleQuadraticMinimizer2D().solve(loss, space)

    assert box_result.coordinates == pytest.approx(
        rectangle_result.coordinates,
        rel=1.0e-13,
        abs=1.0e-13,
    )
    assert box_result.squared_loss == pytest.approx(
        rectangle_result.squared_loss,
        rel=1.0e-13,
        abs=1.0e-13,
    )


def test_face_edge_corner_and_fixed_statuses_are_reported() -> None:
    matrix = (
        (4.0, 1.0, 0.0, 0.0),
        (1.0, 3.0, 0.5, 0.0),
        (0.0, 0.5, 2.0, 0.0),
        (0.0, 0.0, 0.0, 5.0),
    )
    loss = PositiveDefiniteQuadraticLoss((3.0, -4.0, 0.25, 7.0), matrix, 0.5)
    bounds = _space(((0.0, 1.0), (-1.0, 2.0), (-2.0, 2.0), (0.0, 0.0)))

    result = NumpyBoxQuadraticMinimizer().solve(loss, bounds)

    assert result.coordinates == pytest.approx((1.0, -1.0, -0.5, 0.0))
    assert result.bound_statuses == (
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.LOWER,
        BoxQuadraticBoundStatus.FREE,
        BoxQuadraticBoundStatus.FIXED,
    )
    assert result.active_set_change_count >= 0
    assert result.maximum_kkt_ratio <= 1.0


def test_simultaneous_blocking_hits_activate_in_coordinate_order() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, 0.0, 0.0),
        ((3.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        0.0,
    )
    result = NumpyBoxQuadraticMinimizer().solve(
        loss,
        _space(((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
    )

    assert result.coordinates == (1.0, 0.5, 0.5)
    assert result.bound_statuses == (
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.UPPER,
        BoxQuadraticBoundStatus.UPPER,
    )
    assert result.active_set_change_count == 2
    assert result.maximum_kkt_ratio <= 1.0


def test_smallest_index_active_violation_is_released_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, -2.0, -2.0),
        ((14.0, 3.0, 4.0), (3.0, 2.0, 0.0), (4.0, 0.0, 2.0)),
        0.0,
    )
    bounds = _space(((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))
    original_solve = cast(_DenseSolve, np.linalg.solve)
    right_hand_sides: list[tuple[float, ...]] = []

    def recording_solve(
        matrix: Float64Array,
        right_hand_side: Float64Array,
    ) -> Float64Array:
        right_hand_sides.append(
            tuple(float(value) for value in np.asarray(right_hand_side))
        )
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(np.linalg, "solve", recording_solve)
    result = NumpyBoxQuadraticMinimizer().solve(loss, bounds)

    assert right_hand_sides[0] == (-1.0,)
    assert result.coordinates == (1.0, -0.5, 0.0)
    assert result.maximum_kkt_ratio <= 1.0


def test_invalid_active_sign_releases_bound_before_global_corner() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, -10.0),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )
    result = NumpyBoxQuadraticMinimizer().solve(
        loss,
        _space(((-1.0, 1.0), (-1.0, 1.0))),
    )

    assert result.coordinates == (-1.0, -1.0)
    assert result.bound_statuses == (
        BoxQuadraticBoundStatus.LOWER,
        BoxQuadraticBoundStatus.LOWER,
    )
    assert result.active_set_change_count >= 2
    assert result.maximum_kkt_ratio <= 1.0


def test_permutation_covariance_and_repeatability() -> None:
    center = (2.0, -3.0, 0.4, 4.0)
    matrix = (
        (5.0, 1.0, 0.2, 0.0),
        (1.0, 4.0, -0.3, 0.1),
        (0.2, -0.3, 2.0, 0.4),
        (0.0, 0.1, 0.4, 3.0),
    )
    bounds = ((-1.0, 1.0), (-2.0, 0.0), (-0.5, 1.0), (2.0, 2.0))
    solver = NumpyBoxQuadraticMinimizer()
    original = solver.solve(
        PositiveDefiniteQuadraticLoss(center, matrix, 0.25),
        _space(bounds),
    )
    assert (
        solver.solve(
            PositiveDefiniteQuadraticLoss(center, matrix, 0.25),
            _space(bounds),
        )
        == original
    )

    permutation = (2, 0, 3, 1)
    permuted_center = tuple(center[index] for index in permutation)
    permuted_matrix = tuple(
        tuple(matrix[row][column] for column in permutation) for row in permutation
    )
    permuted_bounds = tuple(bounds[index] for index in permutation)
    permuted = solver.solve(
        PositiveDefiniteQuadraticLoss(permuted_center, permuted_matrix, 0.25),
        _space(permuted_bounds),
    )

    assert permuted.coordinates == pytest.approx(
        tuple(original.coordinates[index] for index in permutation)
    )
    assert permuted.gradient == pytest.approx(
        tuple(original.gradient[index] for index in permutation)
    )
    assert permuted.bound_statuses == tuple(
        original.bound_statuses[index] for index in permutation
    )
    assert permuted.squared_loss == pytest.approx(original.squared_loss)


def test_ill_conditioned_admitted_diagonal_quadratic() -> None:
    diagonal = (1.0e-8, 1.0, 1.0e8)
    matrix = tuple(
        tuple(value if row == column else 0.0 for column, value in enumerate(diagonal))
        for row in range(3)
    )
    loss = PositiveDefiniteQuadraticLoss((2.0, -2.0, 0.25), matrix, 0.0)
    result = NumpyBoxQuadraticMinimizer().solve(
        loss,
        _space(((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
    )

    assert result.coordinates == (1.0, -1.0, 0.25)
    assert result.maximum_kkt_ratio <= 1.0


@pytest.mark.parametrize("dimension", (1, 2, 3, 4, 6, 12, 27, 36))
def test_concrete_consumer_dimension_inventory(dimension: int) -> None:
    vector = np.linspace(0.25, 1.0, dimension)
    matrix_array = np.eye(dimension) + 0.05 * np.outer(vector, vector)
    center = tuple(float(value) for value in np.linspace(-0.3, 0.3, dimension))
    matrix = tuple(tuple(float(value) for value in row) for row in matrix_array)
    bound = 0.11426128163399648
    bounds = tuple((-bound, bound) for _ in range(dimension))

    result = NumpyBoxQuadraticMinimizer().solve(
        PositiveDefiniteQuadraticLoss(center, matrix, 0.0),
        _space(bounds),
    )

    assert len(result.coordinates) == dimension
    assert result.maximum_kkt_ratio <= 1.0
    assert result.iteration_count <= result.maximum_iterations


def test_declared_linear_solve_failures_are_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, 0.0),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )
    bounds = _space(((-1.0, 1.0), (-1.0, 1.0)))

    def fail_linalg(
        matrix: Float64Array,
        right_hand_side: Float64Array,
    ) -> Float64Array:
        del matrix, right_hand_side
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(np.linalg, "solve", fail_linalg)
    with pytest.raises(RuntimeError, match="active-set face solve") as captured:
        NumpyBoxQuadraticMinimizer().solve(loss, bounds)
    assert isinstance(captured.value.__cause__, np.linalg.LinAlgError)


def test_nonfinite_direction_and_stagnation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, 0.0),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )
    bounds = _space(((-1.0, 1.0), (-1.0, 1.0)))

    def nonfinite_solve(
        matrix: Float64Array,
        right_hand_side: Float64Array,
    ) -> Float64Array:
        del matrix, right_hand_side
        return np.asarray((float("nan"),), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "solve", nonfinite_solve)
    with pytest.raises(FloatingPointError, match="direction must be finite"):
        NumpyBoxQuadraticMinimizer().solve(loss, bounds)

    def stagnant_solve(
        matrix: Float64Array,
        right_hand_side: Float64Array,
    ) -> Float64Array:
        del matrix, right_hand_side
        return np.asarray((0.0,), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "solve", stagnant_solve)
    with pytest.raises(RuntimeError, match="stagnated"):
        NumpyBoxQuadraticMinimizer().solve(loss, bounds)


def test_undeclared_backend_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, 0.0),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )
    bounds = _space(((-1.0, 1.0), (-1.0, 1.0)))

    def fail_unexpected(
        matrix: Float64Array,
        right_hand_side: Float64Array,
    ) -> Float64Array:
        del matrix, right_hand_side
        raise KeyError("injected")

    monkeypatch.setattr(np.linalg, "solve", fail_unexpected)
    with pytest.raises(KeyError, match="injected"):
        NumpyBoxQuadraticMinimizer().solve(loss, bounds)


def test_iteration_limit_fails_closed_before_returning_unchecked_point() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (2.0, 0.0),
        ((2.0, 1.0), (1.0, 2.0)),
        0.0,
    )
    bounds = _space(((-1.0, 1.0), (-1.0, 1.0)))

    with pytest.raises(RuntimeError, match="maximum_iterations"):
        NumpyBoxQuadraticMinimizer(maximum_iterations=1).solve(loss, bounds)


def test_nominal_boundary_rejects_sparse_quadratic_state() -> None:
    pytest.importorskip("scipy")
    sparse_loss = PositiveDefiniteQuadraticLoss(
        (0.0,),
        ((1.0,),),
        0.0,
        sparse=True,
    )

    with pytest.raises(ValueError, match="dense quadratic loss"):
        NumpyBoxQuadraticMinimizer().solve(
            sparse_loss,
            _space(((-1.0, 1.0),)),
        )


def test_final_evaluation_overflow_is_translated() -> None:
    loss = PositiveDefiniteQuadraticLoss((1.0e308,), ((1.0e-100,),), 0.0)
    with pytest.raises(FloatingPointError, match="quadratic evaluation") as captured:
        NumpyBoxQuadraticMinimizer().solve(
            loss,
            _space(((0.0, 0.0),)),
        )
    assert isinstance(captured.value.__cause__, ValueError)
