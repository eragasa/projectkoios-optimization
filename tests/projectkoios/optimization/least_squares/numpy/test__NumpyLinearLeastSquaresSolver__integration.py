from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.least_squares.factory import (  # noqa: E402
    LinearLeastSquaresMethod,
    LinearLeastSquaresSolverFactory,
)
from projectkoios.optimization.least_squares.numpy.linear import (  # noqa: E402
    NumpyLinearLeastSquaresSolver,
)

pytestmark = pytest.mark.integration


def _matrix(values: Any) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(value) for value in row) for row in values)


def test_factory_constructs_exact_numpy_solver() -> None:
    solver = LinearLeastSquaresSolverFactory.create(
        LinearLeastSquaresMethod.NUMPY_DENSE
    )

    assert type(solver) is NumpyLinearLeastSquaresSolver


def test_exact_real_multiple_rhs_solution() -> None:
    design = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    coefficients = np.asarray(((2.0, -1.0), (3.0, 4.0)))
    targets = design @ coefficients

    result = NumpyLinearLeastSquaresSolver().solve(
        _matrix(design),
        _matrix(targets),
    )

    assert np.asarray(result.coefficients) == pytest.approx(coefficients)
    assert np.asarray(result.fitted_values) == pytest.approx(targets)
    assert np.asarray(result.residuals) == pytest.approx(np.zeros_like(targets))
    assert result.rank == 2
    assert result.full_rank_condition_number is not None
    assert result.residual_norms == pytest.approx((0.0, 0.0), abs=1.0e-14)
    assert result.weighted_residual_norms == pytest.approx(
        result.residual_norms,
        abs=1.0e-14,
    )


def test_positive_weight_complex_multiple_rhs_matches_direct_reference() -> None:
    coordinates = np.asarray((0.0, 0.13, 0.37, 0.68, 0.91))
    representatives = np.asarray((-1.0, 0.0, 1.0))
    design = np.exp(2.0j * np.pi * np.outer(coordinates, representatives))
    coefficients = np.asarray(
        (
            (1.0 + 0.5j, -0.25j, 0.75 + 0.0j, -1.0 + 0.25j),
            (2.0 + 0.0j, 0.5 - 0.5j, -0.25 + 0.75j, 0.1 + 0.0j),
            (1.0 - 0.5j, 0.25j, 0.75 + 0.0j, -1.0 - 0.25j),
        )
    )
    noise = 1.0e-3 * np.asarray(
        (
            (1.0j, -1.0, 0.5j, 0.25),
            (-0.5, 0.25j, 1.0, -1.0j),
            (0.75j, 0.5, -0.25j, 0.5),
            (-1.0, -0.5j, 0.75, 0.25j),
            (0.5, 1.0j, -0.5, -0.75j),
        )
    )
    targets = design @ coefficients + noise
    weights = (1.0, 2.0, 0.75, 3.0, 1.5)
    square_root_weights = np.sqrt(np.asarray(weights))
    reference, _, rank, singular_values = np.linalg.lstsq(
        square_root_weights[:, None] * design,
        square_root_weights[:, None] * targets,
        rcond=None,
    )

    result = NumpyLinearLeastSquaresSolver().solve(
        _matrix(design),
        _matrix(targets),
        weights=weights,
    )

    residuals = design @ reference - targets
    assert np.asarray(result.coefficients) == pytest.approx(reference)
    assert np.asarray(result.fitted_values) == pytest.approx(design @ reference)
    assert np.asarray(result.residuals) == pytest.approx(residuals)
    assert result.rank == int(rank)
    assert result.singular_values == pytest.approx(tuple(singular_values))
    assert result.residual_norms == pytest.approx(
        tuple(np.linalg.norm(residuals, axis=0))
    )
    assert result.weighted_residual_norms == pytest.approx(
        tuple(np.linalg.norm(square_root_weights[:, None] * residuals, axis=0))
    )
    assert result.residual_frobenius_norm == pytest.approx(np.linalg.norm(residuals))
    assert result.weighted_residual_frobenius_norm == pytest.approx(
        np.linalg.norm(square_root_weights[:, None] * residuals)
    )
    assert result.maximum_row_residual_norm == pytest.approx(
        np.max(np.linalg.norm(residuals, axis=1))
    )


def test_multiple_rhs_matches_separate_column_fits() -> None:
    design = np.asarray(
        (
            (1.0 + 0.0j, 0.5 - 0.25j),
            (0.0 + 1.0j, -1.0 + 0.5j),
            (2.0 - 0.5j, 0.25 + 1.0j),
            (-0.5 + 0.25j, 1.5 + 0.0j),
        )
    )
    targets = np.asarray(
        (
            (1.0 + 1.0j, 2.0 - 1.0j),
            (0.5 - 0.25j, -1.0 + 0.0j),
            (3.0 + 0.5j, 0.25 + 1.5j),
            (-0.75 + 0.25j, 2.0 + 0.5j),
        )
    )
    weights = (1.0, 2.0, 3.0, 4.0)
    solver = NumpyLinearLeastSquaresSolver()

    combined = solver.solve(_matrix(design), _matrix(targets), weights=weights)
    separate = tuple(
        solver.solve(
            _matrix(design),
            _matrix(targets[:, column : column + 1]),
            weights=weights,
        )
        for column in range(targets.shape[1])
    )

    for column, result in enumerate(separate):
        assert np.asarray(combined.coefficients)[:, column] == pytest.approx(
            np.asarray(result.coefficients)[:, 0]
        )
        assert combined.residual_norms[column] == pytest.approx(
            result.residual_norms[0]
        )


def test_rank_deficiency_is_reported_without_condition_number() -> None:
    design = ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0))
    targets = ((1.0,), (2.0,), (3.0,))

    result = NumpyLinearLeastSquaresSolver().solve(design, targets)

    assert result.rank == 1
    assert result.full_rank_condition_number is None


def test_explicit_rank_tolerance_controls_reported_rank() -> None:
    design = ((1.0, 0.0), (0.0, 1.0e-8))
    targets = ((1.0,), (1.0e-8,))
    solver = NumpyLinearLeastSquaresSolver()

    default = solver.solve(design, targets)
    truncated = solver.solve(design, targets, rank_tolerance=1.0e-6)

    assert default.rank == 2
    assert default.full_rank_condition_number == pytest.approx(1.0e8)
    assert truncated.rank == 1
    assert truncated.full_rank_condition_number is None


def test_explicit_truncation_accepts_residual_in_discarded_singular_direction() -> None:
    result = NumpyLinearLeastSquaresSolver().solve(
        ((1.0, 0.0), (0.0, 0.5)),
        ((0.0,), (1.0,)),
        rank_tolerance=0.75,
    )

    assert result.rank == 1
    assert result.coefficients == ((0.0 + 0.0j,), (0.0 + 0.0j,))
    assert result.residuals == ((0.0 + 0.0j,), (-1.0 + 0.0j,))
    assert result.full_rank_condition_number is None


def test_full_rank_stationarity_handles_canceling_large_products() -> None:
    result = NumpyLinearLeastSquaresSolver().solve(
        ((1.0e155,), (1.0e155,)),
        ((1.0e155,), (-1.0e155,)),
    )

    assert result.rank == 1
    assert result.coefficients[0][0] == pytest.approx(0.0, abs=1.0e-15)
    assert result.residual_frobenius_norm == pytest.approx(2.0**0.5 * 1.0e155)
    assert result.full_rank_condition_number == 1.0


@pytest.mark.parametrize("magnitude", (1.0e200, 1.0e-200))
def test_rank_deficient_residual_norms_remain_scaled_and_nonzero(
    magnitude: float,
) -> None:
    result = NumpyLinearLeastSquaresSolver().solve(
        ((0.0,),),
        ((-magnitude,),),
    )

    assert result.rank == 0
    assert result.residuals == ((complex(magnitude),),)
    assert result.residual_norms == (magnitude,)
    assert result.weighted_residual_norms == (magnitude,)
    assert result.residual_frobenius_norm == magnitude
    assert result.weighted_residual_frobenius_norm == magnitude
    assert result.maximum_row_residual_norm == magnitude


def test_numpy_solve_failure_is_translated_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = np.linalg.LinAlgError("forced least-squares failure")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(np.linalg, "lstsq", fail)

    with pytest.raises(
        ValueError, match="NumPy linear least-squares solve failed"
    ) as caught:
        NumpyLinearLeastSquaresSolver().solve(((1.0,),), ((2.0,),))

    assert caught.value.__cause__ is failure
