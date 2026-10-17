import inspect

import pytest

from projectkoios.optimization.least_squares.linear import (
    LinearLeastSquaresMatrix,
    LinearLeastSquaresResult,
    LinearLeastSquaresSolver,
)
from projectkoios.optimization.least_squares.numpy.linear import (
    NumpyLinearLeastSquaresSolver,
)

pytestmark = pytest.mark.integration


class _InconsistentSolver(LinearLeastSquaresSolver):
    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        return LinearLeastSquaresResult(
            ((0.0 + 0.0j,), (0.0 + 0.0j,)),
            target_matrix,
            ((0.0 + 0.0j,), (0.0 + 0.0j,)),
            (0.0,),
            (0.0,),
            0.0,
            0.0,
            0.0,
            2,
            (1.0, 1.0),
            1.0,
        )


class _NonstationarySolver(LinearLeastSquaresSolver):
    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        return LinearLeastSquaresResult(
            ((0.0 + 0.0j,),),
            ((0.0 + 0.0j,), (0.0 + 0.0j,)),
            ((0.0 + 0.0j,), (-2.0 + 0.0j,)),
            (2.0,),
            (2.0,),
            2.0,
            2.0,
            2.0,
            1,
            (2.0**0.5,),
            1.0,
        )


class _ForgedSpectrumSolver(LinearLeastSquaresSolver):
    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        return LinearLeastSquaresResult(
            target_matrix,
            target_matrix,
            ((0.0 + 0.0j,), (0.0 + 0.0j,)),
            (0.0,),
            (0.0,),
            0.0,
            0.0,
            0.0,
            0,
            (0.0, 0.0),
            None,
        )


class _WrongResultSolver(LinearLeastSquaresSolver):
    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        return object()  # type: ignore[return-value]


def test_solver_boundary_is_nominal_abstract_and_public_solve_is_final() -> None:
    assert inspect.isabstract(LinearLeastSquaresSolver)
    assert issubclass(NumpyLinearLeastSquaresSolver, LinearLeastSquaresSolver)
    assert vars(LinearLeastSquaresSolver.solve)["__final__"] is True
    assert vars(NumpyLinearLeastSquaresSolver)["__final__"] is True


def test_final_template_rejects_nonconforming_backend_results() -> None:
    design = ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j))
    targets = ((1.0 + 0.0j,), (2.0 + 0.0j,))

    with pytest.raises(ValueError, match="fitted_values are inconsistent"):
        _InconsistentSolver().solve(design, targets)
    with pytest.raises(TypeError, match="LinearLeastSquaresResult"):
        _WrongResultSolver().solve(design, targets)
    with pytest.raises(ValueError, match="stationarity"):
        _NonstationarySolver().solve(
            ((1.0 + 0.0j,), (1.0 + 0.0j,)),
            ((0.0 + 0.0j,), (2.0 + 0.0j,)),
        )
    with pytest.raises(ValueError, match="singular_values are inconsistent"):
        _ForgedSpectrumSolver().solve(design, targets)


@pytest.mark.parametrize(
    ("design", "targets", "weights", "rank_tolerance", "error", "message"),
    (
        ((), ((1.0,),), None, None, TypeError, "design_matrix"),
        (((1.0,),), ((1.0,), (2.0,)), None, None, ValueError, "row count"),
        (((1.0,),), ((1.0,),), (0.0,), None, ValueError, "strictly positive"),
        (((1.0,),), ((1.0,),), (1.0, 2.0), None, ValueError, "one value"),
        (((1.0,),), ((1.0,),), None, -1.0, ValueError, "at least zero"),
        (((1.0,),), ((1.0,),), None, 1.0, ValueError, "less than one"),
        (((1.0,),), ((complex("nan"),),), None, None, ValueError, "finite"),
    ),
)
def test_final_template_rejects_invalid_inputs(
    design: object,
    targets: object,
    weights: object,
    rank_tolerance: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        NumpyLinearLeastSquaresSolver().solve(
            design,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            weights=weights,  # type: ignore[arg-type]
            rank_tolerance=rank_tolerance,  # type: ignore[arg-type]
        )
