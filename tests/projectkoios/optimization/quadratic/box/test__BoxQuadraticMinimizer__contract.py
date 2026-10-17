"""Contract tests for the final dense box-quadratic minimizer template."""

from typing import cast

import pytest

from projectkoios.optimization.quadratic.box import (
    BoxQuadraticBackendResult,
    BoxQuadraticMinimizer,
)
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


class _CandidateMinimizer(BoxQuadraticMinimizer):
    def __init__(
        self,
        coordinates: tuple[float, ...],
        *,
        iteration_count: int = 1,
        active_set_change_count: int = 0,
        linear_solve_count: int = 0,
        identifier: str = "test.backend",
        version: str = "1",
    ) -> None:
        super().__init__()
        self._coordinates = coordinates
        self._iteration_count = iteration_count
        self._active_set_change_count = active_set_change_count
        self._linear_solve_count = linear_solve_count
        self._identifier = identifier
        self._version = version

    @property
    def backend_identifier(self) -> str:
        return self._identifier

    @property
    def backend_version(self) -> str:
        return self._version

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> BoxQuadraticBackendResult:
        del loss, bounds
        return BoxQuadraticBackendResult(
            self._coordinates,
            self._iteration_count,
            self._active_set_change_count,
            self._linear_solve_count,
        )


class _WrongResultMinimizer(BoxQuadraticMinimizer):
    @property
    def backend_identifier(self) -> str:
        return "test.backend"

    @property
    def backend_version(self) -> str:
        return "1"

    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> BoxQuadraticBackendResult:
        del loss, bounds
        return cast(BoxQuadraticBackendResult, (0.0, 0.0))


def _loss() -> PositiveDefiniteQuadraticLoss:
    return PositiveDefiniteQuadraticLoss(
        (0.0, 0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        0.0,
    )


def _bounds() -> ParameterSpace:
    return ParameterSpace(
        (
            ParameterBound("x", -1.0, 1.0),
            ParameterBound("y", -1.0, 1.0),
        )
    )


def test_existing_bounds_contract_rejects_boolean_string_and_nonfinite_values() -> None:
    with pytest.raises(TypeError):
        ParameterBound("x", True, 1.0)
    with pytest.raises(TypeError):
        ParameterBound("x", 0.0, "1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ParameterBound("x", 0.0, float("inf"))
    with pytest.raises(ValueError, match="must not exceed"):
        ParameterBound("x", 2.0, 1.0)


def test_final_template_rejects_wrong_nominal_and_relational_state() -> None:
    solver = _CandidateMinimizer((0.0, 0.0))
    with pytest.raises(TypeError, match="PositiveDefiniteQuadraticLoss"):
        solver.solve(object(), _bounds())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ParameterSpace"):
        solver.solve(_loss(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive dimension"):
        solver.solve(
            PositiveDefiniteQuadraticLoss((0.0,), ((1.0,),), 0.0),
            ParameterSpace(()),
        )
    with pytest.raises(ValueError, match="dimensions must match"):
        solver.solve(
            PositiveDefiniteQuadraticLoss((0.0,), ((1.0,),), 0.0),
            _bounds(),
        )
    with pytest.raises(TypeError, match="BoxQuadraticBackendResult"):
        _WrongResultMinimizer().solve(_loss(), _bounds())
    with pytest.raises(ValueError, match="exceeded maximum_iterations"):
        _CandidateMinimizer((0.0, 0.0), iteration_count=10_001).solve(
            _loss(),
            _bounds(),
        )
    with pytest.raises(ValueError, match="inside bounds"):
        _CandidateMinimizer((2.0, 0.0)).solve(_loss(), _bounds())
    with pytest.raises(ValueError, match="backend_identifier"):
        _CandidateMinimizer((0.0, 0.0), identifier="").solve(_loss(), _bounds())


def test_coordinate_local_kkt_check_prevents_cross_coordinate_masking() -> None:
    loss = PositiveDefiniteQuadraticLoss(
        (-5.0e19, 0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        0.0,
    )
    bounds = ParameterSpace(
        (
            ParameterBound("active", 0.0, 1.0),
            ParameterBound("free", -1.0e10, 1.0e10),
        )
    )

    with pytest.raises(ValueError, match="KKT"):
        _CandidateMinimizer((0.0, 5.0e8)).solve(loss, bounds)


def test_fixed_coordinate_has_no_gradient_sign_restriction() -> None:
    loss = PositiveDefiniteQuadraticLoss((1.0,), ((2.0,),), 0.0)
    bounds = ParameterSpace((ParameterBound("fixed", 0.0, 0.0),))
    result = _CandidateMinimizer(
        (0.0,),
        iteration_count=1,
    ).solve(loss, bounds)

    assert result.gradient == (-4.0,)
    assert result.maximum_kkt_violation == 0.0


def test_validated_final_reevaluation_overflow_is_translated() -> None:
    loss = PositiveDefiniteQuadraticLoss((1.0e308,), ((1.0,),), 0.0)
    bounds = ParameterSpace((ParameterBound("fixed", 0.0, 0.0),))

    with pytest.raises(FloatingPointError, match="quadratic evaluation") as captured:
        _CandidateMinimizer((0.0,), iteration_count=1).solve(loss, bounds)
    assert isinstance(captured.value.__cause__, ValueError)
