"""Contract tests for nominal complex-unitary Procrustes solvers."""

from __future__ import annotations

from typing import cast

import pytest

from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDecomposition,
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesMatrix,
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesStatus,
    ComplexUnitaryProcrustesTolerance,
    ScalarComplexUnitaryProcrustesSolver,
)

pytestmark = pytest.mark.unit


class _ConfiguredSolver(ScalarComplexUnitaryProcrustesSolver):
    __slots__ = ("output", "identifier", "version")

    def __init__(
        self,
        output: ComplexUnitaryProcrustesDecomposition,
        *,
        identifier: str = "test.svd",
        version: str = "1",
    ) -> None:
        self.output = output
        self.identifier = identifier
        self.version = version

    @property
    def _backend_identifier(self) -> str:
        return self.identifier

    @property
    def _backend_version(self) -> str:
        return self.version

    def _svd_validated(
        self,
        cross_matrix: ComplexUnitaryProcrustesMatrix,
        /,
    ) -> ComplexUnitaryProcrustesDecomposition:
        del cross_matrix
        return self.output


class _MalformedOutputSolver(_ConfiguredSolver):
    __slots__ = ()

    def _svd_validated(
        self,
        cross_matrix: ComplexUnitaryProcrustesMatrix,
        /,
    ) -> ComplexUnitaryProcrustesDecomposition:
        del cross_matrix
        return cast(ComplexUnitaryProcrustesDecomposition, "malformed")


class _FailingSolver(ScalarComplexUnitaryProcrustesSolver):
    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        return "failing"

    @property
    def _backend_version(self) -> str:
        return "1"

    def _svd_validated(
        self,
        cross_matrix: ComplexUnitaryProcrustesMatrix,
        /,
    ) -> ComplexUnitaryProcrustesDecomposition:
        del cross_matrix
        raise KeyError("backend defect")


def _identity(dimension: int) -> ComplexUnitaryProcrustesMatrix:
    return tuple(
        tuple(
            1.0 + 0.0j if row == column else 0.0 + 0.0j for column in range(dimension)
        )
        for row in range(dimension)
    )


def _diagonal(values: tuple[float, ...]) -> ComplexUnitaryProcrustesMatrix:
    return tuple(
        tuple(
            complex(values[row], 0.0) if row == column else 0.0 + 0.0j
            for column in range(len(values))
        )
        for row in range(len(values))
    )


def _decomposition(
    singular_values: tuple[float, ...],
    left: ComplexUnitaryProcrustesMatrix,
    right_adjoint: ComplexUnitaryProcrustesMatrix,
) -> ComplexUnitaryProcrustesDecomposition:
    return ComplexUnitaryProcrustesDecomposition(
        singular_values=singular_values,
        left_singular_vectors=left,
        right_singular_vectors_adjoint=right_adjoint,
    )


def test_shared_policy_is_concrete_and_scalar_solver_template_is_final() -> None:
    shared_policy = ComplexUnitaryProcrustesSolver()
    with pytest.raises(TypeError):
        ScalarComplexUnitaryProcrustesSolver()  # type: ignore[abstract]

    assert not hasattr(shared_policy, "solve")
    assert vars(ScalarComplexUnitaryProcrustesSolver.solve)["__final__"] is True


def test_solver_rejects_wrong_definition_type() -> None:
    with pytest.raises(TypeError, match="ComplexUnitaryProcrustesDefinition"):
        _ConfiguredSolver(_decomposition((1.0,), _identity(1), _identity(1))).solve(
            cast(ComplexUnitaryProcrustesDefinition, "wrong")
        )


def test_solver_preserves_source_right_to_target_direction() -> None:
    definition = ComplexUnitaryProcrustesDefinition(
        ((0.0 + 1.0j,),),
        ((1.0 + 0.0j,),),
    )
    solver = _ConfiguredSolver(
        _decomposition(
            (1.0,),
            ((0.0 - 1.0j,),),
            ((1.0 + 0.0j,),),
        )
    )

    result = solver.solve(definition)

    assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
    assert result.transformation == ((0.0 - 1.0j,),)
    assert result.optimal_squared_residual == 0.0


def test_solver_withholds_transform_at_rank_cutoff() -> None:
    tolerance = ComplexUnitaryProcrustesTolerance(0.5, 0.0, 1.0e-12, 1.0e-10)
    definition = ComplexUnitaryProcrustesDefinition(
        _identity(2),
        _diagonal((1.0, 0.5)),
        tolerance,
    )
    solver = _ConfiguredSolver(_decomposition((1.0, 0.5), _identity(2), _identity(2)))

    result = solver.solve(definition)

    assert result.numerical_rank == 1
    assert result.applied_rank_cutoff == 0.5
    assert result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
    assert result.transformation is None
    assert result.optimal_squared_residual == 0.25


def test_solver_withholds_transform_at_maximum_rank_cutoff() -> None:
    tolerance = ComplexUnitaryProcrustesTolerance(
        rank_absolute=1.0e-12,
        rank_relative=1.0e-10,
        validation_absolute=1.0e-12,
        validation_relative=1.0e-10,
        rank_cutoff_rule=ComplexUnitaryProcrustesRankCutoffRule.MAXIMUM,
    )
    definition = ComplexUnitaryProcrustesDefinition(
        _identity(2),
        _diagonal((1.0, 1.0e-10)),
        tolerance,
    )
    solver = _ConfiguredSolver(
        _decomposition((1.0, 1.0e-10), _identity(2), _identity(2))
    )

    result = solver.solve(definition)

    assert result.numerical_rank == 1
    assert result.applied_rank_cutoff == 1.0e-10
    assert result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
    assert result.transformation is None


def test_solver_returns_rank_evidence_for_exact_rank_one_overlap() -> None:
    rank_one = _diagonal((1.0, 0.0))
    solver = _ConfiguredSolver(_decomposition((1.0, 0.0), _identity(2), _identity(2)))

    result = solver.solve(ComplexUnitaryProcrustesDefinition(rank_one, rank_one))

    assert result.cross_singular_values == (1.0, 0.0)
    assert result.numerical_rank == 1
    assert result.transformation is None
    assert result.optimal_residual_frobenius_norm == 0.0


def test_solver_rejects_wrong_backend_output_record() -> None:
    definition = ComplexUnitaryProcrustesDefinition(_identity(1), _identity(1))
    solver = _MalformedOutputSolver(_decomposition((1.0,), _identity(1), _identity(1)))

    with pytest.raises(TypeError, match="ComplexUnitaryProcrustesDecomposition"):
        solver.solve(definition)


@pytest.mark.parametrize(
    ("output", "error", "match"),
    [
        pytest.param(
            _decomposition((1.0, 0.0), _identity(1), _identity(1)),
            ValueError,
            "length",
            id="spectrum-length",
        ),
        pytest.param(
            _decomposition((-1.0,), _identity(1), _identity(1)),
            ValueError,
            "nonnegative",
            id="negative-singular-value",
        ),
        pytest.param(
            _decomposition((float("nan"),), _identity(1), _identity(1)),
            FloatingPointError,
            "finite",
            id="nonfinite-singular-value",
        ),
        pytest.param(
            _decomposition((1.0,), ((2.0 + 0.0j,),), _identity(1)),
            ValueError,
            "unitary",
            id="left-not-unitary",
        ),
        pytest.param(
            _decomposition((1.0,), _identity(1), ((2.0 + 0.0j,),)),
            ValueError,
            "unitary",
            id="right-not-unitary",
        ),
        pytest.param(
            _decomposition((2.0,), _identity(1), _identity(1)),
            ValueError,
            "reconstruction",
            id="inconsistent-reconstruction",
        ),
    ],
)
def test_solver_rejects_inconsistent_backend_decomposition(
    output: ComplexUnitaryProcrustesDecomposition,
    error: type[Exception],
    match: str,
) -> None:
    definition = ComplexUnitaryProcrustesDefinition(_identity(1), _identity(1))

    with pytest.raises(error, match=match):
        _ConfiguredSolver(output).solve(definition)


def test_solver_propagates_undeclared_backend_failure() -> None:
    with pytest.raises(KeyError, match="backend defect"):
        _FailingSolver().solve(
            ComplexUnitaryProcrustesDefinition(_identity(1), _identity(1))
        )


@pytest.mark.parametrize(
    ("identifier", "version", "match"),
    [
        pytest.param("", "1", "identifier", id="empty-identifier"),
        pytest.param("test", "", "version", id="empty-version"),
    ],
)
def test_solver_rejects_empty_backend_metadata(
    identifier: str,
    version: str,
    match: str,
) -> None:
    solver = _ConfiguredSolver(
        _decomposition((1.0,), _identity(1), _identity(1)),
        identifier=identifier,
        version=version,
    )

    with pytest.raises(ValueError, match=match):
        solver.solve(ComplexUnitaryProcrustesDefinition(_identity(1), _identity(1)))


def test_finite_cross_product_overflow_fails_closed_before_backend() -> None:
    value = complex(1.0e308, 0.0)

    with pytest.raises(FloatingPointError, match="matrix product"):
        _FailingSolver().solve(
            ComplexUnitaryProcrustesDefinition(((value,),), ((value,),))
        )
