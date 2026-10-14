"""Contract tests for nominal Hermitian spectral-subspace solvers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralMatrix,
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceSolver,
    HermitianSpectralSubspaceTolerance,
)

pytestmark = pytest.mark.unit

type _Output = tuple[
    tuple[float, ...],
    tuple[tuple[complex, ...], ...],
]


class _ConfiguredSolver(HermitianSpectralSubspaceSolver):
    __slots__ = ("output", "identifier", "version")

    def __init__(
        self,
        output: object,
        *,
        identifier: object = "test.eigh",
        version: object = "1",
    ) -> None:
        self.output = output
        self.identifier = identifier
        self.version = version

    @property
    def _backend_identifier(self) -> str:
        return cast(str, self.identifier)

    @property
    def _backend_version(self) -> str:
        return cast(str, self.version)

    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> _Output:
        del representative
        return cast(_Output, self.output)


class _FailingSolver(HermitianSpectralSubspaceSolver):
    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        return "failing"

    @property
    def _backend_version(self) -> str:
        return "1"

    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> _Output:
        del representative
        raise KeyError("backend defect")


class _CapturingSolver(HermitianSpectralSubspaceSolver):
    __slots__ = ("representative",)

    def __init__(self) -> None:
        self.representative: HermitianSpectralMatrix | None = None

    @property
    def _backend_identifier(self) -> str:
        return "capturing"

    @property
    def _backend_version(self) -> str:
        return "1"

    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> _Output:
        self.representative = representative
        return ((1.0,), ((1.0 + 0.0j,),))


def _tolerance(
    *,
    hermiticity_absolute: float = 1.0e-12,
    hermiticity_relative: float = 0.0,
    boundary_absolute: float = 1.0e-12,
    boundary_relative: float = 0.0,
    projector_absolute: float = 1.0e-12,
    spectral_absolute: float = 1.0e-12,
    spectral_relative: float = 0.0,
) -> HermitianSpectralSubspaceTolerance:
    return HermitianSpectralSubspaceTolerance(
        hermiticity_absolute,
        hermiticity_relative,
        boundary_absolute,
        boundary_relative,
        projector_absolute,
        spectral_absolute,
        spectral_relative,
    )


def _diagonal_definition(
    diagonal: tuple[float, ...],
    start: int,
    stop: int,
    *,
    tolerance: HermitianSpectralSubspaceTolerance | None = None,
) -> HermitianSpectralSubspaceDefinition:
    dimension = len(diagonal)
    matrix = tuple(
        tuple(
            complex(diagonal[row], 0.0) if row == column else 0.0 + 0.0j
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    return HermitianSpectralSubspaceDefinition(
        matrix,
        start,
        stop,
        _tolerance() if tolerance is None else tolerance,
    )


def _identity_output(diagonal: tuple[float, ...]) -> _Output:
    dimension = len(diagonal)
    return (
        diagonal,
        tuple(
            tuple(
                1.0 + 0.0j if row == column else 0.0 + 0.0j
                for column in range(dimension)
            )
            for row in range(dimension)
        ),
    )


def test_solver_is_abstract_and_public_template_is_final() -> None:
    with pytest.raises(TypeError):
        HermitianSpectralSubspaceSolver()  # type: ignore[abstract]

    assert vars(HermitianSpectralSubspaceSolver.solve)["__final__"] is True


def test_solver_rejects_wrong_definition_type() -> None:
    with pytest.raises(TypeError, match="HermitianSpectralSubspaceDefinition"):
        _ConfiguredSolver(_identity_output((1.0,))).solve(cast(Any, object()))


def test_solver_validates_complete_eigensystem_and_constructs_projector() -> None:
    definition = _diagonal_definition((0.0, 1.0, 2.0, 3.0), 1, 3)

    result = _ConfiguredSolver(_identity_output((0.0, 1.0, 2.0, 3.0))).solve(definition)

    assert result.eigenvalues == (0.0, 1.0, 2.0, 3.0)
    assert result.projector == (
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
    )
    assert result.lower_boundary_gap == 1.0
    assert result.upper_boundary_gap == 1.0
    assert result.diagnostics.eigensystem_residual_maximum_absolute_defect == 0.0
    assert result.diagnostics.projector_idempotence_maximum_absolute_defect == 0.0


def test_solver_measures_original_input_before_explicit_symmetrization() -> None:
    solver = _CapturingSolver()
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 2.0e-9j,),),
        0,
        1,
        _tolerance(hermiticity_absolute=4.0e-9),
    )

    result = solver.solve(definition)

    assert solver.representative == ((1.0 + 0.0j,),)
    assert result.diagnostics.input_hermiticity_maximum_absolute_defect == 4.0e-9
    assert (
        result.diagnostics.representative_adjustment_maximum_absolute_defect == 2.0e-9
    )
    assert result.diagnostics.applied_hermiticity_cutoff == 4.0e-9


def test_solver_rejects_input_above_hermiticity_cutoff_before_backend() -> None:
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 2.1e-9j,),),
        0,
        1,
        _tolerance(hermiticity_absolute=4.0e-9),
    )

    with pytest.raises(ValueError, match="Hermiticity tolerance"):
        _FailingSolver().solve(definition)


def test_solver_accepts_internal_degeneracy_as_complete_selected_block() -> None:
    diagonal = (0.0, 1.0, 1.0, 3.0)
    result = _ConfiguredSolver(_identity_output(diagonal)).solve(
        _diagonal_definition(diagonal, 1, 3)
    )

    assert result.projector[1][1] == 1.0 + 0.0j
    assert result.projector[2][2] == 1.0 + 0.0j


def test_solver_is_invariant_to_column_phases_and_internal_block_rotations() -> None:
    diagonal = (0.0, 1.0, 1.0, 3.0)
    identity = _identity_output(diagonal)[1]
    phases = (1.0j, -1.0 + 0.0j, -1.0j, 1.0 + 0.0j)
    phased = tuple(
        tuple(identity[row][column] * phases[column] for column in range(4))
        for row in range(4)
    )
    rotated = (
        (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.6 + 0.0j, -0.8 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.8 + 0.0j, 0.6 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j),
    )
    definition = _diagonal_definition(diagonal, 1, 3)

    base_result = _ConfiguredSolver((diagonal, identity)).solve(definition)
    phased_result = _ConfiguredSolver((diagonal, phased)).solve(definition)
    rotated_result = _ConfiguredSolver((diagonal, rotated)).solve(definition)

    for row in range(4):
        for column in range(4):
            assert (
                abs(
                    phased_result.projector[row][column]
                    - base_result.projector[row][column]
                )
                <= 1.0e-15
            )
            assert (
                abs(
                    rotated_result.projector[row][column]
                    - base_result.projector[row][column]
                )
                <= 1.0e-15
            )


def test_solver_rejects_boundary_that_cuts_exact_degeneracy() -> None:
    diagonal = (0.0, 1.0, 1.0, 3.0)

    with pytest.raises(ValueError, match="upper boundary"):
        _ConfiguredSolver(_identity_output(diagonal)).solve(
            _diagonal_definition(diagonal, 0, 2)
        )


def test_solver_rejects_boundary_equal_to_declared_cutoff() -> None:
    tolerance = _tolerance(boundary_absolute=1.0)

    with pytest.raises(ValueError, match="upper boundary"):
        _ConfiguredSolver(_identity_output((0.0, 1.0))).solve(
            _diagonal_definition((0.0, 1.0), 0, 1, tolerance=tolerance)
        )


def test_solver_rejects_trace_preserving_wrong_spectrum() -> None:
    wrong = ((-1.0, 1.0, 3.0), _identity_output((0.0, 1.0, 2.0))[1])

    with pytest.raises(ValueError, match="residual|reconstruction"):
        _ConfiguredSolver(wrong).solve(_diagonal_definition((0.0, 1.0, 2.0), 0, 3))


def test_solver_rejects_permuted_eigenvectors_before_projector_construction() -> None:
    # A projector-only first-moment check could accept indices {0, 3} in place of
    # {1, 2}; complete eigenpair validation rejects the inconsistent ordered frame.
    eigenvectors = (
        (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j),
        (0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
    )

    with pytest.raises(ValueError, match="residual|reconstruction"):
        _ConfiguredSolver(((0.0, 1.0, 2.0, 3.0), eigenvectors)).solve(
            _diagonal_definition((0.0, 1.0, 2.0, 3.0), 1, 3)
        )


@pytest.mark.parametrize(
    ("output", "error", "message"),
    (
        (object(), TypeError, "exact tuple"),
        (((1.0,),), ValueError, "eigenvalues and eigenvectors"),
        (((1.0, 2.0), ((1.0 + 0.0j,),)), ValueError, "dimension"),
        (
            ((2.0, 1.0), ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))),
            ValueError,
            "ascending",
        ),
        (((float("nan"),), ((1.0 + 0.0j,),)), FloatingPointError, "finite"),
        (((1.0,), ((complex(float("nan"), 0.0),),)), FloatingPointError, "finite"),
        (((1.0,), ((2.0 + 0.0j,),)), ValueError, "orthonormality"),
    ),
)
def test_solver_rejects_malformed_or_invalid_backend_eigensystem(
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    dimension = (
        2 if output == ((2.0, 1.0), ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))) else 1
    )
    diagonal = (1.0, 2.0) if dimension == 2 else (1.0,)

    with pytest.raises(error, match=message):
        _ConfiguredSolver(output).solve(_diagonal_definition(diagonal, 0, dimension))


@pytest.mark.parametrize(
    ("identifier", "version", "error", "message"),
    (
        ("", "1", ValueError, "backend_identifier"),
        (1, "1", TypeError, "backend_identifier"),
        ("test", "", ValueError, "backend_version"),
        ("test", 1, TypeError, "backend_version"),
    ),
)
def test_solver_validates_trusted_backend_metadata_structure(
    identifier: object,
    version: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _ConfiguredSolver(
            _identity_output((1.0,)),
            identifier=identifier,
            version=version,
        ).solve(_diagonal_definition((1.0,), 0, 1))


def test_solver_propagates_undeclared_backend_exception_unchanged() -> None:
    with pytest.raises(KeyError, match="backend defect"):
        _FailingSolver().solve(_diagonal_definition((1.0,), 0, 1))
