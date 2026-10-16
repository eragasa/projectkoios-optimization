"""Contract tests for the shared batched Hermitian-spectrum template."""

from __future__ import annotations

from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumSolver,
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrices,
)

pytestmark = pytest.mark.unit


class _ConfiguredSolver(BatchedHermitianSpectrumSolver):
    __slots__ = ("output", "identifier", "version", "calls", "representatives")

    def __init__(
        self,
        output: object,
        *,
        identifier: object = "test.eigvalsh",
        version: object = "1",
    ) -> None:
        self.output = output
        self.identifier = identifier
        self.version = version
        self.calls = 0
        self.representatives: HermitianSpectrumMatrices | None = None

    @property
    def _backend_identifier(self) -> str:
        return cast(str, self.identifier)

    @property
    def _backend_version(self) -> str:
        return cast(str, self.version)

    def _eigvalsh_validated(
        self,
        representatives: HermitianSpectrumMatrices,
        /,
    ) -> object:
        self.calls += 1
        self.representatives = representatives
        return self.output


def _tolerance(
    absolute: float = 0.0, relative: float = 0.0
) -> BatchedHermitianSpectrumTolerance:
    return BatchedHermitianSpectrumTolerance(absolute, relative)


def test_empty_batch_short_circuits_backend_without_fabricated_maxima() -> None:
    solver = _ConfiguredSolver(object())

    result = solver.solve(BatchedHermitianSpectrumDefinition.empty(3, _tolerance()))

    assert solver.calls == 0
    assert solver.representatives is None
    assert result.batch_size == 0
    assert result.dimension == 3
    assert result.eigenvalues == ()
    assert result.matrix_diagnostics == ()
    assert result.diagnostics is None


def test_solver_invokes_backend_once_with_all_explicit_representatives() -> None:
    first = (
        (1.0 + 2.0e-9j, 2.0 + 4.0e-9j),
        (2.0 - 2.0e-9j, 3.0 + 0.0j),
    )
    second = ((4.0, 0.0), (0.0, 5.0))
    solver = _ConfiguredSolver(((0.0, 4.0), (4.0, 5.0)))

    result = solver.solve(
        BatchedHermitianSpectrumDefinition(
            (first, second),
            _tolerance(4.0e-9),
        )
    )

    expected_off_diagonal = (first[0][1] + first[1][0].conjugate()) * 0.5
    assert solver.calls == 1
    assert solver.representatives == (
        (
            (1.0 + 0.0j, expected_off_diagonal),
            (expected_off_diagonal.conjugate(), 3.0 + 0.0j),
        ),
        (
            (4.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 5.0 + 0.0j),
        ),
    )
    assert result.diagnostics is not None
    assert result.diagnostics.maximum_input_hermiticity_absolute_defect == 4.0e-9
    assert (
        result.diagnostics.maximum_representative_adjustment_absolute_defect == 2.0e-9
    )


def test_cutoff_equality_is_accepted() -> None:
    solver = _ConfiguredSolver(((1.0,),))

    result = solver.solve(
        BatchedHermitianSpectrumDefinition(
            (((1.0 + 2.0e-9j,),),),
            _tolerance(4.0e-9),
        )
    )

    assert solver.calls == 1
    assert result.matrix_diagnostics[0].input_hermiticity_maximum_absolute_defect == (
        4.0e-9
    )


def test_one_matrix_above_cutoff_fails_whole_batch_before_backend() -> None:
    solver = _ConfiguredSolver(((1.0,), (2.0,)))

    with pytest.raises(ValueError, match=r"matrices\[1\].*Hermiticity"):
        solver.solve(
            BatchedHermitianSpectrumDefinition(
                (((1.0,),), ((2.0 + 2.1e-9j,),)),
                _tolerance(4.0e-9),
            )
        )

    assert solver.calls == 0


def test_relative_cutoff_is_applied_per_matrix_scale() -> None:
    first = ((1.0 + 5.0e-7j,),)
    second = ((100.0 + 5.0e-7j,),)
    tolerance = _tolerance(relative=1.0e-8)

    with pytest.raises(ValueError, match=r"matrices\[0\]"):
        _ConfiguredSolver(((1.0,), (100.0,))).solve(
            BatchedHermitianSpectrumDefinition((first, second), tolerance)
        )

    accepted = _ConfiguredSolver(((100.0,),)).solve(
        BatchedHermitianSpectrumDefinition((second,), tolerance)
    )
    assert accepted.matrix_diagnostics[0].applied_hermiticity_cutoff == pytest.approx(
        1.0e-6,
        rel=0.0,
        abs=1.0e-18,
    )


@pytest.mark.parametrize(
    "output, error, match",
    [
        (((1.0, 2.0),), ValueError, "spectrum count"),
        (((1.0,), (2.0, 3.0)), ValueError, "dimension"),
        (((2.0, 1.0), (3.0, 4.0)), ValueError, "ascending"),
        (((1.0, float("nan")), (3.0, 4.0)), ValueError, "finite"),
        (((1.0, object()), (3.0, 4.0)), TypeError, "real scalar"),
    ],
)
def test_solver_rejects_malformed_backend_output(
    output: object,
    error: type[Exception],
    match: str,
) -> None:
    definition = BatchedHermitianSpectrumDefinition(
        (
            ((1.0, 0.0), (0.0, 2.0)),
            ((3.0, 0.0), (0.0, 4.0)),
        ),
        _tolerance(),
    )

    with pytest.raises(error, match=match):
        _ConfiguredSolver(output).solve(definition)


def test_solver_rejects_wrong_definition_type() -> None:
    with pytest.raises(TypeError, match="BatchedHermitianSpectrumDefinition"):
        _ConfiguredSolver(((1.0,),)).solve(cast(Any, object()))


def test_finite_symmetrization_overflow_fails_before_backend() -> None:
    solver = _ConfiguredSolver(((1.0e308,),))

    with pytest.raises(FloatingPointError, match="symmetrization"):
        solver.solve(
            BatchedHermitianSpectrumDefinition(
                (((complex(1.0e308, 0.0),),),),
                _tolerance(),
            )
        )

    assert solver.calls == 0
