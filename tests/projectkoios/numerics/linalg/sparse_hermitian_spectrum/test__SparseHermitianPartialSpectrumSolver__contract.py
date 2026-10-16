"""Contract tests for sparse partial Hermitian-spectrum solvers."""

from __future__ import annotations

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.sparse_hermitian_spectrum import (
    SparseComplexCsrMatrix,
    SparseHermitianPartialSpectrumDefinition,
    SparseHermitianPartialSpectrumSolver,
    SparseHermitianSpectrumSelection,
)

pytestmark = pytest.mark.unit


class _ConfiguredSolver(SparseHermitianPartialSpectrumSolver):
    __slots__ = ("calls", "representative")

    def __init__(self) -> None:
        self.calls = 0
        self.representative: SparseComplexCsrMatrix | None = None

    @property
    def _backend_identifier(self) -> str:
        return "test.sparse.eigsh"

    @property
    def _backend_version(self) -> str:
        return "1"

    def _eigsh_validated(
        self,
        representative: SparseComplexCsrMatrix,
        definition: SparseHermitianPartialSpectrumDefinition,
        /,
    ) -> tuple[tuple[float, ...], float]:
        self.calls += 1
        self.representative = representative
        return tuple(
            float(index) for index in range(definition.eigenvalue_count)
        ), 1.0e-12


def _definition(
    matrix: SparseComplexCsrMatrix,
    tolerance: BatchedHermitianSpectrumTolerance,
) -> SparseHermitianPartialSpectrumDefinition:
    return SparseHermitianPartialSpectrumDefinition(
        matrix=matrix,
        eigenvalue_count=1,
        selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
        hermiticity_tolerance=tolerance,
    )


def test_solver_constructs_explicit_sparse_hermitian_representative() -> None:
    matrix = SparseComplexCsrMatrix(
        3,
        (0, 2, 4, 5),
        (0, 1, 0, 1, 2),
        (1.0 + 2.0e-9j, 2.0 + 4.0e-9j, 2.0 - 2.0e-9j, 3.0, 4.0),
    )
    solver = _ConfiguredSolver()

    result = solver.solve(
        _definition(matrix, BatchedHermitianSpectrumTolerance(4.0e-9, 0.0))
    )

    assert solver.calls == 1
    assert solver.representative is not None
    assert solver.representative.indptr == (0, 2, 4, 5)
    assert solver.representative.indices == (0, 1, 0, 1, 2)
    assert solver.representative.data[0] == 1.0 + 0.0j
    assert solver.representative.data[1] == 2.0 + 3.0000000000000004e-9j
    assert solver.representative.data[2] == solver.representative.data[1].conjugate()
    assert result.eigenvalues == (0.0,)
    assert result.residual_maximum_absolute_norm == 1.0e-12
    assert result.matrix_diagnostics.input_hermiticity_maximum_absolute_defect == 4.0e-9


def test_sparse_cutoff_equality_is_accepted() -> None:
    matrix = SparseComplexCsrMatrix(
        3, (0, 1, 2, 3), (0, 1, 2), (1.0 + 2.0e-9j, 2.0, 3.0)
    )
    solver = _ConfiguredSolver()

    solver.solve(_definition(matrix, BatchedHermitianSpectrumTolerance(4.0e-9, 0.0)))

    assert solver.calls == 1


def test_sparse_defect_above_cutoff_fails_before_backend() -> None:
    matrix = SparseComplexCsrMatrix(
        3, (0, 1, 2, 3), (0, 1, 2), (1.0 + 2.1e-9j, 2.0, 3.0)
    )
    solver = _ConfiguredSolver()

    with pytest.raises(ValueError, match="Hermiticity"):
        solver.solve(
            _definition(matrix, BatchedHermitianSpectrumTolerance(4.0e-9, 0.0))
        )

    assert solver.calls == 0
