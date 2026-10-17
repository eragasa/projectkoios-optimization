"""Named analytical scalar Procrustes fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesResult,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesStatus,
)


@dataclass(frozen=True, slots=True)
class UniqueScalarResultCase:
    """Retain one analytically derived one-dimensional alignment result."""

    result: ComplexUnitaryProcrustesResult
    dimension: int
    singular_value: float
    optimal_squared_residual: float


def unique_scalar_result_case() -> UniqueScalarResultCase:
    """Construct a one-dimensional full-rank result from scalar magnitudes."""
    dimension = 1
    source_magnitude = 2.0
    target_magnitude = 1.0
    singular_value = source_magnitude * target_magnitude
    source_power = source_magnitude**2
    target_power = target_magnitude**2
    optimal_residual = abs(source_magnitude - target_magnitude)
    optimal_squared_residual = optimal_residual**2
    tolerance = DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
    applied_cutoff = ComplexUnitaryProcrustesSolver.rank_cutoff(
        tolerance,
        singular_value,
    )
    result = ComplexUnitaryProcrustesResult(
        row_count=dimension,
        column_count=dimension,
        tolerance=tolerance,
        status=ComplexUnitaryProcrustesStatus.UNIQUE,
        transformation=((complex(1.0, 0.0),),),
        cross_singular_values=(singular_value,),
        numerical_rank=dimension,
        applied_rank_cutoff=applied_cutoff,
        source_squared_frobenius_norm=source_power,
        target_squared_frobenius_norm=target_power,
        optimal_squared_residual=optimal_squared_residual,
        optimal_residual_frobenius_norm=optimal_residual,
        backend_identifier="test.svd",
        backend_version="test-version",
    )
    return UniqueScalarResultCase(
        result=result,
        dimension=dimension,
        singular_value=singular_value,
        optimal_squared_residual=optimal_squared_residual,
    )
