"""Direct tests for scalar Procrustes result-validation owners."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import pytest

from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesResult,
    ComplexUnitaryProcrustesStatus,
    ComplexUnitaryProcrustesTolerance,
    _ScalarResultStructure,
    _validate_scalar_result_backend_evidence,
    _validate_scalar_result_rank_evidence,
    _validate_scalar_result_residual_evidence,
    _validate_scalar_result_structure,
    _validate_scalar_result_transformation,
)
from tests.projectkoios.optimization.alignment.fixtures import UniqueScalarResultCase

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _ScalarResultProbe:
    """Mirror result attributes while permitting one deliberately invalid field."""

    row_count: int
    column_count: int
    tolerance: ComplexUnitaryProcrustesTolerance
    status: ComplexUnitaryProcrustesStatus | str
    transformation: tuple[tuple[complex, ...], ...] | None
    cross_singular_values: tuple[float, ...]
    numerical_rank: int
    applied_rank_cutoff: float
    source_squared_frobenius_norm: float
    target_squared_frobenius_norm: float
    optimal_squared_residual: float
    optimal_residual_frobenius_norm: float
    backend_identifier: str
    backend_version: str

    @classmethod
    def from_result(
        cls,
        result: ComplexUnitaryProcrustesResult,
        /,
    ) -> _ScalarResultProbe:
        """Copy one valid result into an immutable validation probe."""
        return cls(
            row_count=result.row_count,
            column_count=result.column_count,
            tolerance=result.tolerance,
            status=result.status,
            transformation=result.transformation,
            cross_singular_values=result.cross_singular_values,
            numerical_rank=result.numerical_rank,
            applied_rank_cutoff=result.applied_rank_cutoff,
            source_squared_frobenius_norm=result.source_squared_frobenius_norm,
            target_squared_frobenius_norm=result.target_squared_frobenius_norm,
            optimal_squared_residual=result.optimal_squared_residual,
            optimal_residual_frobenius_norm=result.optimal_residual_frobenius_norm,
            backend_identifier=result.backend_identifier,
            backend_version=result.backend_version,
        )

    def as_result(self) -> ComplexUnitaryProcrustesResult:
        """Expose the structural probe to a single validation owner."""
        return cast(ComplexUnitaryProcrustesResult, self)


def test_structure_validation_retains_dimension_and_complete_spectrum(
    unique_scalar_result: UniqueScalarResultCase,
) -> None:
    case = unique_scalar_result
    probe = _ScalarResultProbe.from_result(case.result)

    structure = _validate_scalar_result_structure(case.result)

    assert structure == _ScalarResultStructure(
        column_count=case.dimension,
        singular_values=(case.singular_value,),
    )
    with pytest.raises(TypeError, match="status"):
        _validate_scalar_result_structure(replace(probe, status="unique").as_result())


def test_rank_validation_reconstructs_strict_cutoff_rank(
    unique_scalar_result: UniqueScalarResultCase,
) -> None:
    case = unique_scalar_result
    probe = _ScalarResultProbe.from_result(case.result)
    structure = _validate_scalar_result_structure(case.result)
    inconsistent_cutoff = math.nextafter(case.result.applied_rank_cutoff, math.inf)

    assert (
        _validate_scalar_result_rank_evidence(case.result, structure) == case.dimension
    )
    with pytest.raises(ValueError, match="cutoff"):
        _validate_scalar_result_rank_evidence(
            replace(
                probe,
                applied_rank_cutoff=inconsistent_cutoff,
            ).as_result(),
            structure,
        )


def test_residual_validation_reconstructs_optimum(
    unique_scalar_result: UniqueScalarResultCase,
) -> None:
    case = unique_scalar_result
    probe = _ScalarResultProbe.from_result(case.result)
    structure = _validate_scalar_result_structure(case.result)
    inconsistent_residual = math.nextafter(case.optimal_squared_residual, math.inf)

    _validate_scalar_result_residual_evidence(
        case.result,
        structure.singular_values,
    )

    with pytest.raises(ValueError, match="squared_residual"):
        _validate_scalar_result_residual_evidence(
            replace(
                probe,
                optimal_squared_residual=inconsistent_residual,
            ).as_result(),
            structure.singular_values,
        )


def test_transformation_validation_enforces_full_rank_unitarity(
    unique_scalar_result: UniqueScalarResultCase,
) -> None:
    case = unique_scalar_result
    probe = _ScalarResultProbe.from_result(case.result)
    structure = _validate_scalar_result_structure(case.result)
    expected_rank = _validate_scalar_result_rank_evidence(case.result, structure)
    nonunitary_scale = 2.0

    _validate_scalar_result_transformation(case.result, structure, expected_rank)

    with pytest.raises(ValueError, match="unitary"):
        _validate_scalar_result_transformation(
            replace(
                probe,
                transformation=((complex(nonunitary_scale, 0.0),),),
            ).as_result(),
            structure,
            expected_rank,
        )


def test_backend_validation_requires_nonempty_trimmed_strings(
    unique_scalar_result: UniqueScalarResultCase,
) -> None:
    case = unique_scalar_result
    probe = _ScalarResultProbe.from_result(case.result)
    absent_identifier = ""

    _validate_scalar_result_backend_evidence(case.result)

    with pytest.raises(ValueError, match="identifier"):
        _validate_scalar_result_backend_evidence(
            replace(
                probe,
                backend_identifier=absent_identifier,
            ).as_result()
        )
