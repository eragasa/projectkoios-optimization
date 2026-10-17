"""Unit tests for complex-unitary Procrustes result invariants."""

from __future__ import annotations

import pytest

from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesResult,
    ComplexUnitaryProcrustesStatus,
)

pytestmark = pytest.mark.unit


def _unique_result() -> ComplexUnitaryProcrustesResult:
    return ComplexUnitaryProcrustesResult(
        row_count=1,
        column_count=1,
        tolerance=DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
        status=ComplexUnitaryProcrustesStatus.UNIQUE,
        transformation=((1.0 + 0.0j,),),
        cross_singular_values=(2.0,),
        numerical_rank=1,
        applied_rank_cutoff=2.0e-12,
        source_squared_frobenius_norm=4.0,
        target_squared_frobenius_norm=1.0,
        optimal_squared_residual=1.0,
        optimal_residual_frobenius_norm=1.0,
        backend_identifier="test.svd",
        backend_version="1",
    )


def _deficient_result() -> ComplexUnitaryProcrustesResult:
    return ComplexUnitaryProcrustesResult(
        row_count=1,
        column_count=2,
        tolerance=DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
        status=ComplexUnitaryProcrustesStatus.RANK_DEFICIENT,
        transformation=None,
        cross_singular_values=(2.0, 0.0),
        numerical_rank=1,
        applied_rank_cutoff=2.0e-12,
        source_squared_frobenius_norm=4.0,
        target_squared_frobenius_norm=1.0,
        optimal_squared_residual=1.0,
        optimal_residual_frobenius_norm=1.0,
        backend_identifier="test.svd",
        backend_version="1",
    )


def test_unique_result_retains_reconstructable_evidence() -> None:
    result = _unique_result()

    assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
    assert result.transformation == ((1.0 + 0.0j,),)
    assert result.numerical_rank == result.column_count
    assert result.optimal_squared_residual == (
        result.source_squared_frobenius_norm
        + result.target_squared_frobenius_norm
        - 2.0 * sum(result.cross_singular_values)
    )


def test_deficient_result_withholds_full_transform() -> None:
    result = _deficient_result()

    assert result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
    assert result.transformation is None
    assert result.numerical_rank < result.column_count


def test_unique_status_requires_transform() -> None:
    values = _unique_result()

    with pytest.raises((TypeError, ValueError), match="transformation"):
        ComplexUnitaryProcrustesResult(
            values.row_count,
            values.column_count,
            values.tolerance,
            values.status,
            None,
            values.cross_singular_values,
            values.numerical_rank,
            values.applied_rank_cutoff,
            values.source_squared_frobenius_norm,
            values.target_squared_frobenius_norm,
            values.optimal_squared_residual,
            values.optimal_residual_frobenius_norm,
            values.backend_identifier,
            values.backend_version,
        )


def test_deficient_status_rejects_null_space_completion() -> None:
    values = _deficient_result()

    with pytest.raises(ValueError, match="must not expose"):
        ComplexUnitaryProcrustesResult(
            values.row_count,
            values.column_count,
            values.tolerance,
            values.status,
            ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j)),
            values.cross_singular_values,
            values.numerical_rank,
            values.applied_rank_cutoff,
            values.source_squared_frobenius_norm,
            values.target_squared_frobenius_norm,
            values.optimal_squared_residual,
            values.optimal_residual_frobenius_norm,
            values.backend_identifier,
            values.backend_version,
        )


def test_unique_result_rejects_nonunitary_transform() -> None:
    values = _unique_result()

    with pytest.raises(ValueError, match="unitary"):
        ComplexUnitaryProcrustesResult(
            values.row_count,
            values.column_count,
            values.tolerance,
            values.status,
            ((2.0 + 0.0j,),),
            values.cross_singular_values,
            values.numerical_rank,
            values.applied_rank_cutoff,
            values.source_squared_frobenius_norm,
            values.target_squared_frobenius_norm,
            values.optimal_squared_residual,
            values.optimal_residual_frobenius_norm,
            values.backend_identifier,
            values.backend_version,
        )
