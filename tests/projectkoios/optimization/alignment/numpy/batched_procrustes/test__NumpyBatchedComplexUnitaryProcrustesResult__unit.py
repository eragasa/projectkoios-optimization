"""Unit tests for NumPy batched complex-unitary Procrustes results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)

pytestmark = pytest.mark.unit

CANONICAL_ARRAY_FIELD_NAMES = (
    "right_unitary_completions",
    "cross_singular_values",
    "numerical_ranks",
    "full_rank",
    "applied_rank_cutoffs",
    "source_squared_frobenius_norms",
    "target_squared_frobenius_norms",
    "optimal_squared_residuals",
    "optimal_residual_frobenius_norms",
    "applied_residual_roundoff_cutoffs",
)


def _full_rank_result() -> NumpyBatchedComplexUnitaryProcrustesResult:
    source = np.broadcast_to(
        np.eye(2, dtype=np.complex128),
        (2, 3, 2, 2),
    ).copy()
    return NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(source, source)
    )


def _result() -> NumpyBatchedComplexUnitaryProcrustesResult:
    source = np.broadcast_to(
        np.eye(2, dtype=np.complex128),
        (2, 3, 2, 2),
    ).copy()
    target = source.copy()
    source[0, 1, :, 1] = 0.0 + 0.0j
    target[0, 1, :, 1] = 0.0 + 0.0j
    return NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(source, target)
    )


def test_result_zero_copy_factory_retains_canonical_arrays_by_identity() -> None:
    original = _result()

    retained = NumpyBatchedComplexUnitaryProcrustesResult.from_immutable_arrays(
        row_count=original.row_count,
        tolerance=original.tolerance,
        right_unitary_completions=original.right_unitary_completions,
        cross_singular_values=original.cross_singular_values,
        numerical_ranks=original.numerical_ranks,
        full_rank=original.full_rank,
        applied_rank_cutoffs=original.applied_rank_cutoffs,
        source_squared_frobenius_norms=original.source_squared_frobenius_norms,
        target_squared_frobenius_norms=original.target_squared_frobenius_norms,
        optimal_squared_residuals=original.optimal_squared_residuals,
        optimal_residual_frobenius_norms=(original.optimal_residual_frobenius_norms),
        applied_residual_roundoff_cutoffs=(original.applied_residual_roundoff_cutoffs),
        backend_identifier=original.backend_identifier,
        backend_version=original.backend_version,
    )

    for field_name in CANONICAL_ARRAY_FIELD_NAMES:
        assert getattr(retained, field_name) is getattr(original, field_name)


def test_result_preserves_batch_shape_and_read_only_evidence() -> None:
    result = _result()

    assert result.batch_shape == (2, 3)
    assert result.batch_size == 6
    assert result.row_count == 2
    assert result.column_count == 2
    assert result.right_unitary_completions.shape == (2, 3, 2, 2)
    assert result.cross_singular_values.shape == (2, 3, 2)
    assert result.numerical_ranks.shape == (2, 3)
    assert result.backend_identifier == "numpy.linalg.svd"
    for value in (
        result.right_unitary_completions,
        result.cross_singular_values,
        result.numerical_ranks,
        result.full_rank,
        result.applied_rank_cutoffs,
        result.source_squared_frobenius_norms,
        result.target_squared_frobenius_norms,
        result.optimal_squared_residuals,
        result.optimal_residual_frobenius_norms,
        result.applied_residual_roundoff_cutoffs,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_rank_deficient_completion_is_status_bearing_and_read_only() -> None:
    result = _result()

    assert result.numerical_ranks[0, 1] == 1
    assert not bool(result.full_rank[0, 1])
    assert not result.all_full_rank
    assert result.rank_deficient_count == 1
    assert result.first_rank_deficient_index == (0, 1)
    completion = result.right_unitary_completions[0, 1]
    np.testing.assert_allclose(
        completion.conjugate().T @ completion,
        np.eye(2),
        rtol=0.0,
        atol=1.0e-14,
    )
    with pytest.raises(ValueError):
        completion[0, 0] = 2.0 + 0.0j
    with pytest.raises(ValueError, match=r"batch index \(0, 1\) is rank deficient"):
        result.require_unique_right_unitary_transformations()


def test_full_rank_result_exposes_fail_closed_unique_transformations() -> None:
    result = _full_rank_result()

    assert result.all_full_rank
    assert result.rank_deficient_count == 0
    assert result.first_rank_deficient_index is None
    transformations = result.require_unique_right_unitary_transformations()
    assert transformations is result.right_unitary_completions
    assert not transformations.flags.writeable


def test_result_is_frozen() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.row_count = 3  # type: ignore[misc]
