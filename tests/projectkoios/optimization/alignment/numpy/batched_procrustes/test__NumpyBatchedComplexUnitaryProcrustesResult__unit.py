"""Unit tests for NumPy batched complex-unitary Procrustes results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)

pytestmark = pytest.mark.unit


def _result() -> NumpyBatchedComplexUnitaryProcrustesResult:
    source = np.broadcast_to(
        np.eye(2, dtype=np.complex128),
        (2, 3, 2, 2),
    ).copy()
    target = source.copy()
    source[0, 1, :, 1] = 0.0 + 0.0j
    target[0, 1, :, 1] = 0.0 + 0.0j
    return NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, target)
    )


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
    completion = result.right_unitary_completions[0, 1]
    np.testing.assert_allclose(
        completion.conjugate().T @ completion,
        np.eye(2),
        rtol=0.0,
        atol=1.0e-14,
    )
    with pytest.raises(ValueError):
        completion[0, 0] = 2.0 + 0.0j


def test_result_defensively_copies_public_arrays() -> None:
    result = _result()
    completions = result.right_unitary_completions.copy()
    spectra = result.cross_singular_values.copy()
    copied = replace(
        result,
        right_unitary_completions=completions,
        cross_singular_values=spectra,
    )
    completions[...] = 4.0 + 0.0j
    spectra[...] = 9.0

    assert not np.all(copied.right_unitary_completions == 4.0 + 0.0j)
    assert not np.all(copied.cross_singular_values == 9.0)


def test_result_rejects_rank_exceeding_matrix_row_count() -> None:
    result = _result()

    with pytest.raises(ValueError, match="row-count"):
        replace(result, row_count=1)


def test_result_rejects_inconsistent_rank_cutoff() -> None:
    result = _result()
    cutoff = result.applied_rank_cutoffs.copy()
    cutoff[0, 0] += 1.0

    with pytest.raises(ValueError, match="cutoff"):
        replace(result, applied_rank_cutoffs=cutoff)


def test_result_rejects_nonunitary_completion() -> None:
    result = _result()
    completion = result.right_unitary_completions.copy()
    completion[1, 2, 0, 0] = 2.0 + 0.0j

    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        replace(result, right_unitary_completions=completion)


def test_result_is_frozen() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.row_count = 3  # type: ignore[misc]
