"""Unit tests for NumPy batched complex-unitary Procrustes results."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
    NumpyBatchedComplexUnitaryProcrustesSolver,
    contract,
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


def _rebuild_from_arrays(
    result: NumpyBatchedComplexUnitaryProcrustesResult,
    *,
    completions: Complex128Array | None = None,
    spectra: Float64Array | None = None,
    rank_cutoffs: Float64Array | None = None,
) -> NumpyBatchedComplexUnitaryProcrustesResult:
    return NumpyBatchedComplexUnitaryProcrustesResult.from_arrays(
        row_count=result.row_count,
        tolerance=result.tolerance,
        right_unitary_completions=(
            result.right_unitary_completions if completions is None else completions
        ),
        cross_singular_values=(
            result.cross_singular_values if spectra is None else spectra
        ),
        numerical_ranks=result.numerical_ranks,
        full_rank=result.full_rank,
        applied_rank_cutoffs=(
            result.applied_rank_cutoffs if rank_cutoffs is None else rank_cutoffs
        ),
        source_squared_frobenius_norms=result.source_squared_frobenius_norms,
        target_squared_frobenius_norms=result.target_squared_frobenius_norms,
        optimal_squared_residuals=result.optimal_squared_residuals,
        optimal_residual_frobenius_norms=result.optimal_residual_frobenius_norms,
        applied_residual_roundoff_cutoffs=(result.applied_residual_roundoff_cutoffs),
        backend_identifier=result.backend_identifier,
        backend_version=result.backend_version,
    )


def test_result_exposes_no_postconstruction_array_installer() -> None:
    assert not hasattr(
        NumpyBatchedComplexUnitaryProcrustesResult,
        "_install_arrays",
    )
    assert "object.__setattr__" not in inspect.getsource(contract)


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


def test_raw_result_constructor_rejects_writable_array_storage() -> None:
    result = _result()
    writable_completions = result.right_unitary_completions.copy()

    with pytest.raises(ValueError, match="read-only"):
        replace(result, right_unitary_completions=writable_completions)


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


def test_result_defensively_copies_public_arrays() -> None:
    result = _result()
    completions = result.right_unitary_completions.copy()
    spectra = result.cross_singular_values.copy()
    copied = _rebuild_from_arrays(
        result,
        completions=completions,
        spectra=spectra,
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
        _rebuild_from_arrays(result, rank_cutoffs=cutoff)


def test_result_rejects_nonunitary_completion() -> None:
    result = _result()
    completion = result.right_unitary_completions.copy()
    completion[1, 2, 0, 0] = 2.0 + 0.0j

    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        _rebuild_from_arrays(result, completions=completion)


def test_result_is_frozen() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.row_count = 3  # type: ignore[misc]
