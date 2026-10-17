from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import projectkoios.optimization.alignment.numpy.batched_procrustes as facade
from projectkoios.optimization.alignment.numpy.batched_procrustes.contract import (
    NumpyBatchedComplexUnitaryProcrustesResult,
    ResultArrays,
)
from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
)
from tests.projectkoios.optimization.alignment.fixtures import (
    BatchedProcrustesEvidenceFixture,
)


@dataclass(frozen=True, slots=True)
class _ResultArraysCase:
    record: ResultArrays
    expected_fields: tuple[np.ndarray, ...]


@pytest.fixture
def result_arrays_case(
    procrustes_evidence: BatchedProcrustesEvidenceFixture,
) -> _ResultArraysCase:
    """Return shared expected references and their construction record."""
    fields = (
        procrustes_evidence.completions,
        procrustes_evidence.spectra,
        procrustes_evidence.ranks,
        procrustes_evidence.full_rank,
        procrustes_evidence.rank_cutoffs,
        procrustes_evidence.source_powers,
        procrustes_evidence.target_powers,
        procrustes_evidence.squared_residuals,
        procrustes_evidence.residual_norms,
        procrustes_evidence.residual_cutoffs,
    )
    return _ResultArraysCase(
        record=ResultArrays(
            completions=fields[0],
            spectra=fields[1],
            ranks=fields[2],
            full_rank=fields[3],
            rank_cutoffs=fields[4],
            source_powers=fields[5],
            target_powers=fields[6],
            squared_residuals=fields[7],
            residual_norms=fields[8],
            residual_cutoffs=fields[9],
        ),
        expected_fields=fields,
    )


def test_result_arrays_groups_exact_evidence_references(
    result_arrays_case: _ResultArraysCase,
) -> None:
    record = result_arrays_case.record
    actual_fields = (
        record.completions,
        record.spectra,
        record.ranks,
        record.full_rank,
        record.rank_cutoffs,
        record.source_powers,
        record.target_powers,
        record.squared_residuals,
        record.residual_norms,
        record.residual_cutoffs,
    )

    assert all(
        actual is expected
        for actual, expected in zip(
            actual_fields,
            result_arrays_case.expected_fields,
            strict=True,
        )
    )


def test_shared_evidence_satisfies_public_result_relations(
    procrustes_evidence: BatchedProcrustesEvidenceFixture,
) -> None:
    result = NumpyBatchedComplexUnitaryProcrustesResult.from_arrays(
        row_count=2,
        tolerance=DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
        right_unitary_completions=procrustes_evidence.completions,
        cross_singular_values=procrustes_evidence.spectra,
        numerical_ranks=procrustes_evidence.ranks,
        full_rank=procrustes_evidence.full_rank,
        applied_rank_cutoffs=procrustes_evidence.rank_cutoffs,
        source_squared_frobenius_norms=procrustes_evidence.source_powers,
        target_squared_frobenius_norms=procrustes_evidence.target_powers,
        optimal_squared_residuals=procrustes_evidence.squared_residuals,
        optimal_residual_frobenius_norms=procrustes_evidence.residual_norms,
        applied_residual_roundoff_cutoffs=procrustes_evidence.residual_cutoffs,
        backend_identifier="fixture.svd",
        backend_version="1",
    )

    assert result.all_full_rank
    assert result.rank_deficient_count == 0


def test_result_arrays_has_frozen_slots_and_is_not_a_facade_export(
    result_arrays_case: _ResultArraysCase,
) -> None:
    record = result_arrays_case.record
    assert not hasattr(record, "__dict__")
    assert not hasattr(facade, "ResultArrays")
    with pytest.raises(FrozenInstanceError):
        record.ranks = np.array([1], dtype=np.intp)  # type: ignore[misc]
