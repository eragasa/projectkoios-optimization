from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import projectkoios.optimization.alignment.numpy.batched_procrustes as facade
from projectkoios.optimization.alignment.numpy.batched_procrustes.solver import (
    CompletionEvidence,
)
from tests.projectkoios.optimization.alignment.fixtures import (
    BatchedProcrustesEvidenceFixture,
)


@dataclass(frozen=True, slots=True)
class _CompletionEvidenceCase:
    evidence: CompletionEvidence
    completions: np.ndarray
    ranks: np.ndarray
    full_rank: np.ndarray
    rank_cutoffs: np.ndarray


@pytest.fixture
def completion_evidence_case(
    procrustes_evidence: BatchedProcrustesEvidenceFixture,
) -> _CompletionEvidenceCase:
    """Return shared completion inputs and their grouped evidence."""
    completions = procrustes_evidence.completions
    ranks = procrustes_evidence.ranks
    full_rank = procrustes_evidence.full_rank
    rank_cutoffs = procrustes_evidence.rank_cutoffs
    return _CompletionEvidenceCase(
        evidence=CompletionEvidence(
            completions=completions,
            ranks=ranks,
            full_rank=full_rank,
            rank_cutoffs=rank_cutoffs,
        ),
        completions=completions,
        ranks=ranks,
        full_rank=full_rank,
        rank_cutoffs=rank_cutoffs,
    )


def test_completion_evidence_groups_exact_solver_references(
    completion_evidence_case: _CompletionEvidenceCase,
) -> None:
    evidence = completion_evidence_case.evidence
    assert evidence.completions is completion_evidence_case.completions
    assert evidence.ranks is completion_evidence_case.ranks
    assert evidence.full_rank is completion_evidence_case.full_rank
    assert evidence.rank_cutoffs is completion_evidence_case.rank_cutoffs


def test_completion_evidence_has_frozen_slots_and_is_not_a_facade_export(
    completion_evidence_case: _CompletionEvidenceCase,
) -> None:
    evidence = completion_evidence_case.evidence
    assert not hasattr(evidence, "__dict__")
    assert not hasattr(facade, "CompletionEvidence")
    with pytest.raises(FrozenInstanceError):
        evidence.ranks = np.array([1], dtype=np.intp)  # type: ignore[misc]
