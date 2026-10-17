from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import projectkoios.optimization.alignment.numpy.batched_procrustes as facade
from projectkoios.optimization.alignment.numpy.batched_procrustes.solver import (
    ResidualArrays,
)
from tests.projectkoios.optimization.alignment.fixtures import (
    BatchedProcrustesEvidenceFixture,
)


@dataclass(frozen=True, slots=True)
class _ResidualArraysCase:
    evidence: ResidualArrays
    source_powers: np.ndarray
    target_powers: np.ndarray
    squared_residuals: np.ndarray
    residual_norms: np.ndarray
    residual_cutoffs: np.ndarray


@pytest.fixture
def residual_arrays_case(
    procrustes_evidence: BatchedProcrustesEvidenceFixture,
) -> _ResidualArraysCase:
    """Return shared residual inputs and their grouped evidence."""
    source_powers = procrustes_evidence.source_powers
    target_powers = procrustes_evidence.target_powers
    squared_residuals = procrustes_evidence.squared_residuals
    residual_norms = procrustes_evidence.residual_norms
    residual_cutoffs = procrustes_evidence.residual_cutoffs
    return _ResidualArraysCase(
        evidence=ResidualArrays(
            source_powers=source_powers,
            target_powers=target_powers,
            squared_residuals=squared_residuals,
            residual_norms=residual_norms,
            residual_cutoffs=residual_cutoffs,
        ),
        source_powers=source_powers,
        target_powers=target_powers,
        squared_residuals=squared_residuals,
        residual_norms=residual_norms,
        residual_cutoffs=residual_cutoffs,
    )


def test_residual_arrays_groups_exact_solver_references(
    residual_arrays_case: _ResidualArraysCase,
) -> None:
    evidence = residual_arrays_case.evidence
    assert evidence.source_powers is residual_arrays_case.source_powers
    assert evidence.target_powers is residual_arrays_case.target_powers
    assert evidence.squared_residuals is residual_arrays_case.squared_residuals
    assert evidence.residual_norms is residual_arrays_case.residual_norms
    assert evidence.residual_cutoffs is residual_arrays_case.residual_cutoffs


def test_residual_arrays_has_frozen_slots_and_is_not_a_facade_export(
    residual_arrays_case: _ResidualArraysCase,
) -> None:
    evidence = residual_arrays_case.evidence
    assert not hasattr(evidence, "__dict__")
    assert not hasattr(facade, "ResidualArrays")
    with pytest.raises(FrozenInstanceError):
        evidence.squared_residuals = np.array(  # type: ignore[misc]
            [1.0], dtype=np.float64
        )
