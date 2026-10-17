from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesBackendEvidence,
    ComplexUnitaryProcrustesDecomposition,
    ComplexUnitaryProcrustesResidualEvidence,
)


def test_decomposition_groups_one_complete_factorization(
    scalar_identity_decomposition: ComplexUnitaryProcrustesDecomposition,
) -> None:
    decomposition = scalar_identity_decomposition

    assert len(decomposition.singular_values) == len(
        decomposition.left_singular_vectors
    )
    replacement = tuple(reversed(decomposition.singular_values))
    with pytest.raises(FrozenInstanceError):
        decomposition.singular_values = replacement  # type: ignore[misc]


@pytest.mark.parametrize(
    ("squared_residual", "roundoff_cutoff", "match"),
    [
        pytest.param(-1.0, 0.0, "squared_residual", id="negative-residual"),
        pytest.param(0.0, -1.0, "roundoff_cutoff", id="negative-cutoff"),
        pytest.param(float("inf"), 0.0, "squared_residual", id="nonfinite-residual"),
    ],
)
def test_residual_evidence_rejects_invalid_values(
    squared_residual: float,
    roundoff_cutoff: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ComplexUnitaryProcrustesResidualEvidence(
            squared_residual=squared_residual,
            roundoff_cutoff=roundoff_cutoff,
        )


def test_residual_evidence_preserves_valid_values() -> None:
    squared_residual = 1.0
    roundoff_cutoff = 2.0 * squared_residual
    evidence = ComplexUnitaryProcrustesResidualEvidence(
        squared_residual,
        roundoff_cutoff,
    )

    assert evidence.squared_residual == squared_residual
    assert evidence.roundoff_cutoff == roundoff_cutoff


@pytest.mark.parametrize(
    ("identifier", "version", "match"),
    [
        pytest.param("", "1", "identifier", id="empty-identifier"),
        pytest.param("svd", "", "version", id="empty-version"),
        pytest.param(" svd", "1", "identifier", id="untrimmed-identifier"),
    ],
)
def test_backend_evidence_rejects_invalid_text(
    identifier: str,
    version: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ComplexUnitaryProcrustesBackendEvidence(identifier, version)


def test_backend_evidence_preserves_valid_text() -> None:
    evidence = ComplexUnitaryProcrustesBackendEvidence("numpy.linalg.svd", "2.0")

    assert evidence.identifier == "numpy.linalg.svd"
    assert evidence.version == "2.0"
