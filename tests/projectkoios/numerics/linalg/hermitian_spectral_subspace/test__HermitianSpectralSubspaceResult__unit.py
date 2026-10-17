"""Unit tests for Hermitian spectral-subspace results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDiagnostics,
    HermitianSpectralSubspaceResult,
    HermitianSpectralSubspaceTolerance,
)

pytestmark = pytest.mark.unit


def _tolerance() -> HermitianSpectralSubspaceTolerance:
    return HermitianSpectralSubspaceTolerance(*(1.0e-12,) * 7)


def _diagnostics() -> HermitianSpectralSubspaceDiagnostics:
    return HermitianSpectralSubspaceDiagnostics(*(0.0,) * 18)


def test_lower_edge_result_retains_correlated_values_and_is_frozen() -> None:
    eigenvalues = (1.0, 3.0)
    projector = ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 0.0 + 0.0j))
    result = HermitianSpectralSubspaceResult(
        dimension=2,
        start_index=0,
        stop_index=1,
        tolerance=_tolerance(),
        eigenvalues=eigenvalues,
        projector=projector,
        lower_boundary_gap=None,
        upper_boundary_gap=2.0,
        diagnostics=_diagnostics(),
        backend_identifier="example.eigh",
        backend_version="1.2.3",
    )

    assert result.eigenvalues is eigenvalues
    assert result.projector is projector
    assert result.upper_boundary_gap == 2.0
    with pytest.raises(FrozenInstanceError):
        result.dimension = 3  # type: ignore[misc]


def test_lower_edge_result_requires_an_upper_boundary_gap() -> None:
    with pytest.raises(ValueError):
        HermitianSpectralSubspaceResult(
            dimension=2,
            start_index=0,
            stop_index=1,
            tolerance=_tolerance(),
            eigenvalues=(1.0, 3.0),
            projector=(
                (1.0 + 0.0j, 0.0 + 0.0j),
                (0.0 + 0.0j, 0.0 + 0.0j),
            ),
            lower_boundary_gap=None,
            upper_boundary_gap=None,
            diagnostics=_diagnostics(),
            backend_identifier="example.eigh",
            backend_version="1.2.3",
        )


def test_upper_edge_result_requires_only_a_lower_boundary_gap() -> None:
    result = HermitianSpectralSubspaceResult(
        dimension=2,
        start_index=1,
        stop_index=2,
        tolerance=_tolerance(),
        eigenvalues=(1.0, 3.0),
        projector=((0.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j)),
        lower_boundary_gap=2.0,
        upper_boundary_gap=None,
        diagnostics=_diagnostics(),
        backend_identifier="example.eigh",
        backend_version="1.2.3",
    )

    assert result.lower_boundary_gap == 2.0
    assert result.upper_boundary_gap is None
