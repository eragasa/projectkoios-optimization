"""Unit tests for Hermitian spectral-subspace results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

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


def _result(**changes: object) -> HermitianSpectralSubspaceResult:
    values: dict[str, object] = {
        "dimension": 2,
        "start_index": 0,
        "stop_index": 1,
        "tolerance": _tolerance(),
        "eigenvalues": (1.0, 3.0),
        "projector": ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 0.0 + 0.0j)),
        "lower_boundary_gap": None,
        "upper_boundary_gap": 2.0,
        "diagnostics": _diagnostics(),
        "backend_identifier": "example.eigh",
        "backend_version": "1.2.3",
    }
    values.update(changes)
    return HermitianSpectralSubspaceResult(**cast(Any, values))


def test_result_retains_correlated_dependency_free_values_and_is_frozen() -> None:
    result = _result()

    assert result.dimension == 2
    assert result.eigenvalues == (1.0, 3.0)
    assert result.upper_boundary_gap == 2.0
    assert result.backend_identifier == "example.eigh"
    with pytest.raises(FrozenInstanceError):
        result.dimension = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    (
        ({"dimension": True}, TypeError, "dimension"),
        ({"stop_index": 3}, ValueError, "dimension"),
        ({"eigenvalues": (3.0, 1.0)}, ValueError, "ascending"),
        ({"eigenvalues": (1.0,)}, ValueError, "dimension"),
        ({"eigenvalues": (1.0, float("nan"))}, ValueError, "finite"),
        ({"projector": ((1.0 + 0.0j,),)}, ValueError, "dimension"),
        ({"lower_boundary_gap": 0.0}, ValueError, "lower edge"),
        ({"upper_boundary_gap": None}, ValueError, "required"),
        ({"diagnostics": object()}, TypeError, "Diagnostics"),
        ({"backend_identifier": ""}, ValueError, "nonempty"),
        ({"backend_version": 1}, TypeError, "string"),
    ),
)
def test_result_rejects_malformed_local_structure(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _result(**changes)


def test_result_requires_present_lower_and_absent_upper_gap_for_upper_edge() -> None:
    result = _result(
        start_index=1,
        stop_index=2,
        lower_boundary_gap=2.0,
        upper_boundary_gap=None,
        projector=((0.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j)),
    )

    assert result.lower_boundary_gap == 2.0
    assert result.upper_boundary_gap is None
