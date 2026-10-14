"""Unit tests for Hermitian spectral-subspace diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDiagnostics,
)

pytestmark = pytest.mark.unit


def _diagnostics() -> HermitianSpectralSubspaceDiagnostics:
    return HermitianSpectralSubspaceDiagnostics(*(float(index) for index in range(18)))


def test_diagnostics_freezes_all_eighteen_exact_fields() -> None:
    diagnostics = _diagnostics()

    assert len(fields(diagnostics)) == 18
    assert diagnostics.input_hermiticity_maximum_absolute_defect == 0.0
    assert diagnostics.representative_adjustment_maximum_absolute_defect == 1.0
    assert diagnostics.eigenvector_orthonormality_maximum_absolute_defect == 2.0
    assert diagnostics.eigensystem_residual_maximum_absolute_defect == 3.0
    assert diagnostics.representative_reconstruction_maximum_absolute_defect == 4.0
    assert diagnostics.applied_spectral_validation_cutoff == 17.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.spectral_diameter = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (True, TypeError),
        (1, TypeError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ),
)
def test_diagnostics_rejects_invalid_field_values(
    value: object,
    error: type[Exception],
) -> None:
    values: list[object] = [0.0] * 18
    values[7] = value

    with pytest.raises(error, match="projector_trace_absolute_defect"):
        HermitianSpectralSubspaceDiagnostics(
            *cast(
                tuple[
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                ],
                tuple(values),
            )
        )
