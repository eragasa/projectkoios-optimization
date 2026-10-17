"""Unit tests for sparse Hermitian partial-spectrum results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    HermitianSpectrumMatrixDiagnostics,
)
from projectkoios.numerics.linalg.sparse_hermitian_spectrum import (
    SparseHermitianPartialSpectrumResult,
    SparseHermitianSpectrumSelection,
)

pytestmark = pytest.mark.unit


def _diagnostics() -> HermitianSpectrumMatrixDiagnostics:
    return HermitianSpectrumMatrixDiagnostics(0.0, 0.0, 2.0, 0.0)


def test_result_retains_canonical_storage_and_is_frozen() -> None:
    eigenvalues = (1.0, 2.0)
    result = SparseHermitianPartialSpectrumResult(
        dimension=3,
        eigenvalue_count=2,
        selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
        eigenvalues=eigenvalues,
        residual_maximum_absolute_norm=0.0,
        matrix_diagnostics=_diagnostics(),
        backend_identifier="example.eigsh",
        backend_version="1.2.3",
    )

    assert result.eigenvalues is eigenvalues
    assert result.residual_maximum_absolute_norm == 0.0
    with pytest.raises(FrozenInstanceError):
        result.dimension = 4  # type: ignore[misc]


def test_constructor_requires_partial_spectrum_smaller_than_dimension() -> None:
    with pytest.raises(ValueError):
        SparseHermitianPartialSpectrumResult(
            dimension=3,
            eigenvalue_count=3,
            selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            eigenvalues=(1.0, 2.0, 3.0),
            residual_maximum_absolute_norm=0.0,
            matrix_diagnostics=_diagnostics(),
            backend_identifier="example.eigsh",
            backend_version="1.2.3",
        )


def test_constructor_requires_ascending_eigenvalues() -> None:
    with pytest.raises(ValueError):
        SparseHermitianPartialSpectrumResult(
            dimension=3,
            eigenvalue_count=2,
            selection=SparseHermitianSpectrumSelection.SMALLEST_ALGEBRAIC,
            eigenvalues=(2.0, 1.0),
            residual_maximum_absolute_norm=0.0,
            matrix_diagnostics=_diagnostics(),
            backend_identifier="example.eigsh",
            backend_version="1.2.3",
        )
