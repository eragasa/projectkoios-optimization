"""Unit tests for batched Hermitian-spectrum result invariants."""

from __future__ import annotations

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDiagnostics,
    BatchedHermitianSpectrumResult,
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrixDiagnostics,
)

pytestmark = pytest.mark.unit


def _tolerance() -> BatchedHermitianSpectrumTolerance:
    return BatchedHermitianSpectrumTolerance(1.0e-12, 1.0e-12)


def _records() -> tuple[HermitianSpectrumMatrixDiagnostics, ...]:
    return (
        HermitianSpectrumMatrixDiagnostics(1.0e-13, 5.0e-14, 2.0, 3.0e-12),
        HermitianSpectrumMatrixDiagnostics(2.0e-13, 1.0e-13, 4.0, 5.0e-12),
    )


def _aggregate() -> BatchedHermitianSpectrumDiagnostics:
    return BatchedHermitianSpectrumDiagnostics(2.0e-13, 1.0e-13, 4.0, 5.0e-12)


def test_result_retains_canonical_spectra_and_derives_batch_size() -> None:
    eigenvalues = ((1.0, 2.0), (3.0, 4.0))
    result = BatchedHermitianSpectrumResult(
        dimension=2,
        tolerance=_tolerance(),
        eigenvalues=eigenvalues,
        matrix_diagnostics=_records(),
        diagnostics=_aggregate(),
        backend_identifier="test.eigvalsh",
        backend_version="1",
    )

    assert result.batch_size == 2
    assert result.dimension == 2
    assert result.eigenvalues is eigenvalues
    assert result.diagnostics == _aggregate()


def test_empty_result_has_absent_aggregate_diagnostics() -> None:
    result = BatchedHermitianSpectrumResult(
        dimension=3,
        tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
        eigenvalues=(),
        matrix_diagnostics=(),
        diagnostics=None,
        backend_identifier="test.eigvalsh",
        backend_version="1",
    )

    assert result.batch_size == 0
    assert result.dimension == 3
    assert result.diagnostics is None


def test_empty_result_rejects_fabricated_zero_maxima() -> None:
    with pytest.raises(ValueError, match="absent for an empty batch"):
        BatchedHermitianSpectrumResult(
            dimension=3,
            tolerance=BatchedHermitianSpectrumTolerance(0.0, 0.0),
            eigenvalues=(),
            matrix_diagnostics=(),
            diagnostics=BatchedHermitianSpectrumDiagnostics(0.0, 0.0, 0.0, 0.0),
            backend_identifier="test.eigvalsh",
            backend_version="1",
        )
