"""Unit tests for batched Hermitian-spectrum result invariants."""

from __future__ import annotations

from typing import Any, cast

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


def _result(**overrides: object) -> BatchedHermitianSpectrumResult:
    values: dict[str, object] = {
        "dimension": 2,
        "tolerance": _tolerance(),
        "eigenvalues": [[1.0, 2.0], [3.0, 4.0]],
        "matrix_diagnostics": _records(),
        "diagnostics": _aggregate(),
        "backend_identifier": "test.eigvalsh",
        "backend_version": "1",
    }
    values.update(overrides)
    return BatchedHermitianSpectrumResult(**cast(Any, values))


def test_result_normalizes_spectra_and_derives_batch_size() -> None:
    result = _result()

    assert result.batch_size == 2
    assert result.dimension == 2
    assert result.eigenvalues == ((1.0, 2.0), (3.0, 4.0))
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


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({"dimension": 0}, ValueError, "dimension"),
        ({"eigenvalues": ((1.0,), (2.0,))}, ValueError, "dimension"),
        ({"eigenvalues": ((2.0, 1.0), (3.0, 4.0))}, ValueError, "ascending"),
        ({"eigenvalues": ((1.0, float("nan")), (3.0, 4.0))}, ValueError, "finite"),
        ({"matrix_diagnostics": (_records()[0],)}, ValueError, "batch_size"),
        ({"matrix_diagnostics": (object(), object())}, TypeError, "entries"),
        (
            {
                "matrix_diagnostics": (
                    HermitianSpectrumMatrixDiagnostics(
                        1.0e-13,
                        5.0e-14,
                        2.0,
                        4.0e-12,
                    ),
                    _records()[1],
                )
            },
            ValueError,
            "cutoff is inconsistent",
        ),
        (
            {
                "matrix_diagnostics": (
                    HermitianSpectrumMatrixDiagnostics(
                        4.0e-12,
                        5.0e-14,
                        2.0,
                        3.0e-12,
                    ),
                    _records()[1],
                )
            },
            ValueError,
            "defect exceeds",
        ),
        (
            {
                "diagnostics": BatchedHermitianSpectrumDiagnostics(
                    2.0e-13,
                    1.0e-13,
                    4.0,
                    6.0e-12,
                )
            },
            ValueError,
            "aggregate maxima otherwise",
        ),
        ({"backend_identifier": ""}, ValueError, "backend_identifier"),
        ({"backend_identifier": 1}, TypeError, "backend_identifier"),
        ({"backend_version": ""}, ValueError, "backend_version"),
        ({"backend_version": 1}, TypeError, "backend_version"),
    ],
)
def test_result_rejects_inconsistent_fields(
    overrides: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _result(**overrides)
