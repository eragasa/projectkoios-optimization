"""Integration tests for the SciPy batched Hermitian-spectrum backend."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import scipy
import scipy.linalg

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrices,
)
from projectkoios.numerics.linalg.scipy.hermitian_spectrum import (
    ScipyBatchedHermitianSpectrumSolver,
)
from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array

pytestmark = pytest.mark.integration


HERMITIAN_MATRIX_BATCH: HermitianSpectrumMatrices = (
    (
        (2.0 + 0.0j, 1.0 - 1.0j),
        (1.0 + 1.0j, 3.0 + 0.0j),
    ),
    (
        (4.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 5.0 + 0.0j),
    ),
)


def test_backend_uses_one_full_spectrum_batched_scipy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scipy.linalg.eigvalsh
    observed: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def recording_eigvalsh(
        matrices: object,
        **kwargs: object,
    ) -> Float64Array:
        array = cast(Complex128Array, np.asarray(matrices, dtype=np.complex128))
        observed.append((tuple(int(value) for value in array.shape), dict(kwargs)))
        return np.asarray(original(array, **cast(Any, kwargs)), dtype=np.float64)

    monkeypatch.setattr(scipy.linalg, "eigvalsh", recording_eigvalsh)
    result = ScipyBatchedHermitianSpectrumSolver().solve(
        BatchedHermitianSpectrumDefinition(
            HERMITIAN_MATRIX_BATCH,
            BatchedHermitianSpectrumTolerance(0.0, 0.0),
        )
    )

    assert observed == [
        (
            (2, 2, 2),
            {
                "lower": True,
                "overwrite_a": False,
                "check_finite": False,
                "driver": "evd",
            },
        )
    ]
    np.testing.assert_allclose(
        np.asarray(result.eigenvalues),
        np.linalg.eigvalsh(np.asarray(HERMITIAN_MATRIX_BATCH, dtype=np.complex128)),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.backend_identifier == "scipy.linalg.eigvalsh"
    assert result.backend_version == scipy.__version__


def test_empty_batch_does_not_call_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_eigvalsh(matrices: object, **kwargs: object) -> Float64Array:
        del matrices, kwargs
        raise AssertionError("empty batch must not invoke SciPy")

    monkeypatch.setattr(scipy.linalg, "eigvalsh", forbidden_eigvalsh)
    result = ScipyBatchedHermitianSpectrumSolver().solve(
        BatchedHermitianSpectrumDefinition.empty(
            4,
            BatchedHermitianSpectrumTolerance(0.0, 0.0),
        )
    )

    assert result.batch_size == 0
    assert result.dimension == 4
    assert result.diagnostics is None


def test_backend_translates_scipy_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonconvergent(matrices: object, **kwargs: object) -> Float64Array:
        del matrices, kwargs
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(scipy.linalg, "eigvalsh", nonconvergent)

    with pytest.raises(RuntimeError):
        ScipyBatchedHermitianSpectrumSolver().solve(
            BatchedHermitianSpectrumDefinition(
                HERMITIAN_MATRIX_BATCH,
                BatchedHermitianSpectrumTolerance(0.0, 0.0),
            )
        )


def test_backend_rejects_nonfinite_scipy_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite(matrices: object, **kwargs: object) -> Float64Array:
        del matrices, kwargs
        return np.asarray([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64)

    monkeypatch.setattr(scipy.linalg, "eigvalsh", nonfinite)

    with pytest.raises(FloatingPointError):
        ScipyBatchedHermitianSpectrumSolver().solve(
            BatchedHermitianSpectrumDefinition(
                HERMITIAN_MATRIX_BATCH,
                BatchedHermitianSpectrumTolerance(0.0, 0.0),
            )
        )
