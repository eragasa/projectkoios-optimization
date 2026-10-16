"""Integration tests for the NumPy batched Hermitian-spectrum backend."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
    NumpyBatchedHermitianSpectrumSolver,
)
from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array

pytestmark = pytest.mark.integration


def _array_batch() -> Complex128Array:
    return np.asarray(
        [
            [[2.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]],
            [[4.0, 0.0], [0.0, 5.0]],
        ],
        dtype=np.complex128,
    )


def test_backend_accepts_numpy_arrays_and_uses_one_batched_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.linalg.eigvalsh
    observed_shapes: list[tuple[int, ...]] = []

    def recording_eigvalsh(
        matrices: object,
        UPLO: Literal["L", "U"] = "L",
    ) -> Float64Array:
        array = cast(Complex128Array, np.asarray(matrices, dtype=np.complex128))
        observed_shapes.append(tuple(int(value) for value in array.shape))
        return np.asarray(original(array, UPLO=UPLO), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "eigvalsh", recording_eigvalsh)
    result = NumpyBatchedHermitianSpectrumSolver().solve(
        BatchedHermitianSpectrumDefinition(
            _array_batch(),
            BatchedHermitianSpectrumTolerance(np.float64(0.0), np.float64(0.0)),
        )
    )

    assert observed_shapes == [(2, 2, 2)]
    assert result.batch_size == 2
    np.testing.assert_allclose(
        np.asarray(result.eigenvalues),
        original(_array_batch()),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_empty_batch_does_not_call_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_eigvalsh(
        matrices: object,
        UPLO: Literal["L", "U"] = "L",
    ) -> Float64Array:
        del matrices, UPLO
        raise AssertionError("empty batch must not invoke NumPy")

    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden_eigvalsh)
    result = NumpyBatchedHermitianSpectrumSolver().solve(
        BatchedHermitianSpectrumDefinition.empty(
            np.int64(4),
            BatchedHermitianSpectrumTolerance(0.0, 0.0),
        )
    )

    assert result.batch_size == 0
    assert result.dimension == 4
    assert result.diagnostics is None


def test_backend_translates_only_eigensolver_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonconvergent(
        matrices: object,
        UPLO: Literal["L", "U"] = "L",
    ) -> Float64Array:
        del matrices, UPLO
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(np.linalg, "eigvalsh", nonconvergent)

    with pytest.raises(RuntimeError, match="did not converge"):
        NumpyBatchedHermitianSpectrumSolver().solve(
            BatchedHermitianSpectrumDefinition(
                _array_batch(),
                BatchedHermitianSpectrumTolerance(0.0, 0.0),
            )
        )


def test_backend_rejects_nonfinite_eigenvalues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite(
        matrices: object,
        UPLO: Literal["L", "U"] = "L",
    ) -> Float64Array:
        del matrices, UPLO
        return np.asarray([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64)

    monkeypatch.setattr(np.linalg, "eigvalsh", nonfinite)

    with pytest.raises(FloatingPointError, match="nonfinite eigenvalues"):
        NumpyBatchedHermitianSpectrumSolver().solve(
            BatchedHermitianSpectrumDefinition(
                _array_batch(),
                BatchedHermitianSpectrumTolerance(0.0, 0.0),
            )
        )
