"""Integration tests for dense NumPy spectral norms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.numpy.spectral_norm import NumpySpectralNorm
from projectkoios.numerics.numpy.typing import DenseMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (
            np.asarray(((0.0, 2.0), (0.0, 0.0)), dtype=np.float64),
            2.0,
        ),
        (
            np.asarray(
                ((3.0 + 4.0j, 0.0j, 0.0j), (0.0j, 2.0j, 0.0j)),
                dtype=np.complex128,
            ),
            5.0,
        ),
        (
            np.asarray(((3.0, 0.0, 0.0), (4.0, 0.0, 0.0)), dtype=np.float64),
            5.0,
        ),
        (np.asarray(((3.0, 4.0),), dtype=np.float64), 5.0),
        (np.asarray(((3.0j,), (4.0 + 0.0j,)), dtype=np.complex128), 5.0),
        (np.zeros((2, 3), dtype=np.float64), 0.0),
    ],
)
def test_evaluation_returns_largest_singular_value(
    matrix: DenseMatrix,
    expected: float,
) -> None:
    evaluation = NumpySpectralNorm(matrix)

    result = evaluation.result()

    # Scale the binary64 roundoff allowance by problem dimension and analytic norm.
    epsilon = math.ulp(1.0)
    error_bound = 64.0 * max(matrix.shape) * epsilon * max(1.0, expected)

    assert evaluation.matrix_norm is MatrixNormEnum.spectral
    assert abs(result - expected) <= error_bound
    assert type(result) is float


def test_accepts_fortran_layout_noncontiguous_view_and_read_only_storage() -> None:
    fortran_matrix: DenseMatrix = np.asfortranarray(
        np.asarray(((3.0, 0.0), (0.0, 4.0)), dtype=np.float64)
    )
    source = np.asarray(
        ((3.0, 99.0, 0.0, 99.0), (0.0, 99.0, 4.0, 99.0)),
        dtype=np.float64,
    )
    noncontiguous_view: DenseMatrix = source[:, ::2]
    read_only: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)
    read_only.setflags(write=False)

    epsilon = math.ulp(1.0)
    for matrix, expected in (
        (fortran_matrix, 4.0),
        (noncontiguous_view, 4.0),
        (read_only, 5.0),
    ):
        result = NumpySpectralNorm(matrix).result()
        error_bound = 64.0 * max(matrix.shape) * epsilon * expected
        assert abs(result - expected) <= error_bound
    assert not read_only.flags.writeable


def test_evaluation_does_not_mutate_caller_storage() -> None:
    matrix: DenseMatrix = np.asarray(
        ((1.0 + 2.0j, 3.0 + 4.0j), (5.0 - 1.0j, 2.0j)),
        dtype=np.complex128,
    )
    before = matrix.copy()

    NumpySpectralNorm(matrix).result()

    assert np.array_equal(matrix, before)
    assert matrix.flags.writeable


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            np.asarray((1.0, 2.0), dtype=np.float64),
            "^matrix must be two-dimensional$",
        ),
        (np.empty((0, 2), dtype=np.float64), "^matrix must be nonempty$"),
        (np.empty((2, 0), dtype=np.complex128), "^matrix must be nonempty$"),
    ],
)
def test_rejects_non_matrix_or_empty_shape(
    matrix: DenseMatrix,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NumpySpectralNorm(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((1.0,),), dtype=np.float32),
        np.asarray(((1,),), dtype=np.int64),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: DenseMatrix) -> None:
    with pytest.raises(
        TypeError,
        match="^matrix dtype must be exactly float64 or complex128$",
    ):
        NumpySpectralNorm(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((float("nan"),),), dtype=np.float64),
        np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128),
    ],
)
def test_rejects_nonfinite_entries(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="^matrix entries must be finite$"):
        NumpySpectralNorm(matrix).result()


def test_finite_input_with_unrepresentable_norm_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(
        ((complex(1.3e308, 1.3e308),),), dtype=np.complex128
    )

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()


def test_translates_numpy_svd_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonconvergent(_matrix: DenseMatrix, /) -> np.ndarray:
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(np.linalg, "svdvals", nonconvergent)

    with pytest.raises(RuntimeError, match="did not converge"):
        NumpySpectralNorm(matrix).result()


def test_rejects_nonfinite_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonfinite(_matrix: DenseMatrix, /) -> np.ndarray:
        return np.asarray((float("nan"),), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "svdvals", nonfinite)

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()
