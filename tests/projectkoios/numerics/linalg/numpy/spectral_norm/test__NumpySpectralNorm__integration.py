"""Integration tests for dense NumPy spectral norms."""

from __future__ import annotations

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

    assert evaluation.matrix_norm is MatrixNormEnum.spectral
    assert result == expected
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

    assert NumpySpectralNorm(fortran_matrix).result() == 4.0
    assert NumpySpectralNorm(noncontiguous_view).result() == 4.0
    assert NumpySpectralNorm(read_only).result() == 5.0
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
    "matrix",
    [
        np.asarray((1.0, 2.0), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.complex128),
    ],
)
def test_rejects_non_matrix_or_empty_shape(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="two-dimensional|nonempty"):
        NumpySpectralNorm(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((1.0,),), dtype=np.float32),
        np.asarray(((1,),), dtype=np.int64),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: DenseMatrix) -> None:
    with pytest.raises(TypeError, match="dtype"):
        NumpySpectralNorm(matrix).result()


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(((float("nan"),),), dtype=np.float64),
        np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128),
    ],
)
def test_rejects_nonfinite_entries(matrix: DenseMatrix) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumpySpectralNorm(matrix).result()


def test_finite_input_with_unrepresentable_norm_fails_closed() -> None:
    matrix: DenseMatrix = np.asarray(
        ((complex(1.3e308, 1.3e308),),), dtype=np.complex128
    )

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()


def test_translates_only_numpy_svd_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonconvergent(_matrix: DenseMatrix, /) -> np.ndarray:
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(np.linalg, "svdvals", nonconvergent)

    with pytest.raises(RuntimeError, match="did not converge") as captured:
        NumpySpectralNorm(matrix).result()

    assert isinstance(captured.value.__cause__, np.linalg.LinAlgError)


def test_rejects_nonfinite_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonfinite(_matrix: DenseMatrix, /) -> np.ndarray:
        return np.asarray((float("nan"),), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "svdvals", nonfinite)

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()
