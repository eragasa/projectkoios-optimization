"""Integration tests for dense NumPy spectral-norm evaluation.

The fixtures use analytically derived largest singular values and explicit
representation boundaries. Backend substitution is limited to otherwise nondeterministic
failure paths:
NumPy SVD nonconvergence and nonfinite backend output.
"""

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
        # This nonnormal matrix has only zero eigenvalues but spectral norm two, proving
        # that the implementation evaluates singular values rather than eigenvalue size.
        (
            np.asarray(((0.0, 2.0), (0.0, 0.0)), dtype=np.float64),
            2.0,
        ),
        # The two nonzero singular values are the magnitudes five and two.
        (
            np.asarray(
                ((3.0 + 4.0j, 0.0j, 0.0j), (0.0j, 2.0j, 0.0j)),
                dtype=np.complex128,
            ),
            5.0,
        ),
        # A rectangular rank-one outer product has one singular value equal to five.
        (
            np.asarray(((3.0, 0.0, 0.0), (4.0, 0.0, 0.0)), dtype=np.float64),
            5.0,
        ),
        # One-row and one-column matrices reduce to their Euclidean vector norms.
        (np.asarray(((3.0, 4.0),), dtype=np.float64), 5.0),
        (np.asarray(((3.0j,), (4.0 + 0.0j,)), dtype=np.complex128), 5.0),
    ],
)
def test_evaluation_returns_largest_singular_value(
    matrix: DenseMatrix,
    expected: float,
) -> None:
    """Verify analytic real and complex largest singular values.

    Parameters
    ----------
    matrix
        Exact dense matrix for one analytic case.
    expected
        Independently derived largest singular value.
    """
    evaluation = NumpySpectralNorm(matrix)

    result = evaluation.result()

    # Scale the binary64 roundoff allowance by problem dimension and analytic norm.
    epsilon = math.ulp(1.0)
    error_bound = 64.0 * max(matrix.shape) * epsilon * max(1.0, expected)

    assert evaluation.matrix_norm is MatrixNormEnum.spectral
    assert abs(result - expected) <= error_bound
    assert type(result) is float


def test_zero_matrix_returns_exact_zero() -> None:
    """Verify an all-zero rectangular matrix returns exact built-in zero.

    Notes
    -----
    Every singular value of the zero matrix is exactly zero. This assertion is exact and
    uses no roundoff allowance.
    """
    matrix: DenseMatrix = np.zeros((2, 3), dtype=np.complex128)

    result = NumpySpectralNorm(matrix).result()

    assert result == 0.0
    assert type(result) is float


def test_accepts_fortran_layout_noncontiguous_view_and_read_only_storage() -> None:
    """Verify storage layout, stride, and writeability do not change the norm.

    Notes
    -----
    The noncontiguous view selects the first and third source columns without copying.
    The read-only case verifies that evaluation needs no write access.
    """
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
    """Verify evaluation preserves writable complex caller data and ownership state.

    Notes
    -----
    The copied oracle preserves every source value. The writeability assertion verifies
    that evaluation does not change the caller's storage policy.
    """
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
    """Verify dimensionality and both empty-axis boundaries fail distinctly.

    Parameters
    ----------
    matrix
        Array with invalid dimensionality or an empty matrix axis.
    message
        Exact public validation message expected for that malformed shape.
    """
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
    """Verify lower-precision and integer arrays are not coerced to binary64.

    Parameters
    ----------
    matrix
        Two-dimensional ndarray with an unsupported exact dtype.
    """
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
    """Verify nonfinite real or complex source entries fail before SVD.

    Parameters
    ----------
    matrix
        Exact supported ndarray containing a nonfinite component.
    """
    with pytest.raises(ValueError, match="^matrix entries must be finite$"):
        NumpySpectralNorm(matrix).result()


def test_finite_input_with_unrepresentable_norm_fails_closed() -> None:
    """Verify a finite matrix whose norm exceeds binary64 fails as arithmetic.

    Notes
    -----
    Source finiteness succeeds because both complex components fit binary64. The
    singular value does not fit binary64, so the backend-output boundary raises
    ``FloatingPointError``.
    """
    # Both components are finite, but their magnitude exceeds the largest binary64
    # value, so successful evaluation cannot return a finite scalar.
    matrix: DenseMatrix = np.asarray(
        ((complex(1.3e308, 1.3e308),),), dtype=np.complex128
    )

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()


def test_translates_numpy_svd_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify exact NumPy SVD nonconvergence becomes the public runtime failure.

    Parameters
    ----------
    monkeypatch
        Scoped replacement used because no finite fixture deterministically forces
        LAPACK SVD nonconvergence across supported platforms.
    """
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonconvergent(_matrix: DenseMatrix, /) -> np.ndarray:
        """Raise the exact backend exception translated by the implementation.

        Parameters
        ----------
        _matrix
            Valid dense matrix forwarded to the substituted backend.

        Raises
        ------
        numpy.linalg.LinAlgError
            Unconditionally, to represent dense SVD nonconvergence.
        """
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(np.linalg, "svdvals", nonconvergent)

    with pytest.raises(RuntimeError, match="did not converge"):
        NumpySpectralNorm(matrix).result()


def test_rejects_nonfinite_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify nonfinite complete-SVD output fails at the backend trust boundary.

    Parameters
    ----------
    monkeypatch
        Scoped replacement used to represent malformed numerical backend output without
        manufacturing invalid public matrix input.
    """
    matrix: DenseMatrix = np.asarray(((3.0, 4.0),), dtype=np.float64)

    def nonfinite(_matrix: DenseMatrix, /) -> np.ndarray:
        """Return a nonfinite singular-value array from the backend boundary.

        Parameters
        ----------
        _matrix
            Valid dense matrix forwarded to the substituted backend.

        Returns
        -------
        numpy.ndarray
            One-element float64 array containing NaN.
        """
        return np.asarray((float("nan"),), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "svdvals", nonfinite)

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        NumpySpectralNorm(matrix).result()
