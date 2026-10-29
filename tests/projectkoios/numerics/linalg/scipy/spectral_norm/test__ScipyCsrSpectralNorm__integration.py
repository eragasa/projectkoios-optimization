"""Integration tests for deterministic SciPy CSR spectral-norm evaluation.

Analytic fixtures cover the scale-safe direct branch and approximate ARPACK branch.
Backend substitution is limited to branch observation and numerical trust-boundary cases
that portable finite matrices cannot force deterministically.
"""

from __future__ import annotations

import math
import sys
from typing import Never

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.scipy.spectral_norm import (
    ScipyCsrSpectralNorm,
    ScipyCsrSpectralNormControls,
)
from projectkoios.numerics.numpy.typing import (
    Complex128Array,
    DenseMatrix,
    Float64Array,
)
from projectkoios.numerics.scipy.typing import CsrMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (sp.csr_array(np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64)), 5.0),
        (
            sp.csr_matrix(
                np.asarray(((3.0j,), (4.0 + 0.0j,), (0.0j,)), dtype=np.complex128)
            ),
            5.0,
        ),
        (
            sp.csr_array(
                np.asarray(((3.0, 0.0, 0.0), (0.0, 4.0, 0.0)), dtype=np.float64)
            ),
            4.0,
        ),
        (
            sp.csr_matrix(
                np.asarray(
                    ((3.0 + 4.0j, 0.0j), (0.0j, 2.0j), (0.0j, 0.0j)),
                    dtype=np.complex128,
                )
            ),
            5.0,
        ),
        (sp.csr_array(np.diag((3.0, 4.0, 5.0))), 5.0),
        (
            sp.csr_matrix(
                np.asarray(
                    (
                        (3.0 + 4.0j, 0.0j, 0.0j),
                        (0.0j, 4.0j, 0.0j),
                        (0.0j, 0.0j, 2.0 + 0.0j),
                        (0.0j, 0.0j, 0.0j),
                    ),
                    dtype=np.complex128,
                )
            ),
            5.0,
        ),
    ],
)
def test_evaluation_returns_analytic_largest_singular_value(
    matrix: CsrMatrix,
    expected: float,
) -> None:
    """Verify direct and iterative real and complex analytic cases.

    Parameters
    ----------
    matrix
        Exact CSR matrix or array for one analytic case.
    expected
        Independently derived largest singular value.
    """
    controls = ScipyCsrSpectralNormControls(1.0e-10, 1_000, None, 17)
    evaluation = ScipyCsrSpectralNorm(matrix, controls)

    result = evaluation.result()

    numerical_bound = max(
        controls.tolerance,
        64.0 * math.ulp(1.0) * max(matrix.shape),
    ) * max(1.0, expected)
    assert evaluation.matrix_norm is MatrixNormEnum.spectral
    assert abs(result - expected) <= numerical_bound
    assert type(result) is float


def test_zero_iterative_shape_returns_exact_zero_without_backend_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a large-branch all-zero matrix short-circuits before ARPACK.

    Parameters
    ----------
    monkeypatch
        Scoped substitution proving the zero branch does not invoke ``svds``.
    """
    matrix: CsrMatrix = sp.csr_array((3, 5), dtype=np.complex128)
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    def forbidden_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Fail if the exact-zero branch reaches SciPy iteration.

        Parameters
        ----------
        _matrix
            Canonical zero CSR value that must short-circuit.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface that must remain unreachable.

        Raises
        ------
        AssertionError
            Unconditionally, because zero storage must not reach ARPACK.
        """
        raise AssertionError("svds must not run for an all-zero matrix")

    monkeypatch.setattr(scipy.sparse.linalg, "svds", forbidden_svds)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert result == 0.0
    assert type(result) is float


def test_private_canonicalization_preserves_caller_duplicate_storage() -> None:
    """Verify represented duplicates are consolidated only on a private copy."""
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((1.0, 2.0, 4.0, -4.0, 5.0), dtype=np.float64),
            np.asarray((0, 0, 1, 1, 2), dtype=np.int64),
            np.asarray((0, 2, 4, 5), dtype=np.int64),
        ),
        shape=(3, 3),
    )
    data_before = matrix.data.copy()
    indices_before = matrix.indices.copy()
    indptr_before = matrix.indptr.copy()
    canonical_before = matrix.has_canonical_format
    controls = ScipyCsrSpectralNormControls(1.0e-10, 500, None, 5)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert abs(result - 5.0) <= 5.0e-10
    assert np.array_equal(matrix.data, data_before)
    assert np.array_equal(matrix.indices, indices_before)
    assert np.array_equal(matrix.indptr, indptr_before)
    assert matrix.has_canonical_format is canonical_before


def test_scale_safe_direct_branch_handles_extreme_representable_range() -> None:
    """Verify small-Gram scaling avoids raw squaring overflow and underflow."""
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(((1.0e308, 1.0e-308),), dtype=np.float64)
    )
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert result == 1.0e308


def test_iterative_branch_never_requests_sparse_matrix_densification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ARPACK evaluation does not call CSR dense-conversion methods.

    Parameters
    ----------
    monkeypatch
        Scoped guards on both supported CSR runtime classes.
    """
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    def forbidden_dense_conversion(
        _matrix: CsrMatrix,
        order: str | None = None,
        out: DenseMatrix | None = None,
    ) -> Never:
        """Fail if owner or backend requests dense matrix materialization.

        Parameters
        ----------
        _matrix
            CSR matrix or array receiving the forbidden conversion request.
        order, out
            Standard SciPy dense-conversion arguments, accepted only to intercept the
            complete public method boundary.

        Raises
        ------
        AssertionError
            Unconditionally, because the sparse path must not densify.
        """
        raise AssertionError("sparse spectral evaluation must not densify")

    monkeypatch.setattr(sp.csr_array, "toarray", forbidden_dense_conversion)
    monkeypatch.setattr(sp.csr_array, "todense", forbidden_dense_conversion)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden_dense_conversion)
    monkeypatch.setattr(sp.csr_matrix, "todense", forbidden_dense_conversion)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert abs(result - 5.0) <= 5.0e-9


def test_read_only_caller_data_remains_read_only_and_unchanged() -> None:
    """Verify private canonicalization requires no write access to caller data."""
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    before = matrix.data.copy()
    matrix.data.setflags(write=False)
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert abs(result - 5.0) <= 5.0e-9
    assert np.array_equal(matrix.data, before)
    assert not matrix.data.flags.writeable


def test_large_implicit_dimension_is_not_densified() -> None:
    """Verify the direct sparse branch handles a billion-column representation."""
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((3.0, 4.0), dtype=np.float64),
            np.asarray((0, 999_999_999), dtype=np.int64),
            np.asarray((0, 2), dtype=np.int64),
        ),
        shape=(1, 1_000_000_000),
    )
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    result = ScipyCsrSpectralNorm(matrix, controls).result()

    assert result == 5.0


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            sp.csr_array(np.asarray((1.0, 2.0), dtype=np.float64)),
            "two-dimensional",
        ),
        (sp.csr_array((0, 2), dtype=np.float64), "nonempty"),
        (sp.csr_matrix((2, 0), dtype=np.complex128), "nonempty"),
    ],
)
def test_rejects_non_matrix_or_empty_shape(
    matrix: CsrMatrix,
    message: str,
) -> None:
    """Verify malformed dimensionality and both empty-axis boundaries.

    Parameters
    ----------
    matrix
        CSR representation with invalid dimensionality or emptiness.
    message
        Stable error-message fragment for the represented defect.
    """
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    with pytest.raises(ValueError, match=message):
        ScipyCsrSpectralNorm(matrix, controls).result()


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray(((1.0,),), dtype=np.float32)),
        sp.csr_matrix(np.asarray(((1,),), dtype=np.int64)),
    ],
)
def test_rejects_unsupported_exact_dtype(matrix: CsrMatrix) -> None:
    """Verify unsupported storage dtypes are never coerced.

    Parameters
    ----------
    matrix
        CSR value with unsupported exact storage dtype.
    """
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    with pytest.raises(TypeError, match="dtype"):
        ScipyCsrSpectralNorm(matrix, controls).result()


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_array(np.asarray(((float("nan"),),), dtype=np.float64)),
        sp.csr_matrix(
            np.asarray(((complex(0.0, float("inf")),),), dtype=np.complex128)
        ),
    ],
)
def test_rejects_nonfinite_source_entries(matrix: CsrMatrix) -> None:
    """Verify source nonfiniteness fails before canonical arithmetic.

    Parameters
    ----------
    matrix
        Supported CSR value containing a nonfinite stored component.
    """
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    with pytest.raises(ValueError, match="matrix entries must be finite"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_duplicate_consolidation_overflow_fails_closed() -> None:
    """Verify finite duplicate entries cannot consolidate to infinity silently."""
    matrix: CsrMatrix = sp.csr_array(
        (
            np.asarray((1.0e308, 1.0e308), dtype=np.float64),
            np.asarray((0, 0), dtype=np.int64),
            np.asarray((0, 2), dtype=np.int64),
        ),
        shape=(1, 1),
    )
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    with pytest.raises(FloatingPointError, match="consolidation became nonfinite"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_direct_unrepresentable_norm_fails_closed() -> None:
    """Verify finite entries whose norm exceeds binary64 fail as arithmetic."""
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(((1.3e308, 1.3e308),), dtype=np.float64)
    )
    controls = ScipyCsrSpectralNormControls(0.0, 100, None, 0)

    with pytest.raises(FloatingPointError, match="spectral norm became nonfinite"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_shape_and_dtype_dependent_krylov_boundaries() -> None:
    """Verify pinned SciPy real and complex Krylov-dimension constraints exactly."""
    real_three: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    complex_three: CsrMatrix = sp.csr_array(
        np.diag(np.asarray((3.0j, 4.0j, 5.0j), dtype=np.complex128))
    )
    complex_four: CsrMatrix = sp.csr_array(
        np.diag(np.asarray((2.0j, 3.0j, 4.0j, 5.0j), dtype=np.complex128))
    )

    real_result = ScipyCsrSpectralNorm(
        real_three,
        ScipyCsrSpectralNormControls(1.0e-9, 500, 2, 0),
    ).result()
    complex_default_result = ScipyCsrSpectralNorm(
        complex_three,
        ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0),
    ).result()
    complex_explicit_result = ScipyCsrSpectralNorm(
        complex_four,
        ScipyCsrSpectralNormControls(1.0e-9, 500, 3, 0),
    ).result()

    assert abs(real_result - 5.0) <= 5.0e-9
    assert abs(complex_default_result - 5.0) <= 5.0e-9
    assert abs(complex_explicit_result - 5.0) <= 5.0e-9

    with pytest.raises(ValueError, match="at most two"):
        ScipyCsrSpectralNorm(
            sp.csr_array(np.eye(2, dtype=np.float64)),
            ScipyCsrSpectralNormControls(0.0, 100, 2, 0),
        ).result()
    with pytest.raises(ValueError, match="real krylov_dimension"):
        ScipyCsrSpectralNorm(
            real_three,
            ScipyCsrSpectralNormControls(0.0, 100, 3, 0),
        ).result()
    with pytest.raises(ValueError, match="complex krylov_dimension"):
        ScipyCsrSpectralNorm(
            complex_three,
            ScipyCsrSpectralNormControls(0.0, 100, 2, 0),
        ).result()
    with pytest.raises(ValueError, match="complex krylov_dimension"):
        ScipyCsrSpectralNorm(
            complex_four,
            ScipyCsrSpectralNormControls(0.0, 100, 4, 0),
        ).result()


def test_complex_start_vector_is_seeded_normalized_and_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complex ARPACK starts use deterministic real and imaginary draws.

    Parameters
    ----------
    monkeypatch
        Scoped backend observer returning an exact singular triplet.
    """
    matrix: CsrMatrix = sp.csr_array(
        np.diag(np.asarray((3.0j, 4.0j, 5.0j), dtype=np.complex128))
    )
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 29)
    starts: list[Float64Array | Complex128Array] = []

    def observing_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Capture the start and return the exact five-valued singular triplet.

        Parameters
        ----------
        _matrix
            Canonical complex diagonal CSR value supplied by the evaluator.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Full SciPy call arguments checked against the owner contract.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Exact finite left vector, singular value, and adjoint right vector.
        """
        assert k == 1
        assert ncv is None
        assert tol == controls.tolerance
        assert which == "LM"
        assert maxiter == controls.maximum_iterations
        assert return_singular_vectors is True
        assert solver == "arpack"
        starts.append(v0.copy())
        return (
            np.asarray(((0.0j,), (0.0j,), (1.0j,)), dtype=np.complex128),
            np.asarray((5.0,), dtype=np.float64),
            np.asarray(((0.0j, 0.0j, 1.0 + 0.0j),), dtype=np.complex128),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", observing_svds)

    first = ScipyCsrSpectralNorm(matrix, controls).result()
    second = ScipyCsrSpectralNorm(matrix, controls).result()

    assert first == 5.0
    assert second == 5.0
    assert len(starts) == 2
    assert np.array_equal(starts[0], starts[1])
    assert abs(float(scipy.linalg.norm(starts[0])) - 1.0) <= math.ulp(1.0)
    assert np.iscomplexobj(starts[0])
    assert bool(np.any(starts[0].imag != 0.0))


def test_rejects_nonunit_backend_singular_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify structural vector checks do not inherit a loose caller tolerance.

    Parameters
    ----------
    monkeypatch
        Scoped backend substitution returning zero vectors and a finite scalar.
    """
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    controls = ScipyCsrSpectralNormControls(0.9, 500, None, 0)

    def nonunit_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Return finite exact-shape vectors whose norms are zero.

        Parameters
        ----------
        _matrix
            Canonical matrix forwarded to the substituted backend.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Finite scalar with structurally invalid zero singular vectors.
        """
        return (
            np.zeros((3, 1), dtype=np.float64),
            np.asarray((5.0,), dtype=np.float64),
            np.zeros((1, 3), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", nonunit_svds)

    with pytest.raises(FloatingPointError, match="unit-norm qualification"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_translates_arpack_nonconvergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify deterministic ARPACK nonconvergence becomes a runtime failure.

    Parameters
    ----------
    monkeypatch
        Scoped backend substitution for an otherwise nondeterministic failure.
    """
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    def nonconvergent_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Raise the exact backend nonconvergence exception.

        Parameters
        ----------
        _matrix
            Canonical matrix forwarded to the substituted backend.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Raises
        ------
        scipy.sparse.linalg.ArpackNoConvergence
            Unconditionally, to represent iterative nonconvergence.
        """
        raise scipy.sparse.linalg.ArpackNoConvergence(
            "failed",
            np.empty((0,), dtype=np.float64),
            np.empty((3, 0), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", nonconvergent_svds)

    with pytest.raises(RuntimeError, match="did not converge"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_rejects_nonfinite_backend_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify nonfinite singular-triplet output fails at the trust boundary.

    Parameters
    ----------
    monkeypatch
        Scoped backend substitution for malformed numerical output.
    """
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    def nonfinite_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Return exact shapes containing one nonfinite singular value.

        Parameters
        ----------
        _matrix
            Canonical matrix forwarded to the substituted backend.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Exact-shape triplet containing one NaN singular value.
        """
        return (
            np.asarray(((1.0,), (0.0,), (0.0,)), dtype=np.float64),
            np.asarray((float("nan"),), dtype=np.float64),
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", nonfinite_svds)

    with pytest.raises(FloatingPointError, match="returned nonfinite"):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_rejects_nonfinite_residual_normalization_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify finite near-unit triplets cannot create infinite denominators.

    Parameters
    ----------
    monkeypatch
        Scoped backend substitution producing finite componentwise residual arithmetic
        but nonfinite scalar norm products.
    """
    matrix: CsrMatrix = sp.csr_array(
        np.diag(np.asarray((1.0e308, 1.0e308, 1.0e308), dtype=np.float64))
    )
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)
    component = math.sqrt(0.5) * (1.0 + 4.0 * math.ulp(1.0))

    def overflowing_normalization_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Return finite near-unit vectors whose scalar norm products overflow.

        Parameters
        ----------
        _matrix
            Canonical extreme-range diagonal matrix supplied by the evaluator.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Finite near-unit vectors and the largest finite binary64 scalar.
        """
        return (
            np.asarray(((component,), (component,), (0.0,)), dtype=np.float64),
            np.asarray((sys.float_info.max,), dtype=np.float64),
            np.asarray(((component, component, 0.0),), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", overflowing_normalization_svds)

    with pytest.raises(
        FloatingPointError,
        match="residual normalization became nonfinite",
    ):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_rejects_nonfinite_residual_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify overflow in a sparse matrix-vector product fails at that boundary.

    Parameters
    ----------
    monkeypatch
        Scoped backend substitution returning finite unit vectors and scalar.
    """
    matrix: CsrMatrix = sp.csr_array(
        np.asarray(
            (
                (1.0e308, 1.0e308, 1.0e308, 1.0e308),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            ),
            dtype=np.float64,
        )
    )
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    def overflowing_residual_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Return a finite triplet whose owner residual multiplication overflows.

        Parameters
        ----------
        _matrix
            Canonical extreme-range matrix supplied by the evaluator.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Finite unit-vector triplet producing a nonfinite matrix-vector product.
        """
        return (
            np.asarray(((1.0,), (0.0,), (0.0,)), dtype=np.float64),
            np.asarray((1.0e308,), dtype=np.float64),
            np.asarray(((0.5, 0.5, 0.5, 0.5),), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", overflowing_residual_svds)

    with pytest.raises(
        FloatingPointError, match="residual arithmetic became nonfinite"
    ):
        ScipyCsrSpectralNorm(matrix, controls).result()


def test_rejects_finite_triplet_outside_residual_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify successful backend return still requires both singular equations.

    Parameters
    ----------
    monkeypatch
        Scoped substitution returning unit vectors with an inconsistent singular value.
    """
    matrix: CsrMatrix = sp.csr_array(np.diag((3.0, 4.0, 5.0)))
    controls = ScipyCsrSpectralNormControls(1.0e-9, 500, None, 0)

    def inconsistent_svds(
        _matrix: CsrMatrix,
        *,
        k: int,
        ncv: int | None,
        tol: float,
        which: str,
        v0: Float64Array | Complex128Array,
        maxiter: int,
        return_singular_vectors: bool,
        solver: str,
    ) -> tuple[DenseMatrix, Float64Array, DenseMatrix]:
        """Return finite unit vectors that do not satisfy the reported value.

        Parameters
        ----------
        _matrix
            Canonical diagonal matrix supplied by the evaluator.
        k, ncv, tol, which, v0, maxiter, return_singular_vectors, solver
            Complete fixed SciPy call surface retained by the substitution.

        Returns
        -------
        tuple[DenseMatrix, Float64Array, DenseMatrix]
            Finite exact-shape triplet outside the owner residual cutoff.
        """
        return (
            np.asarray(((1.0,), (0.0,), (0.0,)), dtype=np.float64),
            np.asarray((5.0,), dtype=np.float64),
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
        )

    monkeypatch.setattr(scipy.sparse.linalg, "svds", inconsistent_svds)

    with pytest.raises(RuntimeError, match="exceeded residual tolerance"):
        ScipyCsrSpectralNorm(matrix, controls).result()
