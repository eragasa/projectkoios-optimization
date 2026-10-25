"""Evaluate spectral norms of SciPy CSR binary64 matrices.

The evaluator uses deterministic scale-safe direct arithmetic when the smaller matrix
axis has size at most two and SciPy ARPACK ``svds`` for larger matrices. Both branches
preserve sparse storage and return the same scalar spectral-norm identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from projectkoios.numerics.numpy.typing import (
    Complex128Array,
    DenseMatrix,
    Float64Array,
)
from projectkoios.numerics.scipy.typing import CsrMatrix

from ..matrix_norm import MatrixNormABC, MatrixNormEnum
from ._csr import _canonical_binary64_csr


@final
@dataclass(frozen=True, slots=True)
class ScipyCsrSpectralNormControls:
    """Store explicit deterministic controls for sparse spectral-norm evaluation.

    Parameters
    ----------
    tolerance
        Exact built-in finite float in the half-open interval ``[0.0, 1.0)``. Zero
        requests SciPy's machine-precision convergence mode.
    maximum_iterations
        Exact built-in positive ARPACK iteration limit.
    krylov_dimension
        ``None`` to delegate Krylov dimension selection to pinned SciPy, or an exact
        built-in positive integer validated against matrix shape and dtype during
        evaluation. The evaluator passes this field only as SciPy's private-boundary
        ``ncv`` keyword.
    seed
        Exact built-in nonnegative seed for deterministic start-vector construction.
    """

    tolerance: float
    maximum_iterations: int
    krylov_dimension: int | None
    seed: int

    def __post_init__(self) -> None:
        """Validate exact scalar types and shape-independent control boundaries.

        Raises
        ------
        TypeError
            If a control does not use its required exact built-in type.
        ValueError
            If a numeric control lies outside its accepted range.
        """
        if type(self.tolerance) is not float:
            raise TypeError("tolerance must be an exact built-in float")
        if not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite")
        if self.tolerance < 0.0 or self.tolerance >= 1.0:
            raise ValueError("tolerance must satisfy 0.0 <= tolerance < 1.0")
        if type(self.maximum_iterations) is not int:
            raise TypeError("maximum_iterations must be an exact built-in integer")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.krylov_dimension is not None:
            if type(self.krylov_dimension) is not int:
                raise TypeError(
                    "krylov_dimension must be None or an exact built-in integer"
                )
            if self.krylov_dimension <= 0:
                raise ValueError("krylov_dimension must be positive when supplied")
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact built-in integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@final
class ScipyCsrSpectralNorm(MatrixNormABC[CsrMatrix]):
    """Evaluate the largest singular value of one binary64 SciPy CSR matrix.

    Parameters
    ----------
    matrix
        Exact nonempty finite two-dimensional SciPy CSR matrix or array with exact
        ``float64`` or ``complex128`` storage.
    controls
        Exact owner controls. No implicit control record is constructed.

    Notes
    -----
    The matrix is retained by reference and read without mutation. Evaluation
    canonicalizes a private copy. Matrices with minimum dimension at most two use the
    primary direct small-Gram branch; larger matrices use approximate ARPACK iteration.
    Neither branch densifies the caller matrix or falls back to another solver.
    """

    __slots__ = ("_controls",)

    def __init__(
        self,
        matrix: CsrMatrix,
        controls: ScipyCsrSpectralNormControls,
        /,
    ) -> None:
        """Retain one CSR matrix and its exact sparse convergence controls.

        Parameters
        ----------
        matrix
            CSR matrix or array evaluated by :meth:`result`.
        controls
            Exact immutable owner controls for direct and iterative branches.

        Raises
        ------
        TypeError
            If ``controls`` is not an exact ``ScipyCsrSpectralNormControls`` instance.
        """
        if type(controls) is not ScipyCsrSpectralNormControls:
            raise TypeError("controls must be ScipyCsrSpectralNormControls")
        super().__init__(matrix)
        self._controls = controls

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the spectral-norm semantic identity.

        Returns
        -------
        MatrixNormEnum
            ``MatrixNormEnum.spectral``.
        """
        return MatrixNormEnum.spectral

    def _evaluate(self, matrix: CsrMatrix, /) -> float:
        """Validate one CSR matrix and return its largest singular value.

        Parameters
        ----------
        matrix
            Retained CSR matrix supplied at construction.

        Returns
        -------
        float
            Exact built-in finite nonnegative largest singular value.

        Raises
        ------
        TypeError
            If the matrix representation or exact dtype is unsupported.
        ValueError
            If the matrix is empty, nonfinite, not two-dimensional, or incompatible
            with its explicit Krylov dimension.
        RuntimeError
            If ARPACK does not converge or its finite singular triplet fails internal
            residual qualification.
        FloatingPointError
            If canonicalization, direct arithmetic, iterative output, or residual
            arithmetic becomes nonfinite.
        """
        canonical = _canonical_binary64_csr(matrix)
        minimum_dimension = min(canonical.shape)
        controls = self._controls
        if minimum_dimension <= 2:
            if controls.krylov_dimension is not None:
                raise ValueError(
                    "krylov_dimension must be None when minimum matrix dimension is "
                    "at most two"
                )
            return self._direct_result(canonical)

        krylov_dimension = controls.krylov_dimension
        if krylov_dimension is not None:
            minimum_krylov_dimension = (
                3 if canonical.dtype == np.dtype(np.complex128) else 2
            )
            if not (minimum_krylov_dimension <= krylov_dimension < minimum_dimension):
                if canonical.dtype == np.dtype(np.complex128):
                    raise ValueError(
                        "complex krylov_dimension must satisfy 3 <= "
                        "krylov_dimension < minimum matrix dimension"
                    )
                raise ValueError(
                    "real krylov_dimension must satisfy 2 <= krylov_dimension < "
                    "minimum matrix dimension"
                )
        if canonical.data.size == 0:
            return 0.0
        return self._iterative_result(canonical)

    def _direct_result(self, matrix: CsrMatrix, /) -> float:
        """Return a scale-safe direct norm for minimum dimension at most two.

        Parameters
        ----------
        matrix
            Canonical finite nonempty CSR matrix with minimum dimension at most two.

        Returns
        -------
        float
            Exact built-in finite nonnegative rounded direct spectral norm.

        Raises
        ------
        FloatingPointError
            If entry magnitude, sparse Gram arithmetic, or final rescaling becomes
            nonfinite, or if the squared norm becomes negative.

        Notes
        -----
        The method forms only a sparse one-by-one or two-by-two scaled Gram matrix. It
        never materializes the input matrix as a dense value.
        """
        if matrix.data.size == 0:
            return 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            scale = float(np.max(np.abs(matrix.data), initial=0.0))
        if not math.isfinite(scale):
            raise FloatingPointError("CSR entry magnitude became nonfinite")
        if scale == 0.0:
            return 0.0

        scaled = matrix.copy()
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scaled.data /= scale
            if matrix.shape[0] <= matrix.shape[1]:
                gram = scaled @ scaled.conjugate().transpose()
            else:
                gram = scaled.conjugate().transpose() @ scaled
        gram.sum_duplicates()
        gram.eliminate_zeros()
        if not bool(np.isfinite(gram.data).all()):
            raise FloatingPointError("direct sparse Gram arithmetic became nonfinite")

        first_diagonal = complex(gram[0, 0])
        if gram.shape[0] == 1:
            scaled_squared_norm = first_diagonal.real
        else:
            second_diagonal = complex(gram[1, 1])
            off_diagonal = complex(gram[0, 1])
            discriminant = math.hypot(
                first_diagonal.real - second_diagonal.real,
                2.0 * abs(off_diagonal),
            )
            scaled_squared_norm = 0.5 * (
                first_diagonal.real + second_diagonal.real + discriminant
            )
        if not math.isfinite(scaled_squared_norm) or scaled_squared_norm < 0.0:
            raise FloatingPointError("direct sparse spectral arithmetic became invalid")
        with np.errstate(over="ignore", invalid="ignore"):
            result = scale * math.sqrt(scaled_squared_norm)
        if not math.isfinite(result):
            raise FloatingPointError("sparse spectral norm became nonfinite")
        return float(result)

    def _start_vector(self, matrix: CsrMatrix, /) -> Float64Array | Complex128Array:
        """Construct the deterministic normalized ARPACK start vector.

        Parameters
        ----------
        matrix
            Canonical iterative-branch CSR matrix defining vector length and dtype.

        Returns
        -------
        Float64Array | Complex128Array
            Finite unit-norm vector derived only from the explicit owner seed.

        Raises
        ------
        FloatingPointError
            If start-vector norm computation is nonfinite or zero.
        """
        dimension = min(matrix.shape)
        rng = np.random.default_rng(self._controls.seed)
        if matrix.dtype == np.dtype(np.complex128):
            start: Float64Array | Complex128Array = np.asarray(
                rng.standard_normal(dimension) + 1.0j * rng.standard_normal(dimension),
                dtype=np.complex128,
            )
        else:
            start = np.asarray(rng.standard_normal(dimension), dtype=np.float64)
        start_norm = float(scipy.linalg.norm(start, check_finite=False))
        if not math.isfinite(start_norm) or start_norm == 0.0:
            raise FloatingPointError(
                "sparse singular-value start vector became invalid"
            )
        return start / start_norm

    def _iterative_result(self, matrix: CsrMatrix, /) -> float:
        """Return one internally qualified ARPACK largest singular value.

        Parameters
        ----------
        matrix
            Canonical finite nonzero CSR matrix with minimum dimension at least three.

        Returns
        -------
        float
            Exact built-in finite nonnegative qualified singular value.

        Raises
        ------
        RuntimeError
            If ARPACK or its decomposition does not converge, or if a finite triplet
            exceeds the owner residual cutoff.
        FloatingPointError
            If backend output, singular-vector structure, matrix-vector arithmetic, or
            residual norm arithmetic becomes invalid or nonfinite.
        """
        controls = self._controls
        start = self._start_vector(matrix)
        try:
            # Request both singular-vector sides solely for internal qualification. The
            # public result remains the scalar norm required by MatrixNormABC.
            left_vectors: DenseMatrix
            singular_values: Float64Array
            right_adjoint_vectors: DenseMatrix
            left_vectors, singular_values, right_adjoint_vectors = (
                scipy.sparse.linalg.svds(
                    matrix,
                    k=1,
                    ncv=controls.krylov_dimension,
                    tol=controls.tolerance,
                    which="LM",
                    v0=start,
                    maxiter=controls.maximum_iterations,
                    return_singular_vectors=True,
                    solver="arpack",
                )
            )
        except scipy.sparse.linalg.ArpackNoConvergence as error:
            raise RuntimeError(
                "SciPy sparse singular-value solver did not converge"
            ) from error
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "SciPy sparse singular-value decomposition did not converge"
            ) from error

        expected_left_shape = (matrix.shape[0], 1)
        expected_right_shape = (1, matrix.shape[1])
        if (
            type(left_vectors) is not np.ndarray
            or type(singular_values) is not np.ndarray
            or type(right_adjoint_vectors) is not np.ndarray
            or left_vectors.shape != expected_left_shape
            or singular_values.shape != (1,)
            or right_adjoint_vectors.shape != expected_right_shape
        ):
            raise FloatingPointError(
                "SciPy sparse singular-value solver returned invalid shapes"
            )
        if (
            not bool(np.isfinite(left_vectors).all())
            or not bool(np.isfinite(singular_values).all())
            or not bool(np.isfinite(right_adjoint_vectors).all())
        ):
            raise FloatingPointError(
                "SciPy sparse singular-value solver returned nonfinite values"
            )

        singular_value = float(singular_values[0])
        if singular_value < 0.0:
            raise FloatingPointError(
                "SciPy sparse singular-value solver returned a negative value"
            )
        left_vector = left_vectors[:, 0]
        right_vector = right_adjoint_vectors[0, :].conjugate()
        left_norm = float(scipy.linalg.norm(left_vector, check_finite=False))
        right_norm = float(scipy.linalg.norm(right_vector, check_finite=False))
        dimension_floor = 64.0 * math.ulp(1.0) * max(matrix.shape)
        if (
            not math.isfinite(left_norm)
            or not math.isfinite(right_norm)
            or abs(left_norm - 1.0) > dimension_floor
            or abs(right_norm - 1.0) > dimension_floor
        ):
            raise FloatingPointError(
                "SciPy sparse singular vectors failed unit-norm qualification"
            )

        # Qualify both singular equations against the exact canonical matrix passed to
        # ARPACK. Dense intermediates are vectors only; no matrix densification occurs.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            matrix_right = matrix @ right_vector
            adjoint_left = matrix.conjugate().transpose() @ left_vector
            left_scaled = singular_value * left_vector
            right_scaled = singular_value * right_vector
            left_residual = matrix_right - left_scaled
            right_residual = adjoint_left - right_scaled
        if (
            not bool(np.isfinite(matrix_right).all())
            or not bool(np.isfinite(adjoint_left).all())
            or not bool(np.isfinite(left_residual).all())
            or not bool(np.isfinite(right_residual).all())
        ):
            raise FloatingPointError(
                "sparse singular-triplet residual arithmetic became nonfinite"
            )

        matrix_right_norm = float(scipy.linalg.norm(matrix_right, check_finite=False))
        adjoint_left_norm = float(scipy.linalg.norm(adjoint_left, check_finite=False))
        left_residual_norm = float(scipy.linalg.norm(left_residual, check_finite=False))
        right_residual_norm = float(
            scipy.linalg.norm(right_residual, check_finite=False)
        )
        residual_terms = (
            matrix_right_norm,
            adjoint_left_norm,
            left_residual_norm,
            right_residual_norm,
        )
        if not all(math.isfinite(term) for term in residual_terms):
            raise FloatingPointError(
                "sparse singular-triplet residual norm became nonfinite"
            )

        left_scaled_norm = singular_value * left_norm
        right_scaled_norm = singular_value * right_norm
        if not math.isfinite(left_scaled_norm) or not math.isfinite(right_scaled_norm):
            raise FloatingPointError(
                "sparse singular-triplet residual normalization became nonfinite"
            )
        left_denominator = max(matrix_right_norm, left_scaled_norm)
        right_denominator = max(adjoint_left_norm, right_scaled_norm)
        if left_denominator == 0.0:
            left_relative = 0.0 if left_residual_norm == 0.0 else math.inf
        else:
            left_relative = left_residual_norm / left_denominator
        if right_denominator == 0.0:
            right_relative = 0.0 if right_residual_norm == 0.0 else math.inf
        else:
            right_relative = right_residual_norm / right_denominator
        residual_cutoff = max(controls.tolerance, dimension_floor)
        if max(left_relative, right_relative) > residual_cutoff:
            raise RuntimeError(
                "SciPy sparse singular triplet exceeded residual tolerance"
            )
        return singular_value


__all__ = ["ScipyCsrSpectralNorm", "ScipyCsrSpectralNormControls"]
