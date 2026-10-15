"""Dense NumPy backend for one complex-unitary right alignment."""

from __future__ import annotations

from typing import final

import numpy as np

from ..procrustes import (
    ComplexUnitaryProcrustesMatrix,
    ComplexUnitaryProcrustesSolver,
)


@final
class NumpyComplexUnitaryProcrustesSolver(ComplexUnitaryProcrustesSolver):
    """Compute the square cross-matrix SVD with NumPy complex128 arithmetic."""

    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        """Return the API-level NumPy operation identity.

        Returns
        -------
        str
            Stable backend identifier.
        """
        return "numpy.linalg.svd"

    @property
    def _backend_version(self) -> str:
        """Return the runtime NumPy package version.

        Returns
        -------
        str
            NumPy version string.
        """
        return str(np.__version__)

    def _svd_validated(
        self,
        cross_matrix: ComplexUnitaryProcrustesMatrix,
        /,
    ) -> tuple[
        tuple[float, ...],
        ComplexUnitaryProcrustesMatrix,
        ComplexUnitaryProcrustesMatrix,
    ]:
        """Compute a complete square SVD of the validated cross matrix.

        Parameters
        ----------
        cross_matrix
            Finite immutable square cross matrix supplied by the shared template.

        Returns
        -------
        tuple
            Singular values, left singular vectors, and adjoint right singular vectors
            converted to exact built-in immutable values.

        Raises
        ------
        RuntimeError
            If ``numpy.linalg.svd`` does not converge.
        FloatingPointError
            If NumPy returns a nonfinite singular value or vector entry.
        """
        matrix = np.asarray(cross_matrix, dtype=np.complex128, order="C")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                left, singular_values, right_adjoint = np.linalg.svd(
                    matrix,
                    full_matrices=True,
                    compute_uv=True,
                )
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "NumPy singular-value decomposition did not converge"
            ) from error
        if not np.all(np.isfinite(singular_values)):
            raise FloatingPointError("NumPy SVD returned nonfinite singular values")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right_adjoint)):
            raise FloatingPointError("NumPy SVD returned nonfinite singular vectors")
        return (
            tuple(float(value) for value in singular_values),
            tuple(tuple(complex(value) for value in row) for row in left),
            tuple(tuple(complex(value) for value in row) for row in right_adjoint),
        )


__all__ = ["NumpyComplexUnitaryProcrustesSolver"]
