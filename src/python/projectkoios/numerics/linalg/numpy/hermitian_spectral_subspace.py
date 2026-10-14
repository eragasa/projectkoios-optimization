"""Dense NumPy Hermitian spectral-subspace eigendecomposition backend.

The public template has already measured and validated the original input and constructs
the explicit Hermitian representative before invoking this backend. This module performs
one complete ``numpy.linalg.eigh`` call and returns transient built-in tuple eigenpairs
for shared validation and projector construction.
"""

from __future__ import annotations

from typing import final

import numpy as np

from ..hermitian_spectral_subspace import (
    HermitianSpectralMatrix,
    HermitianSpectralSubspaceSolver,
)


@final
class NumpyHermitianSpectralSubspaceSolver(HermitianSpectralSubspaceSolver):
    """Compute complete dense complex128 Hermitian eigensystems with NumPy."""

    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        """Return API-level backend identity, not linked-library provenance."""
        return "numpy.linalg.eigh"

    @property
    def _backend_version(self) -> str:
        """Return the runtime NumPy version as trusted API-level metadata."""
        return str(np.__version__)

    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> tuple[tuple[float, ...], tuple[tuple[complex, ...], ...]]:
        """Return the complete ascending eigensystem of one Hermitian representative.

        ``UPLO='L'`` is only a storage choice because the shared template has already
        constructed an exactly Hermitian matrix. Exactly ``numpy.linalg.LinAlgError`` is
        translated to the public convergence-failure class; other exceptions propagate.
        """
        matrix = np.asarray(representative, dtype=np.complex128, order="C")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO="L")
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "NumPy Hermitian eigendecomposition did not converge"
            ) from error

        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError(
                "NumPy eigendecomposition returned nonfinite eigenvalues"
            )
        if not np.all(np.isfinite(eigenvectors)):
            raise FloatingPointError(
                "NumPy eigendecomposition returned nonfinite eigenvectors"
            )

        return (
            tuple(float(value) for value in eigenvalues),
            tuple(tuple(complex(value) for value in row) for row in eigenvectors),
        )


__all__ = ["NumpyHermitianSpectralSubspaceSolver"]
