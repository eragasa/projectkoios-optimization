"""Dense NumPy backend for nonempty batched complete Hermitian spectra."""

from __future__ import annotations

from typing import final

import numpy as np

from ..hermitian_spectrum import (
    BatchedHermitianSpectrumSolver,
    HermitianSpectra,
    HermitianSpectrumMatrices,
)


@final
class NumpyBatchedHermitianSpectrumSolver(BatchedHermitianSpectrumSolver):
    """Computes complete spectra with one batched NumPy complex128 operation.

    Notes
    -----
    The leading array axis indexes matrices. One public ``numpy.linalg.eigvalsh`` call
    evaluates the complete nonempty stack with ``UPLO="L"``.
    """

    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        """Returns the API-level NumPy operation identity.

        Returns
        -------
        str
            Stable backend identifier.
        """
        return "numpy.linalg.eigvalsh"

    @property
    def _backend_version(self) -> str:
        """Returns the runtime NumPy package version.

        Returns
        -------
        str
            NumPy version string.
        """
        return str(np.__version__)

    def _eigvalsh_validated(
        self,
        representatives: HermitianSpectrumMatrices,
        /,
    ) -> HermitianSpectra:
        """Computes every complete spectrum in one batched ``eigvalsh`` call.

        Parameters
        ----------
        representatives
            Nonempty same-dimension batch of exactly Hermitian tuple matrices.

        Returns
        -------
        HermitianSpectra
            Complete ascending spectra converted to built-in immutable values.

        Raises
        ------
        RuntimeError
            If ``numpy.linalg.eigvalsh`` does not converge.
        FloatingPointError
            If NumPy returns a nonfinite eigenvalue.
        """
        matrices = np.asarray(representatives, dtype=np.complex128, order="C")
        try:
            # NumPy interprets the leading axis as a stack, so this remains one public
            # batched operation rather than a Python loop over scalar eigensolvers.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                eigenvalues = np.linalg.eigvalsh(matrices, UPLO="L")
        except np.linalg.LinAlgError as error:
            # The public contract exposes backend nonconvergence without leaking a
            # backend-specific exception class into dependency-free consumers.
            raise RuntimeError(
                "NumPy batched Hermitian eigensolver did not converge"
            ) from error
        # The shared template validates structure and ordering; this backend check
        # classifies nonfinite numerical output as an arithmetic failure.
        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError(
                "NumPy batched eigensolver returned nonfinite eigenvalues"
            )
        return tuple(
            tuple(float(value) for value in spectrum) for spectrum in eigenvalues
        )


__all__ = ["NumpyBatchedHermitianSpectrumSolver"]
