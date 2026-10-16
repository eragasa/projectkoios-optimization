"""Dense SciPy backend for nonempty batched complete Hermitian spectra."""

from __future__ import annotations

from typing import final

import numpy as np
import scipy
import scipy.linalg

from ..hermitian_spectrum import (
    BatchedHermitianSpectrumSolver,
    HermitianSpectra,
    HermitianSpectrumMatrices,
)


@final
class ScipyBatchedHermitianSpectrumSolver(BatchedHermitianSpectrumSolver):
    """Computes complete spectra with one batched SciPy complex128 operation.

    Notes
    -----
    The leading array axis indexes matrices. One public ``scipy.linalg.eigvalsh`` call
    evaluates the complete nonempty stack with the lower triangle and the full-spectrum
    divide-and-conquer driver.
    """

    __slots__ = ()

    @property
    def _backend_identifier(self) -> str:
        """Returns the API-level SciPy operation identity.

        Returns
        -------
        str
            Stable backend identifier.
        """
        return "scipy.linalg.eigvalsh"

    @property
    def _backend_version(self) -> str:
        """Returns the runtime SciPy package version.

        Returns
        -------
        str
            SciPy version string.
        """
        return str(scipy.__version__)

    def _eigvalsh_validated(
        self,
        representatives: HermitianSpectrumMatrices,
        /,
    ) -> HermitianSpectra:
        """Computes every complete spectrum in one batched ``eigvalsh`` call.

        Parameters
        ----------
        representatives
            Nonempty same-dimension batch of represented-Hermitian matrices.

        Returns
        -------
        HermitianSpectra
            Complete ascending spectra converted to immutable built-in values.

        Raises
        ------
        RuntimeError
            If ``scipy.linalg.eigvalsh`` does not converge.
        FloatingPointError
            If SciPy returns a nonfinite eigenvalue.
        """
        matrices = np.asarray(representatives, dtype=np.complex128, order="C")
        try:
            # SciPy interprets the leading axis as a batch and dispatches one public
            # dense operation without a Python loop in this owner backend.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                eigenvalues = scipy.linalg.eigvalsh(
                    matrices,
                    lower=True,
                    overwrite_a=False,
                    check_finite=False,
                    driver="evd",
                )
        except np.linalg.LinAlgError as error:
            # The public failure class remains backend-neutral even though the concrete
            # nonconvergence originated in SciPy or its LAPACK driver.
            raise RuntimeError(
                "SciPy batched Hermitian eigensolver did not converge"
            ) from error
        # Structural and ordering checks remain in the shared template; this check
        # classifies nonfinite numerical output as an arithmetic failure.
        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError(
                "SciPy batched eigensolver returned nonfinite eigenvalues"
            )
        return tuple(
            tuple(float(value) for value in spectrum) for spectrum in eigenvalues
        )


__all__ = ["ScipyBatchedHermitianSpectrumSolver"]
