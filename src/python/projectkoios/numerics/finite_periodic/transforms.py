"""Dependency-free diagnostics for finite periodic Fourier transforms.

Concrete array transforms live in capability-local backend modules. This module owns
only immutable numerical evidence records; it assigns no physical units, shell policy,
threshold, or scientific disposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from projectkoios.numerics.validation import exact_nonnegative_float


@final
@dataclass(frozen=True, slots=True)
class FinitePeriodicParsevalDiagnostics:
    """Record threshold-free finite-transform Parseval evidence.

    Parameters
    ----------
    sample_mean_squared_frobenius_norm
        Mean squared Frobenius norm over the complete uniform sample mesh.
    coefficient_sum_squared_frobenius_norm
        Sum of squared Frobenius norms over the complete Fourier inventory.
    absolute_defect
        Absolute difference between the two represented energies.

    Notes
    -----
    This record reports arithmetic evidence only. A consumer owns units, scaling beyond
    the declared transform normalization, tolerances, and acceptance.
    """

    sample_mean_squared_frobenius_norm: float
    coefficient_sum_squared_frobenius_norm: float
    absolute_defect: float

    def __post_init__(self) -> None:
        """Validate nonnegative finite values and their exact stored relation.

        Raises
        ------
        TypeError
            If a value is not an exact built-in float.
        ValueError
            If a value is negative, nonfinite, or relationally inconsistent.
        """
        sample = exact_nonnegative_float(
            self.sample_mean_squared_frobenius_norm,
            "sample_mean_squared_frobenius_norm",
        )
        coefficients = exact_nonnegative_float(
            self.coefficient_sum_squared_frobenius_norm,
            "coefficient_sum_squared_frobenius_norm",
        )
        defect = exact_nonnegative_float(self.absolute_defect, "absolute_defect")
        if defect != abs(sample - coefficients):
            raise ValueError("absolute_defect must equal the represented energy defect")


@final
@dataclass(frozen=True, slots=True)
class FinitePeriodicHermiticityDiagnostics:
    """Record cyclic and non-Nyquist matrix-coefficient Hermiticity defects.

    Parameters
    ----------
    twisted_cyclic_maximum_frobenius_defect
        Maximum Frobenius defect after pairing every centered mode with its cyclic
        negative and applying the exact mesh-origin wrap phase.
    ordinary_maximum_frobenius_defect
        Maximum Frobenius defect over pairs not touching a Nyquist boundary, where no
        cyclic wrap phase is required. For odd meshes this covers the complete
        inventory because no Nyquist mode exists.

    Notes
    -----
    The diagnostics report represented coefficient relations without asserting that the
    sampled matrices are physical Hamiltonians or imposing a tolerance.
    """

    twisted_cyclic_maximum_frobenius_defect: float
    ordinary_maximum_frobenius_defect: float

    def __post_init__(self) -> None:
        """Validate exact finite nonnegative defects.

        Raises
        ------
        TypeError
            If a defect is not an exact built-in float.
        ValueError
            If a defect is negative or nonfinite.
        """
        exact_nonnegative_float(
            self.twisted_cyclic_maximum_frobenius_defect,
            "twisted_cyclic_maximum_frobenius_defect",
        )
        exact_nonnegative_float(
            self.ordinary_maximum_frobenius_defect,
            "ordinary_maximum_frobenius_defect",
        )


__all__ = [
    "FinitePeriodicHermiticityDiagnostics",
    "FinitePeriodicParsevalDiagnostics",
]
