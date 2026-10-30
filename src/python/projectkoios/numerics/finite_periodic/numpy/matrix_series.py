"""Canonical NumPy records for finite matrix Fourier series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, final

import numpy as np

from projectkoios.numerics.finite_periodic.finite_series import (
    _MAX_EXACT_BINARY64_INTEGER,
)
from projectkoios.numerics.numpy.typing import Complex128Array, Int64Array
from projectkoios.numerics.numpy.validation import (
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)
from projectkoios.numerics.validation import exact_positive_integer


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyFiniteFourierMatrixSeries:
    """Store selected integer modes and square complex matrix coefficients.

    Parameters
    ----------
    dimension
        One, two, or three coordinate components.
    integer_modes
        Canonical immutable int64 array with shape ``(M, dimension)``.
    matrix_coefficients
        Canonical immutable complex128 array with shape ``(M, n, n)`` for positive
        ``n``. The ordered inventory may be empty or contain duplicate modes.

    Notes
    -----
    The raw constructor accepts only canonical immutable byte-backed arrays. Use
    :meth:`from_arrays` for defensive copies or :meth:`from_immutable_arrays` for an
    explicit zero-copy boundary.
    """

    dimension: int
    integer_modes: Int64Array = field(repr=False)
    matrix_coefficients: Complex128Array = field(repr=False)

    @classmethod
    def from_arrays(
        cls,
        dimension: int,
        integer_modes: Int64Array,
        matrix_coefficients: Complex128Array,
        /,
    ) -> Self:
        """Construct from exact arrays using immutable defensive copies.

        Parameters
        ----------
        dimension
            One, two, or three coordinate components.
        integer_modes
            Exact int64 mode array.
        matrix_coefficients
            Exact finite complex128 matrix-coefficient array.

        Returns
        -------
        NumpyFiniteFourierMatrixSeries
            Canonical independently owned finite series.
        """
        return cls(
            dimension,
            NumpyImmutableArrayConstructor(
                integer_modes,
                "integer_modes",
                np.dtype(np.int64),
            ).execute(),
            NumpyImmutableArrayConstructor(
                matrix_coefficients,
                "matrix_coefficients",
                np.dtype(np.complex128),
            ).execute(),
        )

    @classmethod
    def from_immutable_arrays(
        cls,
        dimension: int,
        integer_modes: Int64Array,
        matrix_coefficients: Complex128Array,
        /,
    ) -> Self:
        """Construct without copying canonical immutable arrays.

        Parameters
        ----------
        dimension
            One, two, or three coordinate components.
        integer_modes
            Canonical immutable int64 mode array.
        matrix_coefficients
            Canonical immutable finite complex128 coefficient array.

        Returns
        -------
        NumpyFiniteFourierMatrixSeries
            Series retaining both supplied arrays by identity.
        """
        return cls(dimension, integer_modes, matrix_coefficients)

    def __post_init__(self) -> None:
        """Validate canonical arrays and correlated series shape.

        Raises
        ------
        TypeError
            If fields do not have exact nominal or array representations.
        ValueError
            If dimension, ranks, shapes, squareness, or finiteness is invalid.
        """
        dimension = exact_positive_integer(self.dimension, "dimension")
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be one, two, or three")
        modes = NumpyImmutableArrayValidator(
            self.integer_modes,
            "integer_modes",
            np.dtype(np.int64),
        ).execute()
        coefficients = NumpyImmutableArrayValidator(
            self.matrix_coefficients,
            "matrix_coefficients",
            np.dtype(np.complex128),
        ).execute()
        if modes.ndim != 2 or modes.shape[1] != dimension:
            raise ValueError("integer_modes must have shape (term_count, dimension)")
        if bool(
            (
                (modes < -_MAX_EXACT_BINARY64_INTEGER)
                | (modes > _MAX_EXACT_BINARY64_INTEGER)
            ).any()
        ):
            raise ValueError("integer mode magnitudes must not exceed 2**53")
        if coefficients.ndim != 3:
            raise ValueError("matrix_coefficients must have shape (term_count, n, n)")
        if coefficients.shape[0] != modes.shape[0]:
            raise ValueError("one matrix coefficient is required per integer mode")
        if coefficients.shape[1] <= 0 or coefficients.shape[2] != coefficients.shape[1]:
            raise ValueError("matrix coefficients must be nonempty and square")

    @property
    def term_count(self) -> int:
        """Return the ordered coefficient count.

        Returns
        -------
        int
            Nonnegative number of terms.
        """
        return int(self.integer_modes.shape[0])

    @property
    def matrix_size(self) -> int:
        """Return the shared square-matrix dimension.

        Returns
        -------
        int
            Positive matrix dimension.
        """
        return int(self.matrix_coefficients.shape[1])


__all__ = ["NumpyFiniteFourierMatrixSeries"]
