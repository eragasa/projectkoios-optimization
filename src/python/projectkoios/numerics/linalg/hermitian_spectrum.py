"""Complete spectra of finite dense complex Hermitian matrix batches.

The dependency-free contract accepts canonical immutable matrices made from exact
built-in tuple and complex storage. For each finite square matrix ``A``, the
solver measures represented-input Hermiticity, constructs
``H = (A + A.conjugate().T) / 2``, and returns every eigenvalue of ``H`` in ascending
order with algebraic multiplicity. The capability returns no eigenvectors, projectors,
gap decisions, or scientific policy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import final

from projectkoios.numerics.typing import FiniteComplexMatrix
from projectkoios.numerics.validation import (
    exact_nonnegative_float,
    exact_positive_integer,
    exact_relative_float,
    validate_finite_complex_matrix,
    validate_finite_float_tuple,
)

type HermitianSpectrumMatrix = FiniteComplexMatrix
type HermitianSpectrumMatrices = tuple[HermitianSpectrumMatrix, ...]
type HermitianSpectra = tuple[tuple[float, ...], ...]


def _maximum_difference(
    left: HermitianSpectrumMatrix,
    right: HermitianSpectrumMatrix,
    /,
) -> float:
    """Returns the maximum absolute entrywise difference.

    Parameters
    ----------
    left, right
        Equally shaped finite matrices.

    Returns
    -------
    float
        Maximum absolute difference.

    Raises
    ------
    FloatingPointError
        If subtraction or magnitude becomes nonfinite.
    """
    maximum = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        for left_entry, right_entry in zip(left_row, right_row, strict=True):
            # Component finiteness is checked before accepting the magnitude because
            # subtraction can overflow even when both operands are finite.
            difference = left_entry - right_entry
            magnitude = abs(difference)
            if (
                not math.isfinite(difference.real)
                or not math.isfinite(difference.imag)
                or not math.isfinite(magnitude)
            ):
                raise FloatingPointError("matrix difference became nonfinite")
            maximum = max(maximum, magnitude)
    return maximum


def _conjugate_transpose(matrix: HermitianSpectrumMatrix, /) -> HermitianSpectrumMatrix:
    """Returns the conjugate transpose of a square matrix.

    Parameters
    ----------
    matrix
        Validated square matrix.

    Returns
    -------
    HermitianSpectrumMatrix
        Conjugate transpose.
    """
    dimension = len(matrix)
    return tuple(
        tuple(matrix[column][row].conjugate() for column in range(dimension))
        for row in range(dimension)
    )


def _representative(matrix: HermitianSpectrumMatrix, /) -> HermitianSpectrumMatrix:
    """Constructs ``(A + A†) / 2`` with checked binary64 arithmetic.

    Parameters
    ----------
    matrix
        Validated square input matrix.

    Returns
    -------
    HermitianSpectrumMatrix
        Exactly Hermitian representative.

    Raises
    ------
    FloatingPointError
        If construction becomes nonfinite or not exactly Hermitian.
    """
    dimension = len(matrix)
    result = tuple(
        tuple(
            (matrix[row][column] + matrix[column][row].conjugate()) * 0.5
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    if any(
        not math.isfinite(entry.real) or not math.isfinite(entry.imag)
        for row in result
        for entry in row
    ):
        raise FloatingPointError("Hermitian symmetrization became nonfinite")
    # Backends may read only one triangle; require the represented matrix itself to
    # satisfy Hermiticity rather than relying on that backend convention.
    if result != _conjugate_transpose(result):
        raise FloatingPointError("constructed representative is not exactly Hermitian")
    return result


def _spectra(value: HermitianSpectra, dimension: int, /) -> HermitianSpectra:
    """Validate canonical complete finite ascending spectra.

    Parameters
    ----------
    value
        Candidate batch-major spectra.
    dimension
        Required width of every spectrum.

    Returns
    -------
    HermitianSpectra
        Immutable built-in spectra.

    Raises
    ------
    TypeError
        If containers or entries are unsupported.
    ValueError
        If width, finiteness, or ordering is invalid.
    """
    if type(value) is not tuple:
        raise TypeError("eigenvalues must be an exact tuple")
    for index, spectrum in enumerate(value):
        validate_finite_float_tuple(spectrum, f"eigenvalues[{index}]")
        if len(spectrum) != dimension:
            raise ValueError(f"eigenvalues[{index}] length must equal dimension")
        # Nondecreasing order retains repeated eigenvalues and therefore preserves
        # algebraic multiplicity without introducing a degeneracy policy.
        if any(
            left > right for left, right in zip(spectrum, spectrum[1:], strict=False)
        ):
            raise ValueError(f"eigenvalues[{index}] must be in ascending order")
    return value


@final
@dataclass(frozen=True, slots=True)
class BatchedHermitianSpectrumTolerance:
    """Defines tolerances for represented-input Hermiticity.

    Parameters
    ----------
    hermiticity_absolute
        Absolute defect cutoff component.
    hermiticity_relative
        Relative component multiplied by each matrix's maximum entry magnitude.

    Notes
    -----
    Matrix ``b`` uses the cutoff
    ``hermiticity_absolute + hermiticity_relative * max(abs(A_b))``. A represented
    defect equal to the cutoff is accepted.
    """

    hermiticity_absolute: float
    hermiticity_relative: float

    def __post_init__(self) -> None:
        """Validate canonical finite tolerance components."""
        exact_nonnegative_float(self.hermiticity_absolute, "hermiticity_absolute")
        exact_relative_float(self.hermiticity_relative, "hermiticity_relative")


@final
@dataclass(frozen=True, slots=True)
class BatchedHermitianSpectrumDefinition:
    """Defines a canonical immutable matrix batch and Hermiticity policy.

    Notes
    -----
    The sole constructor accepts exact canonical matrix storage. The :meth:`empty`
    constructor supplies dimension when no matrix carries shape.
    """

    matrices: HermitianSpectrumMatrices
    tolerance: BatchedHermitianSpectrumTolerance
    _empty_dimension: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate canonical matrix storage and its inferred or empty dimension."""
        if type(self.matrices) is not tuple:
            raise TypeError("matrices must be an exact tuple")
        if type(self.tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError("tolerance must be BatchedHermitianSpectrumTolerance")
        if not self.matrices:
            if self._empty_dimension is None:
                raise ValueError("empty matrices require an explicit dimension")
            exact_positive_integer(self._empty_dimension, "dimension")
            return
        if self._empty_dimension is not None:
            raise ValueError("empty dimension must be absent for a nonempty batch")
        validate_finite_complex_matrix(self.matrices[0], "matrices[0]")
        dimension = len(self.matrices[0])
        for index, matrix in enumerate(self.matrices):
            validate_finite_complex_matrix(
                matrix,
                f"matrices[{index}]",
                row_count=dimension,
                column_count=dimension,
            )

    @property
    def dimension(self) -> int:
        """Return the inferred nonempty or explicitly retained empty dimension."""
        if self.matrices:
            return len(self.matrices[0])
        assert self._empty_dimension is not None
        return self._empty_dimension

    @classmethod
    def empty(
        cls,
        dimension: int,
        tolerance: BatchedHermitianSpectrumTolerance,
        /,
    ) -> BatchedHermitianSpectrumDefinition:
        """Creates an empty definition with an explicit spectrum width.

        Parameters
        ----------
        dimension
            Positive integral matrix dimension.
        tolerance
            Nominal Hermiticity policy.

        Returns
        -------
        BatchedHermitianSpectrumDefinition
            Empty immutable definition.
        """
        return cls((), tolerance, dimension)

    @property
    def batch_size(self) -> int:
        """Returns the matrix count.

        Returns
        -------
        int
            Nonnegative batch size.
        """
        return len(self.matrices)


@final
@dataclass(frozen=True, slots=True)
class HermitianSpectrumMatrixDiagnostics:
    """Stores input and representative diagnostics for one matrix.

    Parameters
    ----------
    input_hermiticity_maximum_absolute_defect
        Maximum absolute difference from the conjugate transpose.
    representative_adjustment_maximum_absolute_defect
        Maximum absolute representative adjustment.
    input_matrix_maximum_absolute_entry
        Scale used by the relative cutoff.
    applied_hermiticity_cutoff
        Absolute-plus-relative cutoff applied to this matrix.
    """

    input_hermiticity_maximum_absolute_defect: float
    representative_adjustment_maximum_absolute_defect: float
    input_matrix_maximum_absolute_entry: float
    applied_hermiticity_cutoff: float

    def __post_init__(self) -> None:
        """Validate canonical finite nonnegative diagnostic values."""
        exact_nonnegative_float(
            self.input_hermiticity_maximum_absolute_defect,
            "input_hermiticity_maximum_absolute_defect",
        )
        exact_nonnegative_float(
            self.representative_adjustment_maximum_absolute_defect,
            "representative_adjustment_maximum_absolute_defect",
        )
        exact_nonnegative_float(
            self.input_matrix_maximum_absolute_entry,
            "input_matrix_maximum_absolute_entry",
        )
        exact_nonnegative_float(
            self.applied_hermiticity_cutoff,
            "applied_hermiticity_cutoff",
        )


@final
@dataclass(frozen=True, slots=True)
class BatchedHermitianSpectrumDiagnostics:
    """Stores componentwise diagnostic maxima for a nonempty batch.

    Parameters
    ----------
    maximum_input_hermiticity_absolute_defect
        Maximum input Hermiticity defect.
    maximum_representative_adjustment_absolute_defect
        Maximum representative adjustment.
    maximum_input_matrix_absolute_entry
        Maximum input matrix scale.
    maximum_applied_hermiticity_cutoff
        Maximum applied cutoff.
    """

    maximum_input_hermiticity_absolute_defect: float
    maximum_representative_adjustment_absolute_defect: float
    maximum_input_matrix_absolute_entry: float
    maximum_applied_hermiticity_cutoff: float

    def __post_init__(self) -> None:
        """Validate canonical finite nonnegative aggregate values."""
        exact_nonnegative_float(
            self.maximum_input_hermiticity_absolute_defect,
            "maximum_input_hermiticity_absolute_defect",
        )
        exact_nonnegative_float(
            self.maximum_representative_adjustment_absolute_defect,
            "maximum_representative_adjustment_absolute_defect",
        )
        exact_nonnegative_float(
            self.maximum_input_matrix_absolute_entry,
            "maximum_input_matrix_absolute_entry",
        )
        exact_nonnegative_float(
            self.maximum_applied_hermiticity_cutoff,
            "maximum_applied_hermiticity_cutoff",
        )


def _aggregate(
    records: tuple[HermitianSpectrumMatrixDiagnostics, ...],
    /,
) -> BatchedHermitianSpectrumDiagnostics | None:
    """Constructs aggregate maxima only when observations exist.

    Parameters
    ----------
    records
        Ordered per-matrix diagnostics.

    Returns
    -------
    BatchedHermitianSpectrumDiagnostics | None
        Componentwise maxima, or ``None`` for an empty batch.
    """
    if not records:
        # Absence distinguishes an undefined maximum over no observations from an
        # observed maximum whose numerical value happens to be zero.
        return None
    return BatchedHermitianSpectrumDiagnostics(
        max(record.input_hermiticity_maximum_absolute_defect for record in records),
        max(
            record.representative_adjustment_maximum_absolute_defect
            for record in records
        ),
        max(record.input_matrix_maximum_absolute_entry for record in records),
        max(record.applied_hermiticity_cutoff for record in records),
    )


@final
@dataclass(frozen=True, slots=True)
class BatchedHermitianSpectrumResult:
    """Stores complete spectra and diagnostics for one matrix batch.

    Parameters
    ----------
    dimension
        Positive width of every spectrum.
    tolerance
        Applied Hermiticity policy.
    eigenvalues
        One complete ascending spectrum per input matrix.
    matrix_diagnostics
        One evidence record per input matrix.
    diagnostics
        Aggregate maxima, or ``None`` for an empty batch.
    backend_identifier, backend_version
        API-level solver metadata.
    """

    dimension: int
    tolerance: BatchedHermitianSpectrumTolerance
    eigenvalues: HermitianSpectra
    matrix_diagnostics: tuple[HermitianSpectrumMatrixDiagnostics, ...]
    diagnostics: BatchedHermitianSpectrumDiagnostics | None
    backend_identifier: str
    backend_version: str

    def __post_init__(self) -> None:  # noqa: C901
        """Validate canonical structure and reconstructable relations.

        Raises
        ------
        TypeError
            If a nominal or scalar type is unsupported.
        ValueError
            If shape, diagnostics, ordering, or metadata are inconsistent.
        FloatingPointError
            If reconstructed cutoff arithmetic becomes nonfinite.
        """
        exact_positive_integer(self.dimension, "dimension")
        if type(self.tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError("tolerance must be BatchedHermitianSpectrumTolerance")
        if type(self.eigenvalues) is not tuple:
            raise TypeError("eigenvalues must be an exact tuple")
        for index, spectrum in enumerate(self.eigenvalues):
            validate_finite_float_tuple(spectrum, f"eigenvalues[{index}]")
            if len(spectrum) != self.dimension:
                raise ValueError(f"eigenvalues[{index}] length must equal dimension")
            if any(
                left > right
                for left, right in zip(spectrum, spectrum[1:], strict=False)
            ):
                raise ValueError(f"eigenvalues[{index}] must be in ascending order")

        if type(self.matrix_diagnostics) is not tuple:
            raise TypeError("matrix_diagnostics must be an exact tuple")
        if len(self.matrix_diagnostics) != len(self.eigenvalues):
            raise ValueError("matrix_diagnostics length must equal batch_size")
        if any(
            type(record) is not HermitianSpectrumMatrixDiagnostics
            for record in self.matrix_diagnostics
        ):
            raise TypeError(
                "matrix_diagnostics entries must be HermitianSpectrumMatrixDiagnostics"
            )
        for record in self.matrix_diagnostics:
            # Reconstructing the cutoff prevents independently created result records
            # from pairing diagnostics with a different tolerance policy.
            cutoff = (
                self.tolerance.hermiticity_absolute
                + self.tolerance.hermiticity_relative
                * record.input_matrix_maximum_absolute_entry
            )
            if not math.isfinite(cutoff):
                raise FloatingPointError("result Hermiticity cutoff became nonfinite")
            if record.applied_hermiticity_cutoff != cutoff:
                raise ValueError("matrix diagnostic Hermiticity cutoff is inconsistent")
            if record.input_hermiticity_maximum_absolute_defect > cutoff:
                raise ValueError(
                    "matrix diagnostic Hermiticity defect exceeds its cutoff"
                )
        if self.diagnostics != _aggregate(self.matrix_diagnostics):
            raise ValueError(
                "diagnostics must be absent for an empty batch and equal aggregate "
                "maxima otherwise"
            )

        if type(self.backend_identifier) is not str:
            raise TypeError("backend_identifier must be an exact string")
        if not self.backend_identifier:
            raise ValueError("backend_identifier must be nonempty")
        if type(self.backend_version) is not str:
            raise TypeError("backend_version must be an exact string")
        if not self.backend_version:
            raise ValueError("backend_version must be nonempty")

    @property
    def batch_size(self) -> int:
        """Returns the evaluated matrix count.

        Returns
        -------
        int
            Nonnegative batch size.
        """
        return len(self.eigenvalues)


class BatchedHermitianSpectrumSolver(ABC):
    """Defines shared validation and diagnostics for nominal solvers.

    Notes
    -----
    Empty definitions return without invoking a numerical backend. Nonempty definitions
    are rejected as a whole before backend invocation when any matrix exceeds its own
    Hermiticity cutoff.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def _backend_identifier(self) -> str:
        """Returns the API-level backend identity.

        Returns
        -------
        str
            Nonempty backend identifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _backend_version(self) -> str:
        """Returns the backend package version.

        Returns
        -------
        str
            Nonempty version string.
        """
        raise NotImplementedError

    @final
    def solve(
        self,
        definition: BatchedHermitianSpectrumDefinition,
        /,
    ) -> BatchedHermitianSpectrumResult:
        """Computes complete spectra for one normalized matrix batch.

        Parameters
        ----------
        definition
            Exact nominal batch and Hermiticity policy.

        Returns
        -------
        BatchedHermitianSpectrumResult
            Spectra and diagnostics. Empty definitions skip the numerical backend.

        Raises
        ------
        TypeError
            If definition or backend output types are unsupported.
        ValueError
            If an input exceeds its cutoff or output structure is inconsistent.
        FloatingPointError
            If shared or backend arithmetic becomes nonfinite.
        RuntimeError
            If the backend eigensolver does not converge.
        """
        if type(definition) is not BatchedHermitianSpectrumDefinition:
            raise TypeError("definition must be BatchedHermitianSpectrumDefinition")
        identifier = self._backend_identifier
        version = self._backend_version
        if not definition.matrices:
            # There is no numerical problem to solve. Short-circuiting avoids coupling
            # the mathematical contract to a backend's empty-stack conventions.
            return BatchedHermitianSpectrumResult(
                definition.dimension,
                definition.tolerance,
                (),
                (),
                None,
                identifier,
                version,
            )

        representatives: list[HermitianSpectrumMatrix] = []
        diagnostics: list[HermitianSpectrumMatrixDiagnostics] = []
        tolerance = definition.tolerance
        for index, matrix in enumerate(definition.matrices):
            # Each matrix receives its own relative scale; one large matrix cannot
            # relax the Hermiticity cutoff for another matrix in the same batch.
            scale = max(abs(entry) for row in matrix for entry in row)
            if not math.isfinite(scale):
                raise FloatingPointError("matrix-entry magnitude became nonfinite")
            defect = _maximum_difference(matrix, _conjugate_transpose(matrix))
            cutoff = (
                tolerance.hermiticity_absolute + tolerance.hermiticity_relative * scale
            )
            if not math.isfinite(cutoff):
                raise FloatingPointError("applied Hermiticity cutoff became nonfinite")
            # Equality satisfies the caller's declared closed acceptance bound.
            if defect > cutoff:
                raise ValueError(
                    f"matrices[{index}] exceeds the declared Hermiticity tolerance"
                )
            representative = _representative(matrix)
            representatives.append(representative)
            diagnostics.append(
                HermitianSpectrumMatrixDiagnostics(
                    defect,
                    _maximum_difference(representative, matrix),
                    scale,
                    cutoff,
                )
            )

        records = tuple(diagnostics)
        eigenvalues = _spectra(
            self._eigvalsh_validated(tuple(representatives)),
            definition.dimension,
        )
        if len(eigenvalues) != definition.batch_size:
            raise ValueError("backend spectrum count must equal batch size")
        return BatchedHermitianSpectrumResult(
            definition.dimension,
            tolerance,
            eigenvalues,
            records,
            _aggregate(records),
            identifier,
            version,
        )

    @abstractmethod
    def _eigvalsh_validated(
        self,
        representatives: HermitianSpectrumMatrices,
        /,
    ) -> HermitianSpectra:
        """Returns spectra from one nonempty batched backend operation.

        Parameters
        ----------
        representatives
            Nonempty same-dimension exactly Hermitian matrices.

        Returns
        -------
        HermitianSpectra
            Immutable batch of complete built-in spectra.
        """
        raise NotImplementedError


__all__ = [
    "BatchedHermitianSpectrumDefinition",
    "BatchedHermitianSpectrumDiagnostics",
    "BatchedHermitianSpectrumResult",
    "BatchedHermitianSpectrumSolver",
    "BatchedHermitianSpectrumTolerance",
    "HermitianSpectra",
    "HermitianSpectrumMatrices",
    "HermitianSpectrumMatrix",
    "HermitianSpectrumMatrixDiagnostics",
]
