"""Complete spectra of finite dense complex Hermitian matrix batches.

The dependency-free contract accepts Python or array-like nested iterables and
normalizes them to immutable built-in values. For each finite square matrix ``A``, the
solver measures represented-input Hermiticity, constructs
``H = (A + A.conjugate().T) / 2``, and returns every eigenvalue of ``H`` in ascending
order with algebraic multiplicity. The capability returns no eigenvectors, projectors,
gap decisions, or scientific policy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields
from numbers import Complex, Integral, Real
from typing import cast, final

from projectkoios.numerics.typing import FiniteComplexMatrix

type HermitianSpectrumMatrix = FiniteComplexMatrix
type HermitianSpectrumMatrices = tuple[HermitianSpectrumMatrix, ...]
type HermitianSpectra = tuple[tuple[float, ...], ...]


def _items(value: object, name: str, /) -> tuple[object, ...]:
    """Materializes a non-string iterable exactly once.

    Parameters
    ----------
    value
        Candidate iterable.
    name
        Field name used in failure messages.

    Returns
    -------
    tuple[object, ...]
        Materialized values in iteration order.

    Raises
    ------
    TypeError
        If the value is string-like or not iterable.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a non-string iterable")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error


def _real(
    value: object,
    name: str,
    /,
    *,
    nonnegative: bool = False,
    relative: bool = False,
) -> float:
    """Normalizes one finite real scalar.

    Parameters
    ----------
    value
        Candidate scalar.
    name
        Field name used in failure messages.
    nonnegative
        Whether negative values are invalid.
    relative
        Whether the normalized value must also be less than one.

    Returns
    -------
    float
        Finite built-in value.

    Raises
    ------
    TypeError
        If the value is Boolean or not real.
    ValueError
        If finiteness or requested bounds fail.
    """
    # ``bool`` implements Python's integral protocols, but treating truth values as
    # numerical tolerances would conceal a likely caller error.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if nonnegative and normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if relative and normalized >= 1.0:
        raise ValueError(f"{name} must be less than 1.0")
    return normalized


def _dimension(value: object, /) -> int:
    """Normalizes a positive integral matrix dimension.

    Parameters
    ----------
    value
        Candidate dimension.

    Returns
    -------
    int
        Positive built-in dimension.

    Raises
    ------
    TypeError
        If the value is Boolean or not integral.
    ValueError
        If the value is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("dimension must be an integral scalar")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("dimension must be positive")
    return normalized


def _matrix(
    value: object,
    name: str,
    /,
    *,
    dimension: int | None = None,
) -> HermitianSpectrumMatrix:
    """Normalizes a nonempty finite square numeric matrix.

    Parameters
    ----------
    value
        Candidate nested iterable.
    name
        Field name used in failure messages.
    dimension
        Required dimension, or ``None`` to infer it.

    Returns
    -------
    HermitianSpectrumMatrix
        Immutable built-in complex matrix.

    Raises
    ------
    TypeError
        If containers or numeric entries are unsupported.
    ValueError
        If shape or finiteness invariants fail.
    """
    rows = _items(value, name)
    inferred = len(rows)
    if inferred == 0:
        raise ValueError(f"{name} must be nonempty")
    if dimension is not None and inferred != dimension:
        raise ValueError(f"{name} dimension is inconsistent")
    normalized: list[tuple[complex, ...]] = []
    for row_index, row in enumerate(rows):
        entries = _items(row, f"{name}[{row_index}]")
        if len(entries) != inferred:
            raise ValueError(f"{name} must be square")
        normalized_row: list[complex] = []
        for entry in entries:
            # Normalization provides one immutable complex representation without
            # importing an optional array library into the contract module.
            if isinstance(entry, bool) or not isinstance(entry, Complex):
                raise TypeError(
                    f"{name} entries must be real or complex numeric scalars"
                )
            number = complex(entry)
            if not math.isfinite(number.real) or not math.isfinite(number.imag):
                raise ValueError(f"{name} entries must be finite")
            normalized_row.append(number)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _batch(
    value: object,
    declared_dimension: object | None,
    /,
) -> tuple[int, HermitianSpectrumMatrices]:
    """Normalizes a matrix batch without redundant shape metadata.

    Parameters
    ----------
    value
        Candidate matrix batch.
    declared_dimension
        Required for an empty batch and forbidden otherwise.

    Returns
    -------
    tuple[int, HermitianSpectrumMatrices]
        Matrix dimension and immutable normalized batch.

    Raises
    ------
    TypeError
        If containers or entries are unsupported.
    ValueError
        If shape invariants fail or dimension use is missing or redundant.
    """
    raw = _items(value, "matrices")
    if not raw:
        # An empty iterable carries no trailing matrix shape, so this is the only case
        # in which callers provide dimension metadata.
        if declared_dimension is None:
            raise ValueError("empty matrices require an explicit dimension")
        return _dimension(declared_dimension), ()
    if declared_dimension is not None:
        raise ValueError("dimension must be omitted for a nonempty batch")
    # The first matrix owns the inferred dimension; every later matrix is checked
    # against that value before any numerical backend can run.
    first = _matrix(raw[0], "matrices[0]")
    inferred = len(first)
    return inferred, (first,) + tuple(
        _matrix(matrix, f"matrices[{index}]", dimension=inferred)
        for index, matrix in enumerate(raw[1:], start=1)
    )


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


def _spectra(value: object, dimension: int, /) -> HermitianSpectra:
    """Normalizes complete finite ascending spectra.

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
    normalized: list[tuple[float, ...]] = []
    for index, spectrum in enumerate(_items(value, "eigenvalues")):
        entries = _items(spectrum, f"eigenvalues[{index}]")
        if len(entries) != dimension:
            raise ValueError(f"eigenvalues[{index}] length must equal dimension")
        numbers = tuple(
            _real(entry, f"eigenvalues[{index}] entries") for entry in entries
        )
        # Nondecreasing order retains repeated eigenvalues and therefore preserves
        # algebraic multiplicity without introducing a degeneracy policy.
        if any(left > right for left, right in zip(numbers, numbers[1:], strict=False)):
            raise ValueError(f"eigenvalues[{index}] must be in ascending order")
        normalized.append(numbers)
    return tuple(normalized)


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
        """Normalizes the finite nonnegative tolerance components.

        Raises
        ------
        TypeError
            If either component is not real.
        ValueError
            If bounds or finiteness fail.
        """
        object.__setattr__(
            self,
            "hermiticity_absolute",
            _real(self.hermiticity_absolute, "hermiticity_absolute", nonnegative=True),
        )
        object.__setattr__(
            self,
            "hermiticity_relative",
            _real(
                self.hermiticity_relative,
                "hermiticity_relative",
                nonnegative=True,
                relative=True,
            ),
        )


@final
@dataclass(frozen=True, slots=True, init=False)
class BatchedHermitianSpectrumDefinition:
    """Defines an immutable matrix batch and Hermiticity policy.

    Notes
    -----
    A nonempty batch infers its common square dimension. The :meth:`empty` constructor
    supplies dimension only when no matrix exists from which to infer it.
    """

    dimension: int
    matrices: HermitianSpectrumMatrices
    tolerance: BatchedHermitianSpectrumTolerance

    def __init__(
        self,
        matrices: object,
        tolerance: BatchedHermitianSpectrumTolerance,
        *,
        dimension: object | None = None,
    ) -> None:
        """Normalizes an inferred nonempty or explicitly shaped empty batch.

        Parameters
        ----------
        matrices
            Matrix batch. A nonempty batch infers its common square dimension.
        tolerance
            Nominal Hermiticity policy.
        dimension
            Positive empty-batch dimension, forbidden for nonempty batches.

        Raises
        ------
        TypeError
            If representation or tolerance types are unsupported.
        ValueError
            If shapes fail or dimension use is missing or redundant.
        """
        if type(tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError("tolerance must be BatchedHermitianSpectrumTolerance")
        normalized_dimension, normalized_matrices = _batch(matrices, dimension)
        object.__setattr__(self, "dimension", normalized_dimension)
        object.__setattr__(self, "matrices", normalized_matrices)
        object.__setattr__(self, "tolerance", tolerance)

    @classmethod
    def empty(
        cls,
        dimension: object,
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
        return cls((), tolerance, dimension=dimension)

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
        """Normalizes all fields to finite nonnegative floats.

        Raises
        ------
        TypeError
            If a field is not real.
        ValueError
            If a field is nonfinite or negative.
        """
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _real(getattr(self, field.name), field.name, nonnegative=True),
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
        """Normalizes all fields to finite nonnegative floats.

        Raises
        ------
        TypeError
            If a field is not real.
        ValueError
            If a field is nonfinite or negative.
        """
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _real(getattr(self, field.name), field.name, nonnegative=True),
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

    def __post_init__(self) -> None:
        """Normalizes structure and validates reconstructable relations.

        Raises
        ------
        TypeError
            If a nominal or scalar type is unsupported.
        ValueError
            If shape, diagnostics, ordering, or metadata are inconsistent.
        FloatingPointError
            If reconstructed cutoff arithmetic becomes nonfinite.
        """
        dimension = _dimension(self.dimension)
        if type(self.tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError("tolerance must be BatchedHermitianSpectrumTolerance")
        eigenvalues = _spectra(self.eigenvalues, dimension)
        raw_records = _items(self.matrix_diagnostics, "matrix_diagnostics")
        if len(raw_records) != len(eigenvalues):
            raise ValueError("matrix_diagnostics length must equal batch_size")
        if any(
            type(record) is not HermitianSpectrumMatrixDiagnostics
            for record in raw_records
        ):
            raise TypeError(
                "matrix_diagnostics entries must be HermitianSpectrumMatrixDiagnostics"
            )
        records = cast(tuple[HermitianSpectrumMatrixDiagnostics, ...], raw_records)
        for record in records:
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
        if self.diagnostics != _aggregate(records):
            raise ValueError(
                "diagnostics must be absent for an empty batch and equal aggregate "
                "maxima otherwise"
            )
        if not isinstance(self.backend_identifier, str):
            raise TypeError("backend_identifier must be a string")
        if not self.backend_identifier:
            raise ValueError("backend_identifier must be nonempty")
        if not isinstance(self.backend_version, str):
            raise TypeError("backend_version must be a string")
        if not self.backend_version:
            raise ValueError("backend_version must be nonempty")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "matrix_diagnostics", records)
        object.__setattr__(self, "backend_identifier", str(self.backend_identifier))
        object.__setattr__(self, "backend_version", str(self.backend_version))

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
    ) -> object:
        """Returns spectra from one nonempty batched backend operation.

        Parameters
        ----------
        representatives
            Nonempty same-dimension exactly Hermitian matrices.

        Returns
        -------
        object
            Array-like or iterable batch of complete spectra.
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
