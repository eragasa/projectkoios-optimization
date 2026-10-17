"""Invariant spectral subspaces of finite dense complex Hermitian matrices.

The capability accepts one dependency-free complex matrix and a caller-supplied
half-open eigenvalue-index range. It measures the original matrix's Hermiticity defect,
constructs the explicit representative ``(A + A.conjugate().T) / 2``, validates a
backend-provided complete eigensystem, and returns every ascending eigenvalue plus the
orthogonal projector onto the selected columns.

Backend eigenvectors are transient validation evidence. They never enter the public
result, so this module makes no eigenvector-phase or degenerate-basis convention. The
boundary cutoff is a declared software policy rather than a backward-error certificate
or scientific gap criterion.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import final

from projectkoios.numerics.typing import FiniteComplexMatrix

type HermitianSpectralMatrix = FiniteComplexMatrix
type _ComplexSquareMatrix = FiniteComplexMatrix
type _HermitianSpectralSubspaceBackendOutput = tuple[
    tuple[float, ...],
    _ComplexSquareMatrix,
]


def _exact_nonnegative_float(value: object, name: str, /) -> float:
    """Validate one finite, nonnegative exact built-in float."""
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _exact_relative_float(value: object, name: str, /) -> float:
    """Validate one exact built-in float in the half-open interval [0, 1)."""
    normalized = _exact_nonnegative_float(value, name)
    if normalized >= 1.0:
        raise ValueError(f"{name} must be less than 1.0")
    return normalized


def _exact_positive_integer(value: object, name: str, /) -> int:
    """Validate a positive exact built-in integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _exact_index(value: object, name: str, /) -> int:
    """Validate a nonnegative exact built-in integer index."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _matrix(
    value: object,
    name: str,
    /,
    *,
    dimension: int | None = None,
    nonfinite_backend: bool = False,
) -> HermitianSpectralMatrix:
    """Validate one nonempty exact square tuple matrix of built-in complex values."""
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    inferred_dimension = len(value)
    if inferred_dimension == 0:
        raise ValueError(f"{name} must be nonempty")
    if dimension is not None and inferred_dimension != dimension:
        raise ValueError(f"{name} dimension is inconsistent")

    normalized: list[tuple[complex, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError(f"{name} rows must be exact tuples")
        if len(row) != inferred_dimension:
            raise ValueError(f"{name} must be square")
        normalized_row: list[complex] = []
        for entry in row:
            if type(entry) is not complex:
                raise TypeError(f"{name} entries must be exact built-in complex values")
            if not math.isfinite(entry.real) or not math.isfinite(entry.imag):
                if nonfinite_backend:
                    raise FloatingPointError(f"{name} entries must be finite")
                raise ValueError(f"{name} entries must be finite")
            normalized_row.append(entry)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _eigenvalues(
    value: object,
    dimension: int,
    /,
    *,
    nonfinite_backend: bool,
) -> tuple[float, ...]:
    """Validate a complete finite ascending exact built-in float spectrum."""
    if type(value) is not tuple:
        raise TypeError("eigenvalues must be an exact tuple")
    if len(value) != dimension:
        raise ValueError("eigenvalues length must equal dimension")
    normalized: list[float] = []
    for entry in value:
        if type(entry) is not float:
            raise TypeError("eigenvalues must contain exact built-in floats")
        if not math.isfinite(entry):
            if nonfinite_backend:
                raise FloatingPointError("eigenvalues must be finite")
            raise ValueError("eigenvalues must be finite")
        normalized.append(entry)
    if any(
        left > right for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("eigenvalues must be in ascending order")
    return tuple(normalized)


def _optional_gap(value: object, name: str, /) -> float | None:
    """Validate one optional finite nonnegative exact built-in boundary gap."""
    if value is None:
        return None
    return _exact_nonnegative_float(value, name)


def _nonempty_string(value: object, name: str, /) -> str:
    """Validate one exact nonempty built-in string."""
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _finite(value: float, message: str, /) -> float:
    """Reject one nonfinite shared-arithmetic result."""
    if not math.isfinite(value):
        raise FloatingPointError(message)
    return value


def _finite_complex(value: complex, message: str, /) -> complex:
    """Reject one nonfinite shared complex-arithmetic result."""
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError(message)
    return value


def _maximum_absolute_entry(matrix: _ComplexSquareMatrix, /) -> float:
    """Return the finite maximum absolute matrix entry."""
    maximum = 0.0
    for row in matrix:
        for entry in row:
            magnitude = _finite(abs(entry), "matrix-entry magnitude became nonfinite")
            maximum = max(maximum, magnitude)
    return maximum


def _maximum_absolute_difference(
    left: _ComplexSquareMatrix,
    right: _ComplexSquareMatrix,
    /,
) -> float:
    """Return the finite maximum absolute entrywise difference."""
    maximum = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        for left_entry, right_entry in zip(left_row, right_row, strict=True):
            difference = _finite_complex(
                left_entry - right_entry,
                "matrix difference became nonfinite",
            )
            maximum = max(
                maximum,
                _finite(
                    abs(difference), "matrix-difference magnitude became nonfinite"
                ),
            )
    return maximum


def _conjugate_transpose(matrix: _ComplexSquareMatrix, /) -> _ComplexSquareMatrix:
    """Return the conjugate transpose of a square tuple matrix."""
    dimension = len(matrix)
    return tuple(
        tuple(matrix[column][row].conjugate() for column in range(dimension))
        for row in range(dimension)
    )


def _hermiticity_defect(matrix: _ComplexSquareMatrix, /) -> float:
    """Return the maximum absolute defect from the conjugate transpose."""
    return _maximum_absolute_difference(matrix, _conjugate_transpose(matrix))


def _symmetrize(matrix: HermitianSpectralMatrix, /) -> HermitianSpectralMatrix:
    """Construct ``(A + A†) / 2`` with finite scalar checks."""
    dimension = len(matrix)
    rows: list[tuple[complex, ...]] = []
    for row in range(dimension):
        entries: list[complex] = []
        for column in range(dimension):
            value = _finite_complex(
                (matrix[row][column] + matrix[column][row].conjugate()) * 0.5,
                "Hermitian symmetrization became nonfinite",
            )
            entries.append(value)
        rows.append(tuple(entries))
    representative = tuple(rows)
    if representative != _conjugate_transpose(representative):
        raise FloatingPointError("constructed representative is not exactly Hermitian")
    return representative


def _stable_real_sum(values: tuple[float, ...], message: str, /) -> float:
    """Accumulate finite real terms with ``math.fsum`` and overflow translation."""
    try:
        value = math.fsum(values)
    except OverflowError as error:
        raise FloatingPointError(message) from error
    return _finite(value, message)


def _stable_complex_sum(values: list[complex], message: str, /) -> complex:
    """Accumulate finite complex terms with componentwise ``math.fsum``."""
    try:
        value = complex(
            math.fsum(item.real for item in values),
            math.fsum(item.imag for item in values),
        )
    except OverflowError as error:
        raise FloatingPointError(message) from error
    return _finite_complex(value, message)


def _matrix_product(
    left: _ComplexSquareMatrix,
    right: _ComplexSquareMatrix,
    /,
) -> _ComplexSquareMatrix:
    """Multiply two square matrices with deterministic componentwise stable sums."""
    dimension = len(left)
    rows: list[tuple[complex, ...]] = []
    for row in range(dimension):
        entries: list[complex] = []
        for column in range(dimension):
            products: list[complex] = []
            for inner in range(dimension):
                products.append(
                    _finite_complex(
                        left[row][inner] * right[inner][column],
                        "matrix product became nonfinite",
                    )
                )
            entries.append(_stable_complex_sum(products, "matrix sum became nonfinite"))
        rows.append(tuple(entries))
    return tuple(rows)


def _right_diagonal_product(
    matrix: _ComplexSquareMatrix,
    diagonal: tuple[float, ...],
    /,
) -> _ComplexSquareMatrix:
    """Return a matrix multiplied on the right by a real diagonal matrix."""
    return tuple(
        tuple(
            _finite_complex(
                entry * diagonal[column],
                "diagonal matrix product became nonfinite",
            )
            for column, entry in enumerate(row)
        )
        for row in matrix
    )


def _subtract(
    left: _ComplexSquareMatrix,
    right: _ComplexSquareMatrix,
    /,
) -> _ComplexSquareMatrix:
    """Subtract two square matrices with finite checks."""
    return tuple(
        tuple(
            _finite_complex(
                left_entry - right_entry,
                "matrix subtraction became nonfinite",
            )
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _identity(dimension: int, /) -> _ComplexSquareMatrix:
    """Return the exact complex identity matrix."""
    return tuple(
        tuple(
            1.0 + 0.0j if row == column else 0.0 + 0.0j for column in range(dimension)
        )
        for row in range(dimension)
    )


def _trace(matrix: _ComplexSquareMatrix, /) -> complex:
    """Return the componentwise stable finite trace."""
    return _stable_complex_sum(
        [matrix[index][index] for index in range(len(matrix))],
        "matrix trace became nonfinite",
    )


def _projector(
    eigenvectors: _ComplexSquareMatrix,
    start_index: int,
    stop_index: int,
    /,
) -> HermitianSpectralMatrix:
    """Construct the selected-column orthogonal-projector candidate."""
    dimension = len(eigenvectors)
    rows: list[tuple[complex, ...]] = []
    for row in range(dimension):
        entries: list[complex] = []
        for column in range(dimension):
            products = [
                _finite_complex(
                    eigenvectors[row][index] * eigenvectors[column][index].conjugate(),
                    "projector product became nonfinite",
                )
                for index in range(start_index, stop_index)
            ]
            entries.append(
                _stable_complex_sum(products, "projector sum became nonfinite")
            )
        rows.append(tuple(entries))
    return tuple(rows)


def _absolute_complex(value: complex, message: str, /) -> float:
    """Return one finite complex magnitude."""
    return _finite(abs(_finite_complex(value, message)), message)


def _validate_bound(value: float, bound: float, message: str, /) -> None:
    """Reject a diagnostic larger than its finite inclusive bound."""
    normalized_bound = _finite(bound, f"{message} bound became nonfinite")
    if value > normalized_bound:
        raise ValueError(message)


@final
@dataclass(frozen=True, slots=True)
class HermitianSpectralSubspaceTolerance:
    """Declare distinct input, boundary, projector, and spectral tolerances.

    Relative fields belong to ``[0, 1)``. All fields are numerical software controls;
    ``boundary_absolute`` and ``boundary_relative`` do not establish a physical gap or
    a certified backward-error separation.
    """

    hermiticity_absolute: float
    hermiticity_relative: float
    boundary_absolute: float
    boundary_relative: float
    projector_validation_absolute: float
    spectral_validation_absolute: float
    spectral_validation_relative: float

    def __post_init__(self) -> None:
        """Reject non-built-in, nonfinite, negative, or out-of-range controls."""
        object.__setattr__(
            self,
            "hermiticity_absolute",
            _exact_nonnegative_float(
                self.hermiticity_absolute,
                "hermiticity_absolute",
            ),
        )
        object.__setattr__(
            self,
            "hermiticity_relative",
            _exact_relative_float(
                self.hermiticity_relative,
                "hermiticity_relative",
            ),
        )
        object.__setattr__(
            self,
            "boundary_absolute",
            _exact_nonnegative_float(self.boundary_absolute, "boundary_absolute"),
        )
        object.__setattr__(
            self,
            "boundary_relative",
            _exact_relative_float(self.boundary_relative, "boundary_relative"),
        )
        object.__setattr__(
            self,
            "projector_validation_absolute",
            _exact_nonnegative_float(
                self.projector_validation_absolute,
                "projector_validation_absolute",
            ),
        )
        object.__setattr__(
            self,
            "spectral_validation_absolute",
            _exact_nonnegative_float(
                self.spectral_validation_absolute,
                "spectral_validation_absolute",
            ),
        )
        object.__setattr__(
            self,
            "spectral_validation_relative",
            _exact_relative_float(
                self.spectral_validation_relative,
                "spectral_validation_relative",
            ),
        )


@final
@dataclass(frozen=True, slots=True)
class HermitianSpectralSubspaceDefinition:
    """Define one matrix and one nonempty half-open selected index range."""

    matrix: HermitianSpectralMatrix
    start_index: int
    stop_index: int
    tolerance: HermitianSpectralSubspaceTolerance

    def __post_init__(self) -> None:
        """Validate dependency-free matrix, range, and tolerance invariants."""
        matrix = _matrix(self.matrix, "matrix")
        start_index = _exact_index(self.start_index, "start_index")
        stop_index = _exact_index(self.stop_index, "stop_index")
        if start_index >= stop_index:
            raise ValueError("start_index must be less than stop_index")
        if stop_index > len(matrix):
            raise ValueError("stop_index must not exceed matrix dimension")
        if type(self.tolerance) is not HermitianSpectralSubspaceTolerance:
            raise TypeError("tolerance must be HermitianSpectralSubspaceTolerance")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "start_index", start_index)
        object.__setattr__(self, "stop_index", stop_index)

    @property
    def dimension(self) -> int:
        """Return the positive matrix dimension."""
        return len(self.matrix)

    @property
    def selected_rank(self) -> int:
        """Return the positive number of selected ascending eigenvalue indices."""
        return self.stop_index - self.start_index


@final
@dataclass(frozen=True, slots=True)
class HermitianSpectralSubspaceDiagnostics:
    """Record shared finite defects, scales, and applied numerical cutoffs."""

    input_hermiticity_maximum_absolute_defect: float
    representative_adjustment_maximum_absolute_defect: float
    eigenvector_orthonormality_maximum_absolute_defect: float
    eigensystem_residual_maximum_absolute_defect: float
    representative_reconstruction_maximum_absolute_defect: float
    projector_hermiticity_maximum_absolute_defect: float
    projector_idempotence_maximum_absolute_defect: float
    projector_trace_absolute_defect: float
    projector_commutator_maximum_absolute_defect: float
    selected_spectral_trace_absolute_defect: float
    full_spectral_trace_absolute_defect: float
    input_matrix_maximum_absolute_entry: float
    representative_matrix_maximum_absolute_entry: float
    applied_hermiticity_cutoff: float
    spectral_diameter: float
    applied_boundary_cutoff: float
    applied_projector_validation_cutoff: float
    applied_spectral_validation_cutoff: float

    def __post_init__(self) -> None:
        """Require exact finite nonnegative built-in float diagnostics."""
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _exact_nonnegative_float(getattr(self, field.name), field.name),
            )


@final
@dataclass(frozen=True, slots=True)
class HermitianSpectralSubspaceResult:
    """Record a complete spectrum and one invariant selected projector.

    Backend metadata is a trusted API-level declaration. It is not authenticated and
    does not identify a linked BLAS/LAPACK implementation.
    """

    dimension: int
    start_index: int
    stop_index: int
    tolerance: HermitianSpectralSubspaceTolerance
    eigenvalues: tuple[float, ...]
    projector: HermitianSpectralMatrix
    lower_boundary_gap: float | None
    upper_boundary_gap: float | None
    diagnostics: HermitianSpectralSubspaceDiagnostics
    backend_identifier: str
    backend_version: str

    def __post_init__(self) -> None:
        """Validate local structure and finite-state result invariants."""
        dimension = _exact_positive_integer(self.dimension, "dimension")
        start_index = _exact_index(self.start_index, "start_index")
        stop_index = _exact_index(self.stop_index, "stop_index")
        if start_index >= stop_index:
            raise ValueError("start_index must be less than stop_index")
        if stop_index > dimension:
            raise ValueError("stop_index must not exceed dimension")
        if type(self.tolerance) is not HermitianSpectralSubspaceTolerance:
            raise TypeError("tolerance must be HermitianSpectralSubspaceTolerance")
        eigenvalues = _eigenvalues(
            self.eigenvalues,
            dimension,
            nonfinite_backend=False,
        )
        projector = _matrix(self.projector, "projector", dimension=dimension)
        lower_boundary_gap = _optional_gap(
            self.lower_boundary_gap,
            "lower_boundary_gap",
        )
        upper_boundary_gap = _optional_gap(
            self.upper_boundary_gap,
            "upper_boundary_gap",
        )
        if start_index == 0 and lower_boundary_gap is not None:
            raise ValueError("lower_boundary_gap must be None at the lower edge")
        if start_index > 0 and lower_boundary_gap is None:
            raise ValueError("lower_boundary_gap is required away from the lower edge")
        if stop_index == dimension and upper_boundary_gap is not None:
            raise ValueError("upper_boundary_gap must be None at the upper edge")
        if stop_index < dimension and upper_boundary_gap is None:
            raise ValueError("upper_boundary_gap is required away from the upper edge")
        if type(self.diagnostics) is not HermitianSpectralSubspaceDiagnostics:
            raise TypeError("diagnostics must be HermitianSpectralSubspaceDiagnostics")
        backend_identifier = _nonempty_string(
            self.backend_identifier,
            "backend_identifier",
        )
        backend_version = _nonempty_string(self.backend_version, "backend_version")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "start_index", start_index)
        object.__setattr__(self, "stop_index", stop_index)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "projector", projector)
        object.__setattr__(self, "lower_boundary_gap", lower_boundary_gap)
        object.__setattr__(self, "upper_boundary_gap", upper_boundary_gap)
        object.__setattr__(self, "backend_identifier", backend_identifier)
        object.__setattr__(self, "backend_version", backend_version)


class HermitianSpectralSubspaceSolver(ABC):
    """Nominal SDK boundary with shared complete-eigensystem validation."""

    __slots__ = ()

    @property
    @abstractmethod
    def _backend_identifier(self) -> str:
        """Return trusted API-level backend identity metadata."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _backend_version(self) -> str:
        """Return trusted API-level backend version metadata."""
        raise NotImplementedError

    @final
    def solve(
        self,
        definition: HermitianSpectralSubspaceDefinition,
        /,
    ) -> HermitianSpectralSubspaceResult:
        """Validate, decompose, correlate, and return one invariant subspace.

        The public template measures the original input before symmetrization, validates
        the complete ordered eigensystem, constructs the requested projector itself, and
        derives every diagnostic. Backends cannot self-attest to a selected projector or
        a quality diagnostic.
        """
        if type(definition) is not HermitianSpectralSubspaceDefinition:
            raise TypeError("definition must be HermitianSpectralSubspaceDefinition")

        matrix = definition.matrix
        dimension = definition.dimension
        tolerance = definition.tolerance
        input_scale = _maximum_absolute_entry(matrix)
        input_defect = _hermiticity_defect(matrix)
        hermiticity_cutoff = _finite(
            tolerance.hermiticity_absolute
            + tolerance.hermiticity_relative * input_scale,
            "applied Hermiticity cutoff became nonfinite",
        )
        if input_defect > hermiticity_cutoff:
            raise ValueError("matrix exceeds the declared Hermiticity tolerance")

        representative = _symmetrize(matrix)
        representative_adjustment = _maximum_absolute_difference(
            representative,
            matrix,
        )
        representative_scale = _maximum_absolute_entry(representative)

        backend_output = self._eigendecompose_validated(representative)
        if type(backend_output) is not tuple:
            raise TypeError("backend output must be an exact tuple")
        if len(backend_output) != 2:
            raise ValueError("backend output must contain eigenvalues and eigenvectors")
        eigenvalues = _eigenvalues(
            backend_output[0],
            dimension,
            nonfinite_backend=True,
        )
        eigenvectors = _matrix(
            backend_output[1],
            "backend eigenvectors",
            dimension=dimension,
            nonfinite_backend=True,
        )

        projector_cutoff = tolerance.projector_validation_absolute
        spectral_cutoff = _finite(
            tolerance.spectral_validation_absolute
            + tolerance.spectral_validation_relative * representative_scale,
            "applied spectral validation cutoff became nonfinite",
        )

        eigenvectors_adjoint = _conjugate_transpose(eigenvectors)
        orthonormality_defect = _maximum_absolute_entry(
            _subtract(
                _matrix_product(eigenvectors_adjoint, eigenvectors),
                _identity(dimension),
            )
        )
        representative_times_vectors = _matrix_product(
            representative,
            eigenvectors,
        )
        vectors_times_eigenvalues = _right_diagonal_product(
            eigenvectors,
            eigenvalues,
        )
        eigensystem_residual = _maximum_absolute_difference(
            representative_times_vectors,
            vectors_times_eigenvalues,
        )
        reconstruction = _matrix_product(
            vectors_times_eigenvalues,
            eigenvectors_adjoint,
        )
        reconstruction_defect = _maximum_absolute_difference(
            representative,
            reconstruction,
        )

        _validate_bound(
            orthonormality_defect,
            dimension * projector_cutoff,
            "backend eigenvectors violate the orthonormality bound",
        )
        _validate_bound(
            eigensystem_residual,
            dimension * spectral_cutoff,
            "backend eigensystem violates the residual bound",
        )
        _validate_bound(
            reconstruction_defect,
            2 * dimension * spectral_cutoff,
            "backend eigensystem violates the reconstruction bound",
        )

        spectral_diameter = _finite(
            eigenvalues[-1] - eigenvalues[0],
            "spectral diameter became nonfinite",
        )
        boundary_cutoff = _finite(
            tolerance.boundary_absolute
            + tolerance.boundary_relative * spectral_diameter,
            "applied boundary cutoff became nonfinite",
        )
        lower_boundary_gap = (
            None
            if definition.start_index == 0
            else _finite(
                eigenvalues[definition.start_index]
                - eigenvalues[definition.start_index - 1],
                "lower boundary gap became nonfinite",
            )
        )
        upper_boundary_gap = (
            None
            if definition.stop_index == dimension
            else _finite(
                eigenvalues[definition.stop_index]
                - eigenvalues[definition.stop_index - 1],
                "upper boundary gap became nonfinite",
            )
        )
        if lower_boundary_gap is not None and lower_boundary_gap <= boundary_cutoff:
            raise ValueError(
                "lower boundary does not pass the declared isolation policy"
            )
        if upper_boundary_gap is not None and upper_boundary_gap <= boundary_cutoff:
            raise ValueError(
                "upper boundary does not pass the declared isolation policy"
            )

        projector = _projector(
            eigenvectors,
            definition.start_index,
            definition.stop_index,
        )
        projector_adjoint = _conjugate_transpose(projector)
        projector_hermiticity_defect = _maximum_absolute_difference(
            projector,
            projector_adjoint,
        )
        projector_squared = _matrix_product(projector, projector)
        projector_idempotence_defect = _maximum_absolute_difference(
            projector_squared,
            projector,
        )
        projector_trace_defect = _absolute_complex(
            _trace(projector) - complex(definition.selected_rank, 0.0),
            "projector trace defect became nonfinite",
        )
        representative_projector = _matrix_product(representative, projector)
        projector_representative = _matrix_product(projector, representative)
        commutator_defect = _maximum_absolute_difference(
            representative_projector,
            projector_representative,
        )
        selected_eigenvalue_sum = _stable_real_sum(
            eigenvalues[definition.start_index : definition.stop_index],
            "selected eigenvalue sum became nonfinite",
        )
        selected_trace_defect = _absolute_complex(
            _trace(representative_projector) - complex(selected_eigenvalue_sum, 0.0),
            "selected spectral trace defect became nonfinite",
        )
        complete_eigenvalue_sum = _stable_real_sum(
            eigenvalues,
            "complete eigenvalue sum became nonfinite",
        )
        full_trace_defect = _absolute_complex(
            _trace(representative) - complex(complete_eigenvalue_sum, 0.0),
            "full spectral trace defect became nonfinite",
        )

        _validate_bound(
            projector_hermiticity_defect,
            2 * dimension * projector_cutoff,
            "projector violates the Hermiticity bound",
        )
        _validate_bound(
            projector_idempotence_defect,
            2 * dimension * projector_cutoff,
            "projector violates the idempotence bound",
        )
        _validate_bound(
            projector_trace_defect,
            definition.selected_rank * projector_cutoff,
            "projector violates the trace bound",
        )
        _validate_bound(
            commutator_defect,
            2 * dimension * spectral_cutoff,
            "projector violates the commutator bound",
        )
        _validate_bound(
            selected_trace_defect,
            definition.selected_rank * spectral_cutoff,
            "projector violates the selected spectral trace bound",
        )
        _validate_bound(
            full_trace_defect,
            dimension * spectral_cutoff,
            "eigensystem violates the full spectral trace bound",
        )

        diagnostics = HermitianSpectralSubspaceDiagnostics(
            input_defect,
            representative_adjustment,
            orthonormality_defect,
            eigensystem_residual,
            reconstruction_defect,
            projector_hermiticity_defect,
            projector_idempotence_defect,
            projector_trace_defect,
            commutator_defect,
            selected_trace_defect,
            full_trace_defect,
            input_scale,
            representative_scale,
            hermiticity_cutoff,
            spectral_diameter,
            boundary_cutoff,
            projector_cutoff,
            spectral_cutoff,
        )
        backend_identifier = _nonempty_string(
            self._backend_identifier,
            "backend_identifier",
        )
        backend_version = _nonempty_string(
            self._backend_version,
            "backend_version",
        )
        return HermitianSpectralSubspaceResult(
            dimension,
            definition.start_index,
            definition.stop_index,
            tolerance,
            eigenvalues,
            projector,
            lower_boundary_gap,
            upper_boundary_gap,
            diagnostics,
            backend_identifier,
            backend_version,
        )

    @abstractmethod
    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> _HermitianSpectralSubspaceBackendOutput:
        """Return a complete ascending eigensystem for a validated representative."""
        raise NotImplementedError


class HermitianSpectralSubspaceMethod(StrEnum):
    """Closed selection of owner-provided spectral-subspace implementations."""

    NUMPY_DENSE = "numpy-dense"


@final
class HermitianSpectralSubspaceSolverFactory:
    """Construct reviewed built-ins without importing optional backends eagerly."""

    __slots__ = ()

    @staticmethod
    def create(
        method: HermitianSpectralSubspaceMethod,
        /,
    ) -> HermitianSpectralSubspaceSolver:
        """Construct exactly the requested built-in nominal solver."""
        if type(method) is not HermitianSpectralSubspaceMethod:
            raise TypeError("method must be HermitianSpectralSubspaceMethod")
        if method is HermitianSpectralSubspaceMethod.NUMPY_DENSE:
            from .numpy.hermitian_spectral_subspace import (
                NumpyHermitianSpectralSubspaceSolver,
            )

            return NumpyHermitianSpectralSubspaceSolver()
        raise AssertionError("unhandled Hermitian spectral-subspace method")


__all__ = [
    "HermitianSpectralMatrix",
    "HermitianSpectralSubspaceDefinition",
    "HermitianSpectralSubspaceDiagnostics",
    "HermitianSpectralSubspaceMethod",
    "HermitianSpectralSubspaceResult",
    "HermitianSpectralSubspaceSolver",
    "HermitianSpectralSubspaceSolverFactory",
    "HermitianSpectralSubspaceTolerance",
]
