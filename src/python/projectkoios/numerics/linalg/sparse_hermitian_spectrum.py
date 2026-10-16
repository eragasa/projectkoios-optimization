"""Partial spectra of finite sparse complex Hermitian matrices.

The dependency-free contract stores matrices in canonical compressed sparse row form,
qualifies represented-input Hermiticity, constructs an explicit sparse Hermitian
representative, and requests a deterministic subset of algebraic or magnitude-selected
eigenvalues. Public results contain eigenvalues and residual evidence but no
eigenvectors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from numbers import Complex, Integral, Real
from typing import cast, final

from .hermitian_spectrum import (
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrixDiagnostics,
)


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
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a non-string iterable")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error


def _integer(value: object, name: str, /, *, positive: bool = False) -> int:
    """Normalizes one integral scalar.

    Parameters
    ----------
    value
        Candidate scalar.
    name
        Field name used in failure messages.
    positive
        Whether zero and negative values are invalid.

    Returns
    -------
    int
        Built-in integral value.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integral scalar")
    normalized = int(value)
    if positive and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _optional_positive_integer(value: object, name: str, /) -> int | None:
    """Normalizes an optional positive integral scalar.

    Parameters
    ----------
    value
        Candidate scalar or ``None``.
    name
        Field name used in failure messages.

    Returns
    -------
    int | None
        Normalized value.
    """
    if value is None:
        return None
    return _integer(value, name, positive=True)


def _nonnegative_real(value: object, name: str, /) -> float:
    """Normalizes one finite nonnegative real scalar.

    Parameters
    ----------
    value
        Candidate scalar.
    name
        Field name used in failure messages.

    Returns
    -------
    float
        Finite built-in value.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


@final
@dataclass(frozen=True, slots=True, init=False)
class SparseComplexCsrMatrix:
    """Stores one finite canonical complex matrix in CSR form.

    Notes
    -----
    Column indices are strictly increasing within each row. Explicit zero values are
    omitted, duplicate coordinates are forbidden, and absent coordinates represent zero.
    """

    dimension: int
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    data: tuple[complex, ...]

    def __init__(
        self,
        dimension: object,
        indptr: object,
        indices: object,
        data: object,
    ) -> None:
        """Normalizes and validates one CSR matrix.

        Parameters
        ----------
        dimension
            Positive square matrix dimension.
        indptr
            Row pointer sequence of length ``dimension + 1``.
        indices
            Column index sequence.
        data
            Finite nonzero numeric values corresponding to ``indices``.

        Raises
        ------
        TypeError
            If containers, indices, or values use unsupported representations.
        ValueError
            If dimension, CSR structure, canonical ordering, or finiteness fails.
        """
        normalized_dimension = _integer(dimension, "dimension", positive=True)
        raw_indptr = _items(indptr, "indptr")
        raw_indices = _items(indices, "indices")
        raw_data = _items(data, "data")
        pointers = tuple(_integer(value, "indptr entry") for value in raw_indptr)
        columns = tuple(_integer(value, "indices entry") for value in raw_indices)
        values: list[complex] = []
        for value in raw_data:
            if isinstance(value, bool) or not isinstance(value, Complex):
                raise TypeError("data entries must be real or complex numeric scalars")
            number = complex(value)
            if not math.isfinite(number.real) or not math.isfinite(number.imag):
                raise ValueError("data entries must be finite")
            if number == 0.0:
                raise ValueError("canonical CSR data must omit explicit zeros")
            values.append(number)
        # Canonical CSR invariants make coordinate lookup deterministic and prevent
        # backend-dependent duplicate summation or index reordering.
        if len(pointers) != normalized_dimension + 1:
            raise ValueError("indptr length must equal dimension + 1")
        if not pointers or pointers[0] != 0:
            raise ValueError("indptr must start at zero")
        if any(
            left > right for left, right in zip(pointers, pointers[1:], strict=False)
        ):
            raise ValueError("indptr must be nondecreasing")
        if len(columns) != len(values) or pointers[-1] != len(values):
            raise ValueError("indices, data, and indptr terminal count must agree")
        for row in range(normalized_dimension):
            row_columns = columns[pointers[row] : pointers[row + 1]]
            if any(
                column < 0 or column >= normalized_dimension for column in row_columns
            ):
                raise ValueError("column index is outside matrix dimension")
            if any(
                left >= right
                for left, right in zip(row_columns, row_columns[1:], strict=False)
            ):
                raise ValueError(
                    "column indices must increase strictly within each row"
                )
        object.__setattr__(self, "dimension", normalized_dimension)
        object.__setattr__(self, "indptr", pointers)
        object.__setattr__(self, "indices", columns)
        object.__setattr__(self, "data", tuple(values))

    @property
    def nonzero_count(self) -> int:
        """Returns the number of stored nonzero entries.

        Returns
        -------
        int
            Stored nonzero count.
        """
        return len(self.data)


def _entries(matrix: SparseComplexCsrMatrix, /) -> dict[tuple[int, int], complex]:
    """Returns the stored coordinate map for one canonical CSR matrix.

    Parameters
    ----------
    matrix
        Canonical sparse matrix.

    Returns
    -------
    dict[tuple[int, int], complex]
        Stored nonzero values by coordinate.
    """
    return {
        (row, matrix.indices[position]): matrix.data[position]
        for row in range(matrix.dimension)
        for position in range(matrix.indptr[row], matrix.indptr[row + 1])
    }


def _csr_from_entries(
    dimension: int,
    entries: dict[tuple[int, int], complex],
    /,
) -> SparseComplexCsrMatrix:
    """Constructs canonical CSR storage from nonzero coordinates.

    Parameters
    ----------
    dimension
        Positive matrix dimension.
    entries
        Coordinate map whose zero values are ignored.

    Returns
    -------
    SparseComplexCsrMatrix
        Canonical sparse matrix.
    """
    # Row buckets preserve sparse O(n + nnz log nnz) construction rather than scanning
    # the full coordinate map once for every matrix row.
    rows: list[list[tuple[int, complex]]] = [[] for _ in range(dimension)]
    for (row, column), value in entries.items():
        if value != 0.0:
            rows[row].append((column, value))
    pointers = [0]
    columns: list[int] = []
    values: list[complex] = []
    for row_entries in rows:
        for column, value in sorted(row_entries, key=lambda item: item[0]):
            columns.append(column)
            values.append(value)
        pointers.append(len(values))
    return SparseComplexCsrMatrix(dimension, pointers, columns, values)


def _representative_and_diagnostics(
    matrix: SparseComplexCsrMatrix,
    tolerance: BatchedHermitianSpectrumTolerance,
    /,
) -> tuple[SparseComplexCsrMatrix, HermitianSpectrumMatrixDiagnostics]:
    """Constructs a sparse Hermitian representative and input diagnostics.

    Parameters
    ----------
    matrix
        Canonical finite sparse matrix.
    tolerance
        Represented-input Hermiticity tolerance.

    Returns
    -------
    tuple[SparseComplexCsrMatrix, HermitianSpectrumMatrixDiagnostics]
        Explicit representative and qualification evidence.
    """
    source = _entries(matrix)
    # The coordinate union includes missing transpose partners, so structurally
    # asymmetric sparsity contributes to the same defect as an explicit zero would.
    coordinates = set(source) | {(column, row) for row, column in source}
    scale = max((abs(value) for value in source.values()), default=0.0)
    defect = max(
        (
            abs(
                source.get((row, column), 0.0)
                - source.get((column, row), 0.0).conjugate()
            )
            for row, column in coordinates
        ),
        default=0.0,
    )
    cutoff = tolerance.hermiticity_absolute + tolerance.hermiticity_relative * scale
    if not math.isfinite(defect) or not math.isfinite(cutoff):
        raise FloatingPointError("sparse Hermiticity arithmetic became nonfinite")
    # Equality satisfies the closed represented-input acceptance boundary.
    if defect > cutoff:
        raise ValueError("matrix exceeds the declared Hermiticity tolerance")
    # Symmetrization operates only on the coordinate union and therefore preserves
    # sparsity except where a one-sided entry requires its conjugate partner.
    representative_entries = {
        (row, column): (
            source.get((row, column), 0.0) + source.get((column, row), 0.0).conjugate()
        )
        * 0.5
        for row, column in coordinates
    }
    if any(
        not math.isfinite(value.real) or not math.isfinite(value.imag)
        for value in representative_entries.values()
    ):
        raise FloatingPointError("sparse Hermitian symmetrization became nonfinite")
    if any(
        value != representative_entries.get((column, row), 0.0).conjugate()
        for (row, column), value in representative_entries.items()
    ):
        raise FloatingPointError(
            "constructed sparse representative is not exactly Hermitian"
        )
    adjustment = max(
        (
            abs(value - source.get(coordinate, 0.0))
            for coordinate, value in representative_entries.items()
        ),
        default=0.0,
    )
    if not math.isfinite(adjustment):
        raise FloatingPointError("sparse representative adjustment became nonfinite")
    representative = _csr_from_entries(matrix.dimension, representative_entries)
    return representative, HermitianSpectrumMatrixDiagnostics(
        defect,
        adjustment,
        scale,
        cutoff,
    )


class SparseHermitianSpectrumSelection(StrEnum):
    """Selects a partial Hermitian spectrum by algebraic or magnitude order.

    Attributes
    ----------
    SMALLEST_ALGEBRAIC
        Eigenvalues with the smallest algebraic values.
    LARGEST_ALGEBRAIC
        Eigenvalues with the largest algebraic values.
    SMALLEST_MAGNITUDE
        Eigenvalues with the smallest absolute values.
    LARGEST_MAGNITUDE
        Eigenvalues with the largest absolute values.
    """

    SMALLEST_ALGEBRAIC = "smallest-algebraic"
    LARGEST_ALGEBRAIC = "largest-algebraic"
    SMALLEST_MAGNITUDE = "smallest-magnitude"
    LARGEST_MAGNITUDE = "largest-magnitude"


@final
@dataclass(frozen=True, slots=True)
class SparseHermitianPartialSpectrumDefinition:
    """Defines one deterministic sparse partial-spectrum problem.

    Parameters
    ----------
    matrix
        Canonical sparse matrix.
    eigenvalue_count
        Requested count, strictly smaller than matrix dimension.
    selection
        Algebraic or magnitude selection rule.
    hermiticity_tolerance
        Represented-input Hermiticity policy.
    eigensolver_tolerance
        Nonnegative iterative convergence tolerance; zero selects backend precision.
    maximum_iterations
        Optional positive iteration limit.
    krylov_subspace_dimension
        Optional Krylov dimension strictly larger than the requested count and no larger
        than matrix dimension.
    random_seed
        Nonnegative deterministic seed for backend start-vector construction.
    """

    matrix: SparseComplexCsrMatrix
    eigenvalue_count: int
    selection: SparseHermitianSpectrumSelection
    hermiticity_tolerance: BatchedHermitianSpectrumTolerance
    eigensolver_tolerance: float = 0.0
    maximum_iterations: int | None = None
    krylov_subspace_dimension: int | None = None
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Normalizes controls and validates their relational invariants.

        Raises
        ------
        TypeError
            If nominal records, enums, or scalar controls use unsupported types.
        ValueError
            If counts, tolerances, iteration controls, or Krylov dimensions are invalid.
        """
        if type(self.matrix) is not SparseComplexCsrMatrix:
            raise TypeError("matrix must be SparseComplexCsrMatrix")
        count = _integer(self.eigenvalue_count, "eigenvalue_count", positive=True)
        if count >= self.matrix.dimension:
            raise ValueError("eigenvalue_count must be smaller than matrix dimension")
        if type(self.selection) is not SparseHermitianSpectrumSelection:
            raise TypeError("selection must be SparseHermitianSpectrumSelection")
        if type(self.hermiticity_tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError(
                "hermiticity_tolerance must be BatchedHermitianSpectrumTolerance"
            )
        tolerance = _nonnegative_real(
            self.eigensolver_tolerance,
            "eigensolver_tolerance",
        )
        iterations = _optional_positive_integer(
            self.maximum_iterations,
            "maximum_iterations",
        )
        krylov = _optional_positive_integer(
            self.krylov_subspace_dimension,
            "krylov_subspace_dimension",
        )
        # ARPACK requires a Krylov space larger than the requested invariant subspace,
        # while a value above the matrix dimension has no mathematical meaning.
        if krylov is not None and (krylov <= count or krylov > self.matrix.dimension):
            raise ValueError(
                "krylov_subspace_dimension must exceed eigenvalue_count and not exceed "
                "matrix dimension"
            )
        seed = _integer(self.random_seed, "random_seed")
        if seed < 0:
            raise ValueError("random_seed must be nonnegative")
        object.__setattr__(self, "eigenvalue_count", count)
        object.__setattr__(self, "eigensolver_tolerance", tolerance)
        object.__setattr__(self, "maximum_iterations", iterations)
        object.__setattr__(self, "krylov_subspace_dimension", krylov)
        object.__setattr__(self, "random_seed", seed)


@final
@dataclass(frozen=True, slots=True)
class SparseHermitianPartialSpectrumResult:
    """Stores selected sparse eigenvalues and convergence evidence.

    Parameters
    ----------
    dimension
        Matrix dimension.
    eigenvalue_count
        Requested and returned eigenvalue count.
    selection
        Applied selection rule.
    eigenvalues
        Selected eigenvalues in ascending algebraic order.
    residual_maximum_absolute_norm
        Maximum Euclidean residual norm across returned eigenpairs.
    matrix_diagnostics
        Represented-input and representative evidence.
    backend_identifier, backend_version
        API-level backend metadata.
    """

    dimension: int
    eigenvalue_count: int
    selection: SparseHermitianSpectrumSelection
    eigenvalues: tuple[float, ...]
    residual_maximum_absolute_norm: float
    matrix_diagnostics: HermitianSpectrumMatrixDiagnostics
    backend_identifier: str
    backend_version: str

    def __post_init__(self) -> None:
        """Validates result shape, ordering, evidence, and metadata.

        Raises
        ------
        TypeError
            If nominal records, enums, or eigenvalues use unsupported types.
        ValueError
            If dimensions, counts, ordering, finiteness, or metadata are inconsistent.
        """
        dimension = _integer(self.dimension, "dimension", positive=True)
        count = _integer(self.eigenvalue_count, "eigenvalue_count", positive=True)
        if count >= dimension:
            raise ValueError("eigenvalue_count must be smaller than dimension")
        if type(self.selection) is not SparseHermitianSpectrumSelection:
            raise TypeError("selection must be SparseHermitianSpectrumSelection")
        raw_values = _items(self.eigenvalues, "eigenvalues")
        normalized_values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("eigenvalues must contain real scalars")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("eigenvalues must be finite")
            normalized_values.append(number)
        values = tuple(normalized_values)
        if len(values) != count:
            raise ValueError("eigenvalues length must equal eigenvalue_count")
        # Algebraic ordering gives every selection rule one stable public convention,
        # including selections originally made by magnitude.
        if any(left > right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("eigenvalues must be in ascending algebraic order")
        residual = _nonnegative_real(
            self.residual_maximum_absolute_norm,
            "residual_maximum_absolute_norm",
        )
        if type(self.matrix_diagnostics) is not HermitianSpectrumMatrixDiagnostics:
            raise TypeError(
                "matrix_diagnostics must be HermitianSpectrumMatrixDiagnostics"
            )
        if not isinstance(self.backend_identifier, str) or not self.backend_identifier:
            raise ValueError("backend_identifier must be a nonempty string")
        if not isinstance(self.backend_version, str) or not self.backend_version:
            raise ValueError("backend_version must be a nonempty string")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "eigenvalue_count", count)
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "residual_maximum_absolute_norm", residual)


class SparseHermitianPartialSpectrumSolver(ABC):
    """Defines shared qualification for sparse partial-spectrum solvers."""

    __slots__ = ()

    @property
    @abstractmethod
    def _backend_identifier(self) -> str:
        """Returns the API-level backend identifier.

        Returns
        -------
        str
            Nonempty backend operation identifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _backend_version(self) -> str:
        """Returns the backend package version.

        Returns
        -------
        str
            Nonempty backend package version.
        """
        raise NotImplementedError

    @final
    def solve(
        self,
        definition: SparseHermitianPartialSpectrumDefinition,
        /,
    ) -> SparseHermitianPartialSpectrumResult:
        """Computes one selected partial spectrum with residual evidence.

        Parameters
        ----------
        definition
            Sparse matrix and explicit iterative selection controls.

        Returns
        -------
        SparseHermitianPartialSpectrumResult
            Selected ascending eigenvalues and maximum eigenpair residual.
        """
        if type(definition) is not SparseHermitianPartialSpectrumDefinition:
            raise TypeError(
                "definition must be SparseHermitianPartialSpectrumDefinition"
            )
        representative, diagnostics = _representative_and_diagnostics(
            definition.matrix,
            definition.hermiticity_tolerance,
        )
        eigenvalues, residual = self._eigsh_validated(representative, definition)
        values = tuple(float(value) for value in eigenvalues)
        return SparseHermitianPartialSpectrumResult(
            definition.matrix.dimension,
            definition.eigenvalue_count,
            definition.selection,
            values,
            residual,
            diagnostics,
            self._backend_identifier,
            self._backend_version,
        )

    @abstractmethod
    def _eigsh_validated(
        self,
        representative: SparseComplexCsrMatrix,
        definition: SparseHermitianPartialSpectrumDefinition,
        /,
    ) -> tuple[tuple[float, ...], float]:
        """Returns selected eigenvalues and the maximum residual norm.

        Parameters
        ----------
        representative
            Canonical represented-Hermitian sparse matrix.
        definition
            Validated selection and convergence controls.

        Returns
        -------
        tuple[tuple[float, ...], float]
            Ascending selected eigenvalues and maximum residual norm.
        """
        raise NotImplementedError


__all__ = [
    "SparseComplexCsrMatrix",
    "SparseHermitianPartialSpectrumDefinition",
    "SparseHermitianPartialSpectrumResult",
    "SparseHermitianPartialSpectrumSolver",
    "SparseHermitianSpectrumSelection",
]
