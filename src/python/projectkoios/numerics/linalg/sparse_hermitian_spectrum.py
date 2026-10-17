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
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from projectkoios.numerics import validation as numerical_validation

from .hermitian_spectrum import (
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrixDiagnostics,
)


@final
@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:  # noqa: C901
        """Validate exact canonical CSR storage without replacing fields."""
        dimension = numerical_validation.exact_positive_integer(
            self.dimension,
            "dimension",
        )
        if type(self.indptr) is not tuple:
            raise TypeError("indptr must be an exact tuple")
        if type(self.indices) is not tuple:
            raise TypeError("indices must be an exact tuple")
        if type(self.data) is not tuple:
            raise TypeError("data must be an exact tuple")
        for pointer in self.indptr:
            if type(pointer) is not int:
                raise TypeError("indptr entries must be exact built-in integers")
        for column in self.indices:
            if type(column) is not int:
                raise TypeError("indices entries must be exact built-in integers")
        for data_value in self.data:
            if type(data_value) is not complex:
                raise TypeError("data entries must be exact built-in complex values")
            if not math.isfinite(data_value.real) or not math.isfinite(data_value.imag):
                raise ValueError("data entries must be finite")
            if data_value == 0.0:
                raise ValueError("canonical CSR data must omit explicit zeros")
        # Canonical CSR invariants make coordinate lookup deterministic and prevent
        # backend-dependent duplicate summation or index reordering.
        if len(self.indptr) != dimension + 1:
            raise ValueError("indptr length must equal dimension + 1")
        if not self.indptr or self.indptr[0] != 0:
            raise ValueError("indptr must start at zero")
        if any(
            left > right
            for left, right in zip(self.indptr, self.indptr[1:], strict=False)
        ):
            raise ValueError("indptr must be nondecreasing")
        if len(self.indices) != len(self.data) or self.indptr[-1] != len(self.data):
            raise ValueError("indices, data, and indptr terminal count must agree")
        for row in range(dimension):
            row_columns = self.indices[self.indptr[row] : self.indptr[row + 1]]
            if any(column < 0 or column >= dimension for column in row_columns):
                raise ValueError("column index is outside matrix dimension")
            if any(
                left >= right
                for left, right in zip(row_columns, row_columns[1:], strict=False)
            ):
                raise ValueError(
                    "column indices must increase strictly within each row"
                )

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
    return SparseComplexCsrMatrix(
        dimension,
        tuple(pointers),
        tuple(columns),
        tuple(values),
    )


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
        count = numerical_validation.exact_positive_integer(
            self.eigenvalue_count,
            "eigenvalue_count",
        )
        if count >= self.matrix.dimension:
            raise ValueError("eigenvalue_count must be smaller than matrix dimension")
        if type(self.selection) is not SparseHermitianSpectrumSelection:
            raise TypeError("selection must be SparseHermitianSpectrumSelection")
        if type(self.hermiticity_tolerance) is not BatchedHermitianSpectrumTolerance:
            raise TypeError(
                "hermiticity_tolerance must be BatchedHermitianSpectrumTolerance"
            )
        numerical_validation.exact_nonnegative_float(
            self.eigensolver_tolerance,
            "eigensolver_tolerance",
        )
        if self.maximum_iterations is not None:
            numerical_validation.exact_positive_integer(
                self.maximum_iterations,
                "maximum_iterations",
            )
        if self.krylov_subspace_dimension is not None:
            numerical_validation.exact_positive_integer(
                self.krylov_subspace_dimension,
                "krylov_subspace_dimension",
            )
        # ARPACK requires a Krylov space larger than the requested invariant subspace,
        # while a value above the matrix dimension has no mathematical meaning.
        if self.krylov_subspace_dimension is not None and (
            self.krylov_subspace_dimension <= count
            or self.krylov_subspace_dimension > self.matrix.dimension
        ):
            raise ValueError(
                "krylov_subspace_dimension must exceed eigenvalue_count and not exceed "
                "matrix dimension"
            )
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an exact built-in integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")


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
        dimension = numerical_validation.exact_positive_integer(
            self.dimension,
            "dimension",
        )
        count = numerical_validation.exact_positive_integer(
            self.eigenvalue_count,
            "eigenvalue_count",
        )
        if count >= dimension:
            raise ValueError("eigenvalue_count must be smaller than dimension")
        if type(self.selection) is not SparseHermitianSpectrumSelection:
            raise TypeError("selection must be SparseHermitianSpectrumSelection")
        if type(self.eigenvalues) is not tuple:
            raise TypeError("eigenvalues must be an exact tuple")
        for value in self.eigenvalues:
            numerical_validation.exact_finite_float(value, "eigenvalue")
        if len(self.eigenvalues) != count:
            raise ValueError("eigenvalues length must equal eigenvalue_count")
        # Algebraic ordering gives every selection rule one stable public convention,
        # including selections originally made by magnitude.
        if any(
            left > right
            for left, right in zip(
                self.eigenvalues,
                self.eigenvalues[1:],
                strict=False,
            )
        ):
            raise ValueError("eigenvalues must be in ascending algebraic order")
        numerical_validation.exact_nonnegative_float(
            self.residual_maximum_absolute_norm,
            "residual_maximum_absolute_norm",
        )
        if type(self.matrix_diagnostics) is not HermitianSpectrumMatrixDiagnostics:
            raise TypeError(
                "matrix_diagnostics must be HermitianSpectrumMatrixDiagnostics"
            )
        if type(self.backend_identifier) is not str or not self.backend_identifier:
            raise ValueError("backend_identifier must be a nonempty exact string")
        if type(self.backend_version) is not str or not self.backend_version:
            raise ValueError("backend_version must be a nonempty exact string")


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
