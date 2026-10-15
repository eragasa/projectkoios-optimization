"""Dependency-free contract for one complex-unitary right alignment.

For equally shaped finite complex matrices ``A`` and ``B``, this capability solves

``minimize ||A Q - B||_F over square Q satisfying Q.conjugate().T @ Q = I``.

The returned transformation therefore acts on the right of the source matrix ``A``.
A numerical-rank policy is applied to ``A.conjugate().T @ B``. When that cross matrix is
rank deficient, the optimal value and rank evidence remain available but no arbitrary
null-space completion is exposed as a full unitary transformation.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import final

type ComplexUnitaryProcrustesMatrix = tuple[tuple[complex, ...], ...]
type _BackendOutput = tuple[
    tuple[float, ...],
    ComplexUnitaryProcrustesMatrix,
    ComplexUnitaryProcrustesMatrix,
]


def _positive_integer(value: object, name: str, /) -> int:
    """Validate a positive exact built-in integer.

    Parameters
    ----------
    value
        Candidate value.
    name
        Public field name used in failure messages.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in integer.
    ValueError
        If ``value`` is not positive.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(value: object, name: str, /) -> float:
    """Validate a finite nonnegative exact built-in float.

    Parameters
    ----------
    value
        Candidate value.
    name
        Public field name used in failure messages.

    Returns
    -------
    float
        The validated float.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in float.
    ValueError
        If ``value`` is nonfinite or negative.
    """
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _relative_float(value: object, name: str, /) -> float:
    """Validate an exact relative tolerance in ``[0, 1)``.

    Parameters
    ----------
    value
        Candidate relative tolerance.
    name
        Public field name used in failure messages.

    Returns
    -------
    float
        The validated relative tolerance.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in float.
    ValueError
        If ``value`` is nonfinite or lies outside ``[0, 1)``.
    """
    normalized = _nonnegative_float(value, name)
    if normalized >= 1.0:
        raise ValueError(f"{name} must be less than 1.0")
    return normalized


def _matrix(
    value: object,
    name: str,
    /,
    *,
    row_count: int | None = None,
    column_count: int | None = None,
    nonfinite_backend: bool = False,
) -> ComplexUnitaryProcrustesMatrix:
    """Validate a nonempty finite complex tuple matrix.

    Parameters
    ----------
    value
        Candidate matrix.
    name
        Public field name used in failure messages.
    row_count
        Required row count, or ``None`` to infer it.
    column_count
        Required column count, or ``None`` to infer it.
    nonfinite_backend
        Whether nonfinite entries came from backend output and therefore represent
        failed arithmetic rather than malformed public input.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The validated immutable rectangular matrix.

    Raises
    ------
    TypeError
        If the container, rows, or entries do not use exact public runtime types.
    ValueError
        If the matrix is empty, ragged, shape-inconsistent, or has nonfinite public
        input entries.
    FloatingPointError
        If backend output contains a nonfinite entry.
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    if row_count is not None and len(value) != row_count:
        raise ValueError(f"{name} row count is inconsistent")
    inferred_columns: int | None = None
    normalized: list[tuple[complex, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError(f"{name} rows must be exact tuples")
        if not row:
            raise ValueError(f"{name} rows must be nonempty")
        if inferred_columns is None:
            inferred_columns = len(row)
        elif len(row) != inferred_columns:
            raise ValueError(f"{name} must be rectangular")
        if column_count is not None and len(row) != column_count:
            raise ValueError(f"{name} column count is inconsistent")
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


def _singular_values(
    value: object,
    dimension: int,
    /,
    *,
    nonfinite_backend: bool,
) -> tuple[float, ...]:
    """Validate a complete nonincreasing cross singular spectrum.

    Parameters
    ----------
    value
        Candidate singular-value tuple.
    dimension
        Required number of singular values.
    nonfinite_backend
        Whether a nonfinite value came from backend output.

    Returns
    -------
    tuple[float, ...]
        The validated nonnegative singular values.

    Raises
    ------
    TypeError
        If the tuple or its entries do not use exact public runtime types.
    ValueError
        If the length, sign, ordering, or public-input finiteness is invalid.
    FloatingPointError
        If backend output contains a nonfinite singular value.
    """
    if type(value) is not tuple:
        raise TypeError("cross_singular_values must be an exact tuple")
    if len(value) != dimension:
        raise ValueError("cross_singular_values length must equal column_count")
    normalized: list[float] = []
    for entry in value:
        if type(entry) is not float:
            raise TypeError("cross_singular_values must contain exact built-in floats")
        if not math.isfinite(entry):
            if nonfinite_backend:
                raise FloatingPointError("cross_singular_values must be finite")
            raise ValueError("cross_singular_values must be finite")
        if entry < 0.0:
            raise ValueError("cross_singular_values must be nonnegative")
        normalized.append(entry)
    if any(
        left < right for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("cross_singular_values must be nonincreasing")
    return tuple(normalized)


def _finite(value: float, message: str, /) -> float:
    """Require a finite shared real-arithmetic result.

    Parameters
    ----------
    value
        Arithmetic result to validate.
    message
        Failure message describing the operation.

    Returns
    -------
    float
        The finite value.

    Raises
    ------
    FloatingPointError
        If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise FloatingPointError(message)
    return value


def _finite_complex(value: complex, message: str, /) -> complex:
    """Require a finite shared complex-arithmetic result.

    Parameters
    ----------
    value
        Arithmetic result to validate.
    message
        Failure message describing the operation.

    Returns
    -------
    complex
        The finite value.

    Raises
    ------
    FloatingPointError
        If either component is not finite.
    """
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError(message)
    return value


def _stable_complex_sum(values: list[complex], message: str, /) -> complex:
    """Accumulate complex terms with deterministic componentwise sums.

    Parameters
    ----------
    values
        Finite complex terms in deterministic accumulation order.
    message
        Failure message describing the operation.

    Returns
    -------
    complex
        The componentwise ``math.fsum`` result.

    Raises
    ------
    FloatingPointError
        If either component overflows or becomes nonfinite.
    """
    try:
        result = complex(
            math.fsum(value.real for value in values),
            math.fsum(value.imag for value in values),
        )
    except OverflowError as error:
        raise FloatingPointError(message) from error
    return _finite_complex(result, message)


def _conjugate_transpose(
    matrix: ComplexUnitaryProcrustesMatrix,
    /,
) -> ComplexUnitaryProcrustesMatrix:
    """Return the conjugate transpose of a nonempty rectangular matrix.

    Parameters
    ----------
    matrix
        Validated matrix to transpose.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The conjugate-transposed matrix.
    """
    return tuple(
        tuple(matrix[row][column].conjugate() for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _matrix_product(
    left: ComplexUnitaryProcrustesMatrix,
    right: ComplexUnitaryProcrustesMatrix,
    /,
) -> ComplexUnitaryProcrustesMatrix:
    """Multiply compatible finite complex matrices.

    Parameters
    ----------
    left
        Validated left factor.
    right
        Validated right factor.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The matrix product, accumulated with componentwise ``math.fsum``.

    Raises
    ------
    ValueError
        If the inner dimensions disagree.
    FloatingPointError
        If a product or accumulated entry becomes nonfinite.
    """
    inner_dimension = len(left[0])
    if inner_dimension != len(right):
        raise ValueError("matrix-product dimensions are inconsistent")
    return tuple(
        tuple(
            _stable_complex_sum(
                [
                    _finite_complex(
                        left[row][inner] * right[inner][column],
                        "matrix product became nonfinite",
                    )
                    for inner in range(inner_dimension)
                ],
                "matrix sum became nonfinite",
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _right_diagonal_product(
    matrix: ComplexUnitaryProcrustesMatrix,
    diagonal: tuple[float, ...],
    /,
) -> ComplexUnitaryProcrustesMatrix:
    """Multiply a matrix on the right by a finite real diagonal.

    Parameters
    ----------
    matrix
        Validated matrix.
    diagonal
        Finite diagonal entries in column order.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The checked product with the implicit diagonal matrix.

    Raises
    ------
    FloatingPointError
        If any scaled entry becomes nonfinite.
    """
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


def _matrix_difference(
    left: ComplexUnitaryProcrustesMatrix,
    right: ComplexUnitaryProcrustesMatrix,
    /,
) -> ComplexUnitaryProcrustesMatrix:
    """Subtract equally shaped matrices with finite-arithmetic checks.

    Parameters
    ----------
    left
        Validated minuend.
    right
        Validated subtrahend.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The entrywise difference.

    Raises
    ------
    FloatingPointError
        If any difference becomes nonfinite.
    """
    return tuple(
        tuple(
            _finite_complex(
                left_entry - right_entry,
                "matrix difference became nonfinite",
            )
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _identity(dimension: int, /) -> ComplexUnitaryProcrustesMatrix:
    """Construct an exact complex identity matrix.

    Parameters
    ----------
    dimension
        Positive matrix dimension.

    Returns
    -------
    ComplexUnitaryProcrustesMatrix
        The square identity matrix.
    """
    return tuple(
        tuple(
            1.0 + 0.0j if row == column else 0.0 + 0.0j for column in range(dimension)
        )
        for row in range(dimension)
    )


def _maximum_absolute_entry(
    matrix: ComplexUnitaryProcrustesMatrix,
    /,
) -> float:
    """Return the largest complex-entry magnitude.

    Parameters
    ----------
    matrix
        Validated finite matrix.

    Returns
    -------
    float
        Maximum absolute entry.

    Raises
    ------
    FloatingPointError
        If magnitude evaluation becomes nonfinite.
    """
    maximum = 0.0
    for row in matrix:
        for entry in row:
            maximum = max(
                maximum,
                _finite(abs(entry), "matrix-entry magnitude became nonfinite"),
            )
    return maximum


def _maximum_absolute_difference(
    left: ComplexUnitaryProcrustesMatrix,
    right: ComplexUnitaryProcrustesMatrix,
    /,
) -> float:
    """Return the largest absolute entrywise difference.

    Parameters
    ----------
    left
        Validated left matrix.
    right
        Equally shaped validated right matrix.

    Returns
    -------
    float
        Maximum absolute difference.

    Raises
    ------
    FloatingPointError
        If subtraction or magnitude evaluation becomes nonfinite.
    """
    return _maximum_absolute_entry(_matrix_difference(left, right))


def _squared_frobenius_norm(
    matrix: ComplexUnitaryProcrustesMatrix,
    /,
) -> float:
    """Compute a finite deterministic squared Frobenius norm.

    Parameters
    ----------
    matrix
        Validated finite matrix.

    Returns
    -------
    float
        Row-major ``math.fsum`` of squared entry magnitudes.

    Raises
    ------
    FloatingPointError
        If a squared magnitude or accumulated sum becomes nonfinite.
    """
    terms: list[float] = []
    for row in matrix:
        for entry in row:
            magnitude = _finite(abs(entry), "matrix magnitude became nonfinite")
            terms.append(
                _finite(
                    magnitude * magnitude,
                    "squared Frobenius term became nonfinite",
                )
            )
    try:
        value = math.fsum(terms)
    except OverflowError as error:
        raise FloatingPointError("squared Frobenius norm became nonfinite") from error
    return _finite(value, "squared Frobenius norm became nonfinite")


def _optimal_residual_evidence(
    source_power: float,
    target_power: float,
    singular_values: tuple[float, ...],
    /,
) -> tuple[float, float]:
    """Reconstruct optimal squared-residual evidence from singular values.

    Parameters
    ----------
    source_power
        Squared Frobenius norm of the source matrix.
    target_power
        Squared Frobenius norm of the target matrix.
    singular_values
        Complete singular spectrum of the cross matrix.

    Returns
    -------
    tuple[float, float]
        Optimal squared residual followed by the applied squared-residual roundoff
        cutoff. Values within the cutoff are represented as exact zero.

    Raises
    ------
    FloatingPointError
        If a nuclear-norm contribution, cutoff, residual, or materially negative
        reconstructed residual becomes invalid.
    """
    twice_singular_values = tuple(2.0 * value for value in singular_values)
    if any(not math.isfinite(value) for value in twice_singular_values):
        raise FloatingPointError("cross nuclear-norm contribution became nonfinite")
    scale = max(source_power, target_power, *twice_singular_values)
    if scale == 0.0:
        return 0.0, 0.0
    normalized_terms = [source_power / scale, target_power / scale]
    normalized_terms.extend(-value / scale for value in twice_singular_values)
    normalized = math.fsum(normalized_terms)
    roundoff_factor = max(64.0, 8.0 * float(len(normalized_terms)))
    roundoff_bound = (
        roundoff_factor
        * sys.float_info.epsilon
        * math.fsum(abs(value) for value in normalized_terms)
    )
    applied_roundoff_cutoff = _finite(
        roundoff_bound * scale,
        "residual roundoff cutoff became nonfinite",
    )
    # Near an exact fit, nuclear-norm subtraction loses all small residual digits.
    # The result exposes this squared cutoff instead of hiding that resolution limit.
    if abs(normalized) <= roundoff_bound:
        normalized = 0.0
    elif normalized < 0.0:
        raise FloatingPointError(
            "reconstructed optimal squared residual is materially negative"
        )
    return (
        _finite(
            normalized * scale,
            "optimal squared residual became nonfinite",
        ),
        applied_roundoff_cutoff,
    )


def _rank_cutoff(
    tolerance: ComplexUnitaryProcrustesTolerance,
    largest_singular_value: float,
    /,
) -> float:
    """Compute the absolute-plus-relative numerical-rank cutoff.

    Parameters
    ----------
    tolerance
        Declared software tolerances.
    largest_singular_value
        Largest cross singular value.

    Returns
    -------
    float
        Finite cutoff used with a strict-greater-than rank rule.

    Raises
    ------
    FloatingPointError
        If cutoff arithmetic becomes nonfinite.
    """
    return _finite(
        tolerance.rank_absolute + tolerance.rank_relative * largest_singular_value,
        "applied numerical-rank cutoff became nonfinite",
    )


def _validation_bound(
    tolerance: ComplexUnitaryProcrustesTolerance,
    scale: float,
    count: int,
    /,
) -> float:
    """Compute a dimension-scaled backend-validation bound.

    Parameters
    ----------
    tolerance
        Declared software tolerances.
    scale
        Nonnegative scale for the relative component.
    count
        Positive multiplicative operation-count bound.

    Returns
    -------
    float
        Finite validation bound.

    Raises
    ------
    FloatingPointError
        If bound arithmetic becomes nonfinite.
    """
    return _finite(
        count * (tolerance.validation_absolute + tolerance.validation_relative * scale),
        "validation bound became nonfinite",
    )


def _nonempty_string(value: object, name: str, /) -> str:
    """Validate an exact nonempty built-in metadata string.

    Parameters
    ----------
    value
        Candidate metadata value.
    name
        Public field name used in failure messages.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    TypeError
        If ``value`` is not an exact built-in string.
    ValueError
        If ``value`` is empty.
    """
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


@final
@dataclass(frozen=True, slots=True)
class ComplexUnitaryProcrustesTolerance:
    """Declare numerical-rank and backend-validation software tolerances.

    Parameters
    ----------
    rank_absolute
        Absolute cross-singular-value cutoff component.
    rank_relative
        Relative component multiplied by the largest cross singular value.
    validation_absolute
        Absolute backend-validation bound component.
    validation_relative
        Relative backend-validation bound component.

    Notes
    -----
    Numerical-rank controls are software policy, not scientific overlap or
    identifiability thresholds.
    """

    rank_absolute: float = 0.0
    rank_relative: float = 1.0e-12
    validation_absolute: float = 1.0e-12
    validation_relative: float = 1.0e-10

    def __post_init__(self) -> None:
        """Validate and normalize every tolerance component.

        Raises
        ------
        TypeError
            If a component is not an exact built-in float.
        ValueError
            If a component is nonfinite, negative, out of range, or if either
            tolerance pair is identically zero.
        """
        rank_absolute = _nonnegative_float(self.rank_absolute, "rank_absolute")
        rank_relative = _relative_float(self.rank_relative, "rank_relative")
        validation_absolute = _nonnegative_float(
            self.validation_absolute,
            "validation_absolute",
        )
        validation_relative = _relative_float(
            self.validation_relative,
            "validation_relative",
        )
        if rank_absolute == 0.0 and rank_relative == 0.0:
            raise ValueError("at least one numerical-rank tolerance must be positive")
        if validation_absolute == 0.0 and validation_relative == 0.0:
            raise ValueError("at least one validation tolerance must be positive")
        object.__setattr__(self, "rank_absolute", rank_absolute)
        object.__setattr__(self, "rank_relative", rank_relative)
        object.__setattr__(self, "validation_absolute", validation_absolute)
        object.__setattr__(self, "validation_relative", validation_relative)


DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE = ComplexUnitaryProcrustesTolerance()


@final
@dataclass(frozen=True, slots=True)
class ComplexUnitaryProcrustesDefinition:
    """Define one source-to-target complex-unitary right-alignment problem.

    Parameters
    ----------
    source_matrix
        Nonempty finite matrix ``A`` to transform on the right.
    target_matrix
        Finite matrix ``B`` with the same shape as ``A``.
    tolerance
        Numerical-rank and backend-validation software controls.
    """

    source_matrix: ComplexUnitaryProcrustesMatrix
    target_matrix: ComplexUnitaryProcrustesMatrix
    tolerance: ComplexUnitaryProcrustesTolerance = (
        DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
    )

    def __post_init__(self) -> None:
        """Validate exact input types, finiteness, shape, and tolerance identity.

        Raises
        ------
        TypeError
            If the matrices or tolerance violate nominal public runtime types.
        ValueError
            If either matrix is empty, ragged, nonfinite, or shape-inconsistent.
        """
        source = _matrix(self.source_matrix, "source_matrix")
        target = _matrix(
            self.target_matrix,
            "target_matrix",
            row_count=len(source),
            column_count=len(source[0]),
        )
        if type(self.tolerance) is not ComplexUnitaryProcrustesTolerance:
            raise TypeError("tolerance must be ComplexUnitaryProcrustesTolerance")
        object.__setattr__(self, "source_matrix", source)
        object.__setattr__(self, "target_matrix", target)

    @property
    def row_count(self) -> int:
        """Return the shared positive row count.

        Returns
        -------
        int
            Number of source and target rows.
        """
        return len(self.source_matrix)

    @property
    def column_count(self) -> int:
        """Return the shared column count and transform dimension.

        Returns
        -------
        int
            Number of columns and dimension of the square right-unitary transform.
        """
        return len(self.source_matrix[0])


class ComplexUnitaryProcrustesStatus(StrEnum):
    """Describe whether rank evidence identifies a full transformation.

    Attributes
    ----------
    UNIQUE
        The cross matrix has full numerical rank and the polar unitary is unique under
        the declared software policy.
    RANK_DEFICIENT
        The cross matrix is numerically deficient and no full transform is returned.
    """

    UNIQUE = "unique"
    RANK_DEFICIENT = "rank-deficient"


@final
@dataclass(frozen=True, slots=True)
class ComplexUnitaryProcrustesResult:
    """Record an optimum, rank evidence, and optional unique transform.

    Parameters
    ----------
    row_count, column_count
        Shared input shape.
    tolerance
        Applied numerical software controls.
    status
        Full-rank or deficient transformation disposition.
    transformation
        Unique right-unitary transform at full numerical rank, otherwise ``None``.
    cross_singular_values
        Complete nonincreasing singular spectrum of ``A.conjugate().T @ B``.
    numerical_rank, applied_rank_cutoff
        Rank count and exact cutoff used to obtain it.
    source_squared_frobenius_norm, target_squared_frobenius_norm
        Input powers used in residual reconstruction.
    optimal_squared_residual, optimal_residual_frobenius_norm
        Reconstructed optimum and its nonnegative square root.
    backend_identifier, backend_version
        Trusted API-level backend metadata, not linked-library provenance.
    """

    row_count: int
    column_count: int
    tolerance: ComplexUnitaryProcrustesTolerance
    status: ComplexUnitaryProcrustesStatus
    transformation: ComplexUnitaryProcrustesMatrix | None
    cross_singular_values: tuple[float, ...]
    numerical_rank: int
    applied_rank_cutoff: float
    source_squared_frobenius_norm: float
    target_squared_frobenius_norm: float
    optimal_squared_residual: float
    optimal_residual_frobenius_norm: float
    backend_identifier: str
    backend_version: str

    def __post_init__(self) -> None:
        """Validate local structure and relational numerical evidence.

        Raises
        ------
        TypeError
            If a field violates its exact nominal runtime type.
        ValueError
            If shape, ordering, rank, cutoff, residual, status, transform, or metadata
            relations are inconsistent.
        FloatingPointError
            If reconstructing relational evidence becomes nonfinite.
        """
        row_count = _positive_integer(self.row_count, "row_count")
        column_count = _positive_integer(self.column_count, "column_count")
        if type(self.tolerance) is not ComplexUnitaryProcrustesTolerance:
            raise TypeError("tolerance must be ComplexUnitaryProcrustesTolerance")
        if type(self.status) is not ComplexUnitaryProcrustesStatus:
            raise TypeError("status must be ComplexUnitaryProcrustesStatus")
        singular_values = _singular_values(
            self.cross_singular_values,
            column_count,
            nonfinite_backend=False,
        )
        if type(self.numerical_rank) is not int:
            raise TypeError("numerical_rank must be an exact built-in integer")
        applied_rank_cutoff = _nonnegative_float(
            self.applied_rank_cutoff,
            "applied_rank_cutoff",
        )
        expected_cutoff = _rank_cutoff(self.tolerance, singular_values[0])
        if applied_rank_cutoff != expected_cutoff:
            raise ValueError("applied_rank_cutoff is inconsistent")
        expected_rank = sum(value > expected_cutoff for value in singular_values)
        if self.numerical_rank != expected_rank:
            raise ValueError("numerical_rank is inconsistent")
        if not 0 <= self.numerical_rank <= column_count:
            raise ValueError("numerical_rank lies outside the dimension bound")

        source_power = _nonnegative_float(
            self.source_squared_frobenius_norm,
            "source_squared_frobenius_norm",
        )
        target_power = _nonnegative_float(
            self.target_squared_frobenius_norm,
            "target_squared_frobenius_norm",
        )
        optimal_squared = _nonnegative_float(
            self.optimal_squared_residual,
            "optimal_squared_residual",
        )
        expected_squared, _ = _optimal_residual_evidence(
            source_power,
            target_power,
            singular_values,
        )
        if optimal_squared != expected_squared:
            raise ValueError("optimal_squared_residual is inconsistent")
        optimal_norm = _nonnegative_float(
            self.optimal_residual_frobenius_norm,
            "optimal_residual_frobenius_norm",
        )
        if optimal_norm != math.sqrt(optimal_squared):
            raise ValueError("optimal_residual_frobenius_norm is inconsistent")

        transformation = self.transformation
        if expected_rank == column_count:
            if self.status is not ComplexUnitaryProcrustesStatus.UNIQUE:
                raise ValueError("full numerical rank requires UNIQUE status")
            transformation = _matrix(
                transformation,
                "transformation",
                row_count=column_count,
                column_count=column_count,
            )
            identity = _identity(column_count)
            unitary_defect = _maximum_absolute_difference(
                _matrix_product(
                    _conjugate_transpose(transformation),
                    transformation,
                ),
                identity,
            )
            if unitary_defect > _validation_bound(
                self.tolerance,
                1.0,
                column_count,
            ):
                raise ValueError("transformation violates the unitary bound")
        else:
            if self.status is not ComplexUnitaryProcrustesStatus.RANK_DEFICIENT:
                raise ValueError("deficient numerical rank requires RANK_DEFICIENT")
            if transformation is not None:
                raise ValueError(
                    "rank-deficient result must not expose a full transformation"
                )

        backend_identifier = _nonempty_string(
            self.backend_identifier,
            "backend_identifier",
        )
        backend_version = _nonempty_string(self.backend_version, "backend_version")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "column_count", column_count)
        object.__setattr__(self, "transformation", transformation)
        object.__setattr__(self, "cross_singular_values", singular_values)
        object.__setattr__(self, "applied_rank_cutoff", applied_rank_cutoff)
        object.__setattr__(
            self,
            "source_squared_frobenius_norm",
            source_power,
        )
        object.__setattr__(
            self,
            "target_squared_frobenius_norm",
            target_power,
        )
        object.__setattr__(self, "optimal_squared_residual", optimal_squared)
        object.__setattr__(
            self,
            "optimal_residual_frobenius_norm",
            optimal_norm,
        )
        object.__setattr__(self, "backend_identifier", backend_identifier)
        object.__setattr__(self, "backend_version", backend_version)

    @property
    def applied_residual_roundoff_cutoff(self) -> float:
        """Return the squared-residual cancellation-noise cutoff.

        Returns
        -------
        float
            Applied cutoff in squared-residual units. A reconstructed optimum whose
            magnitude does not exceed this value is represented as zero.
        """
        _, cutoff = _optimal_residual_evidence(
            self.source_squared_frobenius_norm,
            self.target_squared_frobenius_norm,
            self.cross_singular_values,
        )
        return cutoff


class ComplexUnitaryProcrustesSolver(ABC):
    """Define the nominal solver boundary and shared validation template.

    Notes
    -----
    Subclasses implement only the complete square cross-matrix SVD and backend metadata.
    The final public method owns input identity, transformation direction, numerical
    rank, null-space withholding, residual evidence, and result construction.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def _backend_identifier(self) -> str:
        """Return trusted API-level backend identity metadata.

        Returns
        -------
        str
            Nonempty backend operation identifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _backend_version(self) -> str:
        """Return trusted API-level backend version metadata.

        Returns
        -------
        str
            Nonempty backend package version.
        """
        raise NotImplementedError

    @final
    def solve(
        self,
        definition: ComplexUnitaryProcrustesDefinition,
        /,
    ) -> ComplexUnitaryProcrustesResult:
        """Solve one right-alignment problem without null-space completion.

        Parameters
        ----------
        definition
            Exact nominal source, target, and numerical policy record.

        Returns
        -------
        ComplexUnitaryProcrustesResult
            Validated optimum, singular/rank evidence, and a transform only at full
            numerical rank.

        Raises
        ------
        TypeError
            If the definition or backend output violates nominal runtime types.
        ValueError
            If backend SVD evidence, metadata, or a full-rank transformation is
            inconsistent.
        FloatingPointError
            If shared arithmetic or backend values become nonfinite.
        RuntimeError
            If the selected backend reports SVD nonconvergence.
        """
        if type(definition) is not ComplexUnitaryProcrustesDefinition:
            raise TypeError("definition must be ComplexUnitaryProcrustesDefinition")
        source = definition.source_matrix
        target = definition.target_matrix
        dimension = definition.column_count
        tolerance = definition.tolerance
        # A†B fixes the public direction: Q multiplies the source on the right.
        cross = _matrix_product(_conjugate_transpose(source), target)
        output = self._svd_validated(cross)
        if type(output) is not tuple:
            raise TypeError("backend output must be an exact tuple")
        if len(output) != 3:
            raise ValueError(
                "backend output must contain singular values and both factor matrices"
            )
        singular_values = _singular_values(
            output[0],
            dimension,
            nonfinite_backend=True,
        )
        left = _matrix(
            output[1],
            "backend left singular vectors",
            row_count=dimension,
            column_count=dimension,
            nonfinite_backend=True,
        )
        right_adjoint = _matrix(
            output[2],
            "backend right singular vectors adjoint",
            row_count=dimension,
            column_count=dimension,
            nonfinite_backend=True,
        )

        identity = _identity(dimension)
        unitary_bound = _validation_bound(tolerance, 1.0, dimension)
        left_defect = _maximum_absolute_difference(
            _matrix_product(_conjugate_transpose(left), left),
            identity,
        )
        right_defect = _maximum_absolute_difference(
            _matrix_product(right_adjoint, _conjugate_transpose(right_adjoint)),
            identity,
        )
        if left_defect > unitary_bound:
            raise ValueError("backend left singular vectors violate the unitary bound")
        if right_defect > unitary_bound:
            raise ValueError("backend right singular vectors violate the unitary bound")
        reconstructed_cross = _matrix_product(
            _right_diagonal_product(left, singular_values),
            right_adjoint,
        )
        cross_scale = _maximum_absolute_entry(cross)
        if _maximum_absolute_difference(reconstructed_cross, cross) > _validation_bound(
            tolerance,
            cross_scale,
            dimension,
        ):
            raise ValueError("backend SVD violates the reconstruction bound")

        cutoff = _rank_cutoff(tolerance, singular_values[0])
        # Equality is deliberately deficient so the boundary cannot silently flip
        # between producer and verifier implementations.
        numerical_rank = sum(value > cutoff for value in singular_values)
        source_power = _squared_frobenius_norm(source)
        target_power = _squared_frobenius_norm(target)
        optimal_squared, _ = _optimal_residual_evidence(
            source_power,
            target_power,
            singular_values,
        )
        # Complete SVDs contain arbitrary null-space bases. The deficient branch
        # deliberately does not combine those bases into a public unitary transform.
        transformation: ComplexUnitaryProcrustesMatrix | None = None
        status = ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
        if numerical_rank == dimension:
            transformation = _matrix_product(left, right_adjoint)
            status = ComplexUnitaryProcrustesStatus.UNIQUE
            direct_squared = _squared_frobenius_norm(
                _matrix_difference(
                    _matrix_product(source, transformation),
                    target,
                )
            )
            residual_scale = max(
                source_power,
                target_power,
                optimal_squared,
            )
            residual_bound = _validation_bound(
                tolerance,
                residual_scale,
                definition.row_count * dimension,
            )
            if abs(direct_squared - optimal_squared) > residual_bound:
                raise ValueError(
                    "backend transformation is inconsistent with the optimal residual"
                )

        backend_identifier = _nonempty_string(
            self._backend_identifier,
            "backend_identifier",
        )
        backend_version = _nonempty_string(
            self._backend_version,
            "backend_version",
        )
        return ComplexUnitaryProcrustesResult(
            definition.row_count,
            dimension,
            tolerance,
            status,
            transformation,
            singular_values,
            numerical_rank,
            cutoff,
            source_power,
            target_power,
            optimal_squared,
            math.sqrt(optimal_squared),
            backend_identifier,
            backend_version,
        )

    @abstractmethod
    def _svd_validated(
        self,
        cross_matrix: ComplexUnitaryProcrustesMatrix,
        /,
    ) -> _BackendOutput:
        """Return a complete SVD of a validated square cross matrix.

        Parameters
        ----------
        cross_matrix
            Finite square matrix ``A.conjugate().T @ B``.

        Returns
        -------
        tuple
            Nonincreasing singular values, left singular vectors, and the adjoint of
            the right singular-vector matrix.

        Raises
        ------
        RuntimeError
            If the backend decomposition does not converge.
        FloatingPointError
            If the backend produces nonfinite output.
        """
        raise NotImplementedError


class ComplexUnitaryProcrustesMethod(StrEnum):
    """Select an owner-provided one-pair implementation.

    Attributes
    ----------
    NUMPY_DENSE
        Dense complex128 SVD through NumPy.
    """

    NUMPY_DENSE = "numpy-dense"


@final
class ComplexUnitaryProcrustesSolverFactory:
    """Construct reviewed built-ins without eagerly importing NumPy."""

    __slots__ = ()

    @staticmethod
    def create(
        method: ComplexUnitaryProcrustesMethod,
        /,
    ) -> ComplexUnitaryProcrustesSolver:
        """Construct the exact requested nominal solver.

        Parameters
        ----------
        method
            Closed built-in implementation selection.

        Returns
        -------
        ComplexUnitaryProcrustesSolver
            Fresh nominal solver instance.

        Raises
        ------
        TypeError
            If ``method`` is not the exact closed enum type.
        """
        if type(method) is not ComplexUnitaryProcrustesMethod:
            raise TypeError("method must be ComplexUnitaryProcrustesMethod")
        if method is ComplexUnitaryProcrustesMethod.NUMPY_DENSE:
            from .numpy.procrustes import NumpyComplexUnitaryProcrustesSolver

            return NumpyComplexUnitaryProcrustesSolver()
        raise AssertionError("unhandled complex-unitary Procrustes method")


__all__ = [
    "ComplexUnitaryProcrustesDefinition",
    "ComplexUnitaryProcrustesMatrix",
    "ComplexUnitaryProcrustesMethod",
    "ComplexUnitaryProcrustesResult",
    "ComplexUnitaryProcrustesSolver",
    "ComplexUnitaryProcrustesSolverFactory",
    "ComplexUnitaryProcrustesStatus",
    "ComplexUnitaryProcrustesTolerance",
    "DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE",
]
