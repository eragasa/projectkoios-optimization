"""Vectorized NumPy complex-unitary right Procrustes alignment.

For same-shaped complex128 arrays ``A`` and ``B`` with shape ``(..., m, n)``, this
capability solves every leading-index problem

``minimize ||A[index] @ Q[index] - B[index]||_F``

with square right-unitary ``Q[index]``. One public batched ``numpy.linalg.svd`` call
operates on ``A.conjugate().swapaxes(-1, -2) @ B``. Full-rank items expose their
unique polar factor; rank-deficient items retain NumPy's SVD-selected unitary completion
only as diagnostic evidence and are explicitly marked non-full-rank.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import final

import numpy as np

from ....numerics.numpy.typing import (
    BoolArray,
    Complex128Array,
    Float64Array,
    IntpArray,
)
from ..procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesTolerance,
)


def _positive_integer(value: object, name: str, /) -> int:
    """Validate a positive exact built-in integer.

    Parameters
    ----------
    value
        Candidate value.
    name
        Field name used in failure messages.

    Returns
    -------
    int
        Validated positive integer.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonempty_string(value: object, name: str, /) -> str:
    """Validate a nonempty exact built-in string.

    Parameters
    ----------
    value
        Candidate metadata.
    name
        Field name used in failure messages.

    Returns
    -------
    str
        Validated string.
    """
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _array(
    value: object,
    name: str,
    dtype: np.dtype[np.generic],
    /,
) -> np.ndarray:
    """Validate an exact finite NumPy array with one exact dtype.

    Parameters
    ----------
    value
        Candidate array.
    name
        Field name used in failure messages.
    dtype
        Required exact NumPy dtype.

    Returns
    -------
    numpy.ndarray
        Defensive C-order read-only copy.

    Raises
    ------
    TypeError
        If the value is not an exact ndarray or has the wrong dtype.
    ValueError
        If any entry is nonfinite.
    """
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have exact dtype {dtype.name}")
    finite = np.isfinite(value)
    if not bool(finite.all()):
        first = tuple(int(component) for component in np.argwhere(~finite)[0])
        raise ValueError(f"{name} contains a nonfinite entry at index {first}")
    # An immutable bytes backing prevents callers from re-enabling NumPy writes with
    # ``setflags`` while retaining an ordinary C-contiguous ndarray public surface.
    backing = value.tobytes(order="C")
    copied = np.frombuffer(backing, dtype=dtype).reshape(value.shape)
    return copied


def _batch_size(batch_shape: tuple[int, ...], /) -> int:
    """Return the exact number of leading-index items.

    Parameters
    ----------
    batch_shape
        Leading batch dimensions.

    Returns
    -------
    int
        Product of the dimensions, including zero.
    """
    return math.prod(batch_shape)


def _batch_index(flat_index: int, batch_shape: tuple[int, ...], /) -> tuple[int, ...]:
    """Convert a flat C-order item index to its leading batch index.

    Parameters
    ----------
    flat_index
        Zero-based flat item index.
    batch_shape
        Nonempty leading shape.

    Returns
    -------
    tuple[int, ...]
        Deterministic leading index.
    """
    return tuple(
        int(component) for component in np.unravel_index(flat_index, batch_shape)
    )


def _first_true_index(
    mask: BoolArray, batch_shape: tuple[int, ...], /
) -> tuple[int, ...]:
    """Return the first true leading index in C order.

    Parameters
    ----------
    mask
        Boolean item mask with the exact leading shape.
    batch_shape
        Nonempty leading shape.

    Returns
    -------
    tuple[int, ...]
        First failing leading index.
    """
    flat = int(np.flatnonzero(mask.reshape(-1))[0])
    return _batch_index(flat, batch_shape)


def _validation_bound(
    tolerance: ComplexUnitaryProcrustesTolerance,
    scale: Float64Array | float,
    count: int,
    /,
) -> Float64Array:
    """Return vectorized dimension-scaled validation bounds.

    Parameters
    ----------
    tolerance
        Existing owner software tolerances.
    scale
        Nonnegative scalar or per-item scale.
    count
        Positive operation-count bound.

    Returns
    -------
    numpy.ndarray
        Finite float64 validation bounds.
    """
    result = count * (
        tolerance.validation_absolute + tolerance.validation_relative * scale
    )
    array = np.asarray(result, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError("validation bound became nonfinite")
    return array


def _residual_evidence(
    source_power: Float64Array,
    target_power: Float64Array,
    singular_values: Float64Array,
    /,
) -> tuple[Float64Array, Float64Array]:
    """Reconstruct vectorized optimal squared residuals with roundoff cutoffs.

    Parameters
    ----------
    source_power
        Per-item squared source Frobenius norms.
    target_power
        Per-item squared target Frobenius norms.
    singular_values
        Complete per-item cross singular spectra.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Optimal squared residuals and applied squared-residual roundoff cutoffs.

    Raises
    ------
    FloatingPointError
        If arithmetic becomes nonfinite or a residual is materially negative.
    """
    twice_singular = 2.0 * singular_values
    if not bool(np.isfinite(twice_singular).all()):
        raise FloatingPointError("cross nuclear-norm contribution became nonfinite")
    scale = np.maximum(source_power, target_power)
    scale = np.maximum(scale, np.max(twice_singular, axis=-1))
    nonzero = scale > 0.0
    safe_scale = np.where(nonzero, scale, 1.0)
    normalized = source_power / safe_scale + target_power / safe_scale
    normalized = normalized - np.sum(twice_singular / safe_scale[..., None], axis=-1)
    term_count = singular_values.shape[-1] + 2
    roundoff_factor = max(64.0, 8.0 * float(term_count))
    normalized_absolute_sum = source_power / safe_scale + target_power / safe_scale
    normalized_absolute_sum = normalized_absolute_sum + np.sum(
        twice_singular / safe_scale[..., None],
        axis=-1,
    )
    roundoff_bound = roundoff_factor * sys.float_info.epsilon * normalized_absolute_sum
    cutoff = roundoff_bound * scale
    if not bool(np.isfinite(normalized).all()) or not bool(np.isfinite(cutoff).all()):
        raise FloatingPointError("residual reconstruction became nonfinite")
    materially_negative = normalized < -roundoff_bound
    if bool(materially_negative.any()):
        index = _first_true_index(
            np.asarray(materially_negative, dtype=np.bool_),
            source_power.shape,
        )
        raise FloatingPointError(
            "reconstructed optimal squared residual is materially negative "
            f"at batch index {index}"
        )
    normalized = np.where(np.abs(normalized) <= roundoff_bound, 0.0, normalized)
    squared = normalized * scale
    squared = np.where(nonzero, squared, 0.0)
    if not bool(np.isfinite(squared).all()):
        raise FloatingPointError("optimal squared residual became nonfinite")
    return (
        np.asarray(squared, dtype=np.float64),
        np.asarray(cutoff, dtype=np.float64),
    )


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyBatchedComplexUnitaryProcrustesDefinition:
    """Define a tensor batch of complex-unitary right-alignment problems.

    Parameters
    ----------
    source_matrices
        Exact complex128 array with shape ``(..., m, n)`` and at least one leading
        batch dimension.
    target_matrices
        Exact complex128 array with the same shape as ``source_matrices``.
    tolerance
        Existing owner numerical-rank and validation policy.

    Notes
    -----
    Inputs are defensively copied into C-order read-only arrays. Real, complex64,
    object, Boolean, array-like, and implicit-coercion inputs are rejected.
    """

    source_matrices: Complex128Array = field(repr=False)
    target_matrices: Complex128Array = field(repr=False)
    tolerance: ComplexUnitaryProcrustesTolerance = (
        DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
    )

    def __post_init__(self) -> None:
        """Validate exact arrays, shape, finiteness, and nominal tolerance."""
        source = _array(
            self.source_matrices,
            "source_matrices",
            np.dtype(np.complex128),
        )
        target = _array(
            self.target_matrices,
            "target_matrices",
            np.dtype(np.complex128),
        )
        if source.ndim < 3:
            raise ValueError(
                "source_matrices must have at least one leading batch dimension"
            )
        if target.shape != source.shape:
            raise ValueError("target_matrices shape must equal source_matrices shape")
        if source.shape[-2] <= 0 or source.shape[-1] <= 0:
            raise ValueError("matrix row and column dimensions must be positive")
        if type(self.tolerance) is not ComplexUnitaryProcrustesTolerance:
            raise TypeError("tolerance must be ComplexUnitaryProcrustesTolerance")
        object.__setattr__(self, "source_matrices", source)
        object.__setattr__(self, "target_matrices", target)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return the exact leading batch shape.

        Returns
        -------
        tuple[int, ...]
            One or more leading dimensions.
        """
        return self.source_matrices.shape[:-2]

    @property
    def batch_size(self) -> int:
        """Return the number of leading-index problems.

        Returns
        -------
        int
            Product of :attr:`batch_shape`, including zero.
        """
        return _batch_size(self.batch_shape)

    @property
    def row_count(self) -> int:
        """Return the shared source and target row count.

        Returns
        -------
        int
            Positive matrix row count.
        """
        return int(self.source_matrices.shape[-2])

    @property
    def column_count(self) -> int:
        """Return the shared column and right-unitary dimension.

        Returns
        -------
        int
            Positive matrix column count.
        """
        return int(self.source_matrices.shape[-1])


@dataclass(frozen=True, slots=True)
class _ResultArrays:
    """Hold normalized result arrays during relational validation."""

    completions: Complex128Array
    spectra: Float64Array
    ranks: IntpArray
    full_rank: BoolArray
    rank_cutoffs: Float64Array
    source_powers: Float64Array
    target_powers: Float64Array
    squared_residuals: Float64Array
    residual_norms: Float64Array
    residual_cutoffs: Float64Array


@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class NumpyBatchedComplexUnitaryProcrustesResult:
    """Record immutable pointwise completions and Procrustes evidence.

    Parameters
    ----------
    row_count
        Shared source and target row count.
    tolerance
        Applied owner numerical software controls.
    right_unitary_completions
        SVD-selected right-unitary matrices with shape ``(..., n, n)``. A completion
        is the unique Procrustes transformation only where ``full_rank`` is true.
    cross_singular_values
        Complete nonincreasing spectra with shape ``(..., n)``.
    numerical_ranks, full_rank, applied_rank_cutoffs
        Per-item effective-rank evidence and explicit uniqueness status.
    source_squared_frobenius_norms, target_squared_frobenius_norms
        Per-item input powers.
    optimal_squared_residuals, optimal_residual_frobenius_norms
        Per-item reconstructed optimum evidence.
    applied_residual_roundoff_cutoffs
        Per-item cancellation-noise cutoffs in squared-residual units.
    backend_identifier, backend_version
        NumPy evidence, absent only for an empty batch.

    Notes
    -----
    Every public array is defensively copied and read-only. Rank-deficient completions
    are diagnostic SVD choices, not admitted unique solutions.
    """

    row_count: int
    tolerance: ComplexUnitaryProcrustesTolerance
    right_unitary_completions: Complex128Array = field(repr=False)
    cross_singular_values: Float64Array = field(repr=False)
    numerical_ranks: IntpArray = field(repr=False)
    full_rank: BoolArray = field(repr=False)
    applied_rank_cutoffs: Float64Array = field(repr=False)
    source_squared_frobenius_norms: Float64Array = field(repr=False)
    target_squared_frobenius_norms: Float64Array = field(repr=False)
    optimal_squared_residuals: Float64Array = field(repr=False)
    optimal_residual_frobenius_norms: Float64Array = field(repr=False)
    applied_residual_roundoff_cutoffs: Float64Array = field(repr=False)
    backend_identifier: str | None
    backend_version: str | None

    def __post_init__(self) -> None:
        """Validate result shapes, dtypes, relations, evidence, and immutability."""
        row_count = _positive_integer(self.row_count, "row_count")
        if type(self.tolerance) is not ComplexUnitaryProcrustesTolerance:
            raise TypeError("tolerance must be ComplexUnitaryProcrustesTolerance")
        arrays = _normalize_result_arrays(self)
        batch_shape, dimension = _validate_result_array_shapes(arrays)
        _validate_result_rank_evidence(
            arrays,
            self.tolerance,
            row_count,
            dimension,
        )
        _validate_result_residual_evidence(arrays)
        _validate_result_completions(
            arrays.completions,
            batch_shape,
            dimension,
            self.tolerance,
        )
        backend_identifier, backend_version = _validated_backend_evidence(
            self.backend_identifier,
            self.backend_version,
            batch_shape,
        )
        _store_result_arrays(self, arrays)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "backend_identifier", backend_identifier)
        object.__setattr__(self, "backend_version", backend_version)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return the preserved leading batch shape.

        Returns
        -------
        tuple[int, ...]
            One or more leading dimensions.
        """
        return self.right_unitary_completions.shape[:-2]

    @property
    def batch_size(self) -> int:
        """Return the number of leading-index results.

        Returns
        -------
        int
            Product of :attr:`batch_shape`, including zero.
        """
        return _batch_size(self.batch_shape)

    @property
    def column_count(self) -> int:
        """Return the right-unitary dimension.

        Returns
        -------
        int
            Positive completion width.
        """
        return int(self.right_unitary_completions.shape[-1])


def _normalize_result_arrays(
    result: NumpyBatchedComplexUnitaryProcrustesResult,
    /,
) -> _ResultArrays:
    """Return defensive exact-dtype copies of every result array."""
    return _ResultArrays(
        completions=_array(
            result.right_unitary_completions,
            "right_unitary_completions",
            np.dtype(np.complex128),
        ),
        spectra=_array(
            result.cross_singular_values,
            "cross_singular_values",
            np.dtype(np.float64),
        ),
        ranks=_array(result.numerical_ranks, "numerical_ranks", np.dtype(np.intp)),
        full_rank=_array(result.full_rank, "full_rank", np.dtype(np.bool_)),
        rank_cutoffs=_array(
            result.applied_rank_cutoffs,
            "applied_rank_cutoffs",
            np.dtype(np.float64),
        ),
        source_powers=_array(
            result.source_squared_frobenius_norms,
            "source_squared_frobenius_norms",
            np.dtype(np.float64),
        ),
        target_powers=_array(
            result.target_squared_frobenius_norms,
            "target_squared_frobenius_norms",
            np.dtype(np.float64),
        ),
        squared_residuals=_array(
            result.optimal_squared_residuals,
            "optimal_squared_residuals",
            np.dtype(np.float64),
        ),
        residual_norms=_array(
            result.optimal_residual_frobenius_norms,
            "optimal_residual_frobenius_norms",
            np.dtype(np.float64),
        ),
        residual_cutoffs=_array(
            result.applied_residual_roundoff_cutoffs,
            "applied_residual_roundoff_cutoffs",
            np.dtype(np.float64),
        ),
    )


def _validate_result_array_shapes(
    arrays: _ResultArrays,
    /,
) -> tuple[tuple[int, ...], int]:
    """Validate result array dimensions and return the batch shape and dimension."""
    completions = arrays.completions
    if (
        completions.ndim < 3
        or completions.shape[-1] <= 0
        or completions.shape[-2] != completions.shape[-1]
    ):
        raise ValueError(
            "right_unitary_completions must have shape (..., n, n) with positive n"
        )
    batch_shape = completions.shape[:-2]
    dimension = completions.shape[-1]
    if arrays.spectra.shape != batch_shape + (dimension,):
        raise ValueError("cross_singular_values shape is inconsistent")
    if bool((arrays.spectra < 0.0).any()):
        raise ValueError("cross_singular_values must be nonnegative")
    if dimension > 1 and bool(
        (arrays.spectra[..., :-1] < arrays.spectra[..., 1:]).any()
    ):
        raise ValueError("cross_singular_values must be nonincreasing")
    for name, array in (
        ("numerical_ranks", arrays.ranks),
        ("full_rank", arrays.full_rank),
        ("applied_rank_cutoffs", arrays.rank_cutoffs),
        ("source_squared_frobenius_norms", arrays.source_powers),
        ("target_squared_frobenius_norms", arrays.target_powers),
        ("optimal_squared_residuals", arrays.squared_residuals),
        ("optimal_residual_frobenius_norms", arrays.residual_norms),
        ("applied_residual_roundoff_cutoffs", arrays.residual_cutoffs),
    ):
        if array.shape != batch_shape:
            raise ValueError(f"{name} shape must equal batch_shape")
    return batch_shape, dimension


def _validate_result_rank_evidence(
    arrays: _ResultArrays,
    tolerance: ComplexUnitaryProcrustesTolerance,
    row_count: int,
    dimension: int,
    /,
) -> None:
    """Validate numerical-rank evidence and uniqueness status."""
    if bool((arrays.ranks < 0).any()) or bool((arrays.ranks > dimension).any()):
        raise ValueError("numerical_ranks lie outside the dimension bound")
    if bool((arrays.ranks > row_count).any()):
        raise ValueError("numerical_ranks exceed the matrix row-count bound")
    expected_cutoff = (
        tolerance.rank_absolute + tolerance.rank_relative * arrays.spectra[..., 0]
    )
    if not np.array_equal(arrays.rank_cutoffs, expected_cutoff):
        raise ValueError("applied_rank_cutoffs are inconsistent")
    expected_rank = np.sum(
        arrays.spectra > arrays.rank_cutoffs[..., None],
        axis=-1,
        dtype=np.intp,
    )
    if not np.array_equal(arrays.ranks, expected_rank):
        raise ValueError("numerical_ranks are inconsistent")
    if not np.array_equal(arrays.full_rank, arrays.ranks == dimension):
        raise ValueError("full_rank status is inconsistent")


def _validate_result_residual_evidence(arrays: _ResultArrays, /) -> None:
    """Validate powers, residuals, and roundoff evidence."""
    if bool((arrays.source_powers < 0.0).any()) or bool(
        (arrays.target_powers < 0.0).any()
    ):
        raise ValueError("squared Frobenius norms must be nonnegative")
    if bool((arrays.squared_residuals < 0.0).any()) or bool(
        (arrays.residual_norms < 0.0).any()
    ):
        raise ValueError("optimal residual evidence must be nonnegative")
    if bool((arrays.residual_cutoffs < 0.0).any()):
        raise ValueError("residual roundoff cutoffs must be nonnegative")
    if not np.array_equal(
        arrays.residual_norms,
        np.sqrt(arrays.squared_residuals),
    ):
        raise ValueError("optimal residual norms are inconsistent")
    expected_squared, expected_cutoffs = _residual_evidence(
        arrays.source_powers,
        arrays.target_powers,
        arrays.spectra,
    )
    if not np.array_equal(arrays.squared_residuals, expected_squared):
        raise ValueError("optimal squared residuals are inconsistent")
    if not np.array_equal(arrays.residual_cutoffs, expected_cutoffs):
        raise ValueError("residual roundoff cutoffs are inconsistent")


def _validate_result_completions(
    completions: Complex128Array,
    batch_shape: tuple[int, ...],
    dimension: int,
    tolerance: ComplexUnitaryProcrustesTolerance,
    /,
) -> None:
    """Validate right-unitarity of every public completion."""
    identity = np.eye(dimension, dtype=np.complex128)
    gram = np.swapaxes(completions.conjugate(), -1, -2) @ completions
    defects = np.max(np.abs(gram - identity), axis=(-2, -1))
    bound = _validation_bound(tolerance, 1.0, dimension)
    failures = defects > bound
    if bool(failures.any()):
        index = _first_true_index(
            np.asarray(failures, dtype=np.bool_),
            batch_shape,
        )
        raise ValueError(
            "right_unitary_completions violate the unitary bound "
            f"at batch index {index}"
        )


def _validated_backend_evidence(
    identifier: str | None,
    version: str | None,
    batch_shape: tuple[int, ...],
    /,
) -> tuple[str | None, str | None]:
    """Validate coupled backend evidence for empty and nonempty batches."""
    if _batch_size(batch_shape) == 0:
        if identifier is not None or version is not None:
            raise ValueError("empty batch must not claim backend evidence")
        return None, None
    return (
        _nonempty_string(identifier, "backend_identifier"),
        _nonempty_string(version, "backend_version"),
    )


def _store_result_arrays(
    result: NumpyBatchedComplexUnitaryProcrustesResult,
    arrays: _ResultArrays,
    /,
) -> None:
    """Install normalized arrays on a frozen result record."""
    object.__setattr__(result, "right_unitary_completions", arrays.completions)
    object.__setattr__(result, "cross_singular_values", arrays.spectra)
    object.__setattr__(result, "numerical_ranks", arrays.ranks)
    object.__setattr__(result, "full_rank", arrays.full_rank)
    object.__setattr__(result, "applied_rank_cutoffs", arrays.rank_cutoffs)
    object.__setattr__(result, "source_squared_frobenius_norms", arrays.source_powers)
    object.__setattr__(result, "target_squared_frobenius_norms", arrays.target_powers)
    object.__setattr__(result, "optimal_squared_residuals", arrays.squared_residuals)
    object.__setattr__(
        result, "optimal_residual_frobenius_norms", arrays.residual_norms
    )
    object.__setattr__(
        result,
        "applied_residual_roundoff_cutoffs",
        arrays.residual_cutoffs,
    )


@dataclass(frozen=True, slots=True)
class _CompletionEvidence:
    """Hold selected completions and their numerical-rank evidence."""

    completions: Complex128Array
    ranks: IntpArray
    full_rank: BoolArray
    rank_cutoffs: Float64Array


@dataclass(frozen=True, slots=True)
class _ResidualArrays:
    """Hold per-item input-power and optimal-residual evidence."""

    source_powers: Float64Array
    target_powers: Float64Array
    squared_residuals: Float64Array
    residual_norms: Float64Array
    residual_cutoffs: Float64Array


@final
class NumpyBatchedComplexUnitaryProcrustesSolver:
    """Solve a complete leading-dimensional batch with vectorized NumPy operations.

    Notes
    -----
    A nonempty solve makes one public batched ``numpy.linalg.svd`` call. Validation and
    backend failures reject the entire batch; no partial result is returned. A rare SVD
    failure triggers scalar diagnostic retries only to identify the first failing
    C-order batch index.
    """

    __slots__ = ()

    def solve(
        self,
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        /,
    ) -> NumpyBatchedComplexUnitaryProcrustesResult:
        """Solve every source-to-target right-alignment problem.

        Parameters
        ----------
        definition
            Exact immutable complex128 batch and owner tolerance.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Shape-preserving completions, uniqueness status, and numerical evidence.

        Raises
        ------
        TypeError
            If the definition has the wrong nominal type.
        ValueError
            If SVD or completion evidence violates owner validation bounds.
        FloatingPointError
            If shared or backend arithmetic becomes nonfinite.
        RuntimeError
            If NumPy SVD does not converge, with the first reproducible failing batch
            index when diagnostic isolation succeeds.
        """
        if type(definition) is not NumpyBatchedComplexUnitaryProcrustesDefinition:
            raise TypeError(
                "definition must be NumpyBatchedComplexUnitaryProcrustesDefinition"
            )
        batch_shape = definition.batch_shape
        dimension = definition.column_count
        if definition.batch_size == 0:
            return self._empty_result(definition)

        source = definition.source_matrices
        target = definition.target_matrices
        tolerance = definition.tolerance
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cross = np.swapaxes(source.conjugate(), -1, -2) @ target
        if not bool(np.isfinite(cross).all()):
            item_failure = ~np.isfinite(cross).all(axis=(-2, -1))
            index = _first_true_index(
                np.asarray(item_failure, dtype=np.bool_),
                batch_shape,
            )
            raise FloatingPointError(
                f"cross-matrix arithmetic became nonfinite at batch index {index}"
            )

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                left, singular_values, right_adjoint = np.linalg.svd(
                    cross,
                    full_matrices=True,
                    compute_uv=True,
                )
        except np.linalg.LinAlgError as error:
            self._raise_indexed_nonconvergence(cross, batch_shape, error)

        if not bool(np.isfinite(singular_values).all()):
            failure = ~np.isfinite(singular_values).all(axis=-1)
            index = _first_true_index(np.asarray(failure, dtype=np.bool_), batch_shape)
            raise FloatingPointError(
                f"NumPy SVD returned nonfinite singular values at batch index {index}"
            )
        factor_finite = np.isfinite(left).all(axis=(-2, -1)) & np.isfinite(
            right_adjoint
        ).all(axis=(-2, -1))
        if not bool(factor_finite.all()):
            index = _first_true_index(
                np.asarray(~factor_finite, dtype=np.bool_),
                batch_shape,
            )
            raise FloatingPointError(
                f"NumPy SVD returned nonfinite singular vectors at batch index {index}"
            )

        identity = np.eye(dimension, dtype=np.complex128)
        left_gram = np.swapaxes(left.conjugate(), -1, -2) @ left
        right_gram = right_adjoint @ np.swapaxes(
            right_adjoint.conjugate(),
            -1,
            -2,
        )
        unitary_bound = _validation_bound(tolerance, 1.0, dimension)
        left_defect = np.max(np.abs(left_gram - identity), axis=(-2, -1))
        right_defect = np.max(np.abs(right_gram - identity), axis=(-2, -1))
        self._reject_item_mask(
            left_defect > unitary_bound,
            batch_shape,
            "backend left singular vectors violate the unitary bound",
        )
        self._reject_item_mask(
            right_defect > unitary_bound,
            batch_shape,
            "backend right singular vectors violate the unitary bound",
        )

        reconstructed = (left * singular_values[..., None, :]) @ right_adjoint
        cross_scale = np.max(np.abs(cross), axis=(-2, -1))
        reconstruction_defect = np.max(
            np.abs(reconstructed - cross),
            axis=(-2, -1),
        )
        reconstruction_bound = _validation_bound(
            tolerance,
            cross_scale,
            dimension,
        )
        self._reject_item_mask(
            reconstruction_defect > reconstruction_bound,
            batch_shape,
            "backend SVD violates the reconstruction bound",
        )

        completions = left @ right_adjoint
        completion_gram = np.swapaxes(completions.conjugate(), -1, -2) @ completions
        completion_defect = np.max(
            np.abs(completion_gram - identity),
            axis=(-2, -1),
        )
        self._reject_item_mask(
            completion_defect > unitary_bound,
            batch_shape,
            "SVD-selected completion violates the unitary bound",
        )

        completion_evidence = self._completion_evidence(
            definition,
            np.asarray(completions, dtype=np.complex128),
            np.asarray(singular_values, dtype=np.float64),
        )
        residuals = self._residual_arrays(
            definition,
            completion_evidence.completions,
            np.asarray(singular_values, dtype=np.float64),
        )
        return self._result(
            definition,
            np.asarray(singular_values, dtype=np.float64),
            completion_evidence,
            residuals,
        )

    @staticmethod
    def _completion_evidence(
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        completions: Complex128Array,
        singular_values: Float64Array,
        /,
    ) -> _CompletionEvidence:
        """Apply the owner rank policy to validated completion evidence."""
        tolerance = definition.tolerance
        cutoffs = (
            tolerance.rank_absolute + tolerance.rank_relative * singular_values[..., 0]
        )
        if not bool(np.isfinite(cutoffs).all()):
            raise FloatingPointError("applied numerical-rank cutoff became nonfinite")
        ranks = np.sum(
            singular_values > cutoffs[..., None],
            axis=-1,
            dtype=np.intp,
        )
        return _CompletionEvidence(
            completions=completions,
            ranks=np.asarray(ranks, dtype=np.intp),
            full_rank=np.asarray(ranks == definition.column_count, dtype=np.bool_),
            rank_cutoffs=np.asarray(cutoffs, dtype=np.float64),
        )

    @classmethod
    def _residual_arrays(
        cls,
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        completions: Complex128Array,
        singular_values: Float64Array,
        /,
    ) -> _ResidualArrays:
        """Construct and validate per-item power and residual evidence."""
        source = definition.source_matrices
        target = definition.target_matrices
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            source_power = np.sum(
                np.abs(source) ** 2,
                axis=(-2, -1),
                dtype=np.float64,
            )
            target_power = np.sum(
                np.abs(target) ** 2,
                axis=(-2, -1),
                dtype=np.float64,
            )
        power_finite = np.isfinite(source_power) & np.isfinite(target_power)
        if not bool(power_finite.all()):
            index = _first_true_index(
                np.asarray(~power_finite, dtype=np.bool_),
                definition.batch_shape,
            )
            raise FloatingPointError(
                f"squared Frobenius norm became nonfinite at batch index {index}"
            )
        normalized_source_power = np.asarray(source_power, dtype=np.float64)
        normalized_target_power = np.asarray(target_power, dtype=np.float64)
        optimal_squared, residual_cutoff = _residual_evidence(
            normalized_source_power,
            normalized_target_power,
            singular_values,
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            direct_squared = np.sum(
                np.abs(source @ completions - target) ** 2,
                axis=(-2, -1),
                dtype=np.float64,
            )
        if not bool(np.isfinite(direct_squared).all()):
            index = _first_true_index(
                np.asarray(~np.isfinite(direct_squared), dtype=np.bool_),
                definition.batch_shape,
            )
            raise FloatingPointError(
                f"direct residual became nonfinite at batch index {index}"
            )
        residual_scale = np.maximum(source_power, target_power)
        residual_scale = np.maximum(residual_scale, optimal_squared)
        residual_bound = _validation_bound(
            definition.tolerance,
            np.asarray(residual_scale, dtype=np.float64),
            definition.row_count * definition.column_count,
        )
        cls._reject_item_mask(
            np.abs(direct_squared - optimal_squared) > residual_bound,
            definition.batch_shape,
            "SVD-selected completion is inconsistent with the optimal residual",
        )
        return _ResidualArrays(
            source_powers=normalized_source_power,
            target_powers=normalized_target_power,
            squared_residuals=optimal_squared,
            residual_norms=np.asarray(np.sqrt(optimal_squared), dtype=np.float64),
            residual_cutoffs=residual_cutoff,
        )

    @staticmethod
    def _result(
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        singular_values: Float64Array,
        completion: _CompletionEvidence,
        residuals: _ResidualArrays,
        /,
    ) -> NumpyBatchedComplexUnitaryProcrustesResult:
        """Construct a validated nonempty public result from grouped evidence."""
        return NumpyBatchedComplexUnitaryProcrustesResult(
            row_count=definition.row_count,
            tolerance=definition.tolerance,
            right_unitary_completions=completion.completions,
            cross_singular_values=singular_values,
            numerical_ranks=completion.ranks,
            full_rank=completion.full_rank,
            applied_rank_cutoffs=completion.rank_cutoffs,
            source_squared_frobenius_norms=residuals.source_powers,
            target_squared_frobenius_norms=residuals.target_powers,
            optimal_squared_residuals=residuals.squared_residuals,
            optimal_residual_frobenius_norms=residuals.residual_norms,
            applied_residual_roundoff_cutoffs=residuals.residual_cutoffs,
            backend_identifier="numpy.linalg.svd",
            backend_version=str(np.__version__),
        )

    @staticmethod
    def _empty_result(
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        /,
    ) -> NumpyBatchedComplexUnitaryProcrustesResult:
        """Construct an exact shape-preserving result without invoking NumPy SVD.

        Parameters
        ----------
        definition
            Validated definition with zero batch size.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Empty result with absent backend evidence.
        """
        batch_shape = definition.batch_shape
        dimension = definition.column_count
        return NumpyBatchedComplexUnitaryProcrustesResult(
            row_count=definition.row_count,
            tolerance=definition.tolerance,
            right_unitary_completions=np.empty(
                batch_shape + (dimension, dimension),
                dtype=np.complex128,
            ),
            cross_singular_values=np.empty(
                batch_shape + (dimension,),
                dtype=np.float64,
            ),
            numerical_ranks=np.empty(batch_shape, dtype=np.intp),
            full_rank=np.empty(batch_shape, dtype=np.bool_),
            applied_rank_cutoffs=np.empty(batch_shape, dtype=np.float64),
            source_squared_frobenius_norms=np.empty(batch_shape, dtype=np.float64),
            target_squared_frobenius_norms=np.empty(batch_shape, dtype=np.float64),
            optimal_squared_residuals=np.empty(batch_shape, dtype=np.float64),
            optimal_residual_frobenius_norms=np.empty(batch_shape, dtype=np.float64),
            applied_residual_roundoff_cutoffs=np.empty(
                batch_shape,
                dtype=np.float64,
            ),
            backend_identifier=None,
            backend_version=None,
        )

    @staticmethod
    def _reject_item_mask(
        mask: BoolArray,
        batch_shape: tuple[int, ...],
        message: str,
        /,
    ) -> None:
        """Reject a whole batch at the first failing item in C order.

        Parameters
        ----------
        mask
            Per-item failure mask.
        batch_shape
            Exact leading shape.
        message
            Stable failure condition.

        Raises
        ------
        ValueError
            If any item fails, annotated with its leading index.
        """
        normalized = np.asarray(mask, dtype=np.bool_)
        if bool(normalized.any()):
            index = _first_true_index(normalized, batch_shape)
            raise ValueError(f"{message} at batch index {index}")

    @staticmethod
    def _raise_indexed_nonconvergence(
        cross: Complex128Array,
        batch_shape: tuple[int, ...],
        error: np.linalg.LinAlgError,
        /,
    ) -> None:
        """Identify a reproducible scalar SVD failure and reject the whole batch.

        Parameters
        ----------
        cross
            Finite cross matrices whose batched decomposition failed.
        batch_shape
            Exact leading shape.
        error
            Original NumPy batch failure.

        Raises
        ------
        RuntimeError
            Always, with the first scalar-failing C-order index when available.
        """
        dimension = cross.shape[-1]
        for flat_index, matrix in enumerate(cross.reshape((-1, dimension, dimension))):
            try:
                np.linalg.svd(matrix, full_matrices=True, compute_uv=True)
            except np.linalg.LinAlgError as diagnostic_error:
                index = _batch_index(flat_index, batch_shape)
                raise RuntimeError(
                    "NumPy singular-value decomposition did not converge "
                    f"at batch index {index}"
                ) from diagnostic_error
        raise RuntimeError(
            "NumPy batched singular-value decomposition did not converge"
        ) from error


__all__ = [
    "NumpyBatchedComplexUnitaryProcrustesDefinition",
    "NumpyBatchedComplexUnitaryProcrustesResult",
    "NumpyBatchedComplexUnitaryProcrustesSolver",
]
