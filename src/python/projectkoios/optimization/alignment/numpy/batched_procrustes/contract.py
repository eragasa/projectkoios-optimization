"""Own immutable NumPy batched complex-unitary Procrustes contracts.

Notes
-----
This module validates exact-array definitions and results and owns the nominal batched
solver boundary. It delegates residual reconstruction to :mod:`.numerics` and backend
execution to :mod:`.solver`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self, final

import numpy as np

from projectkoios.numerics.numpy.typing import (
    BoolArray,
    Complex128Array,
    Float64Array,
    IntpArray,
)
from projectkoios.numerics.numpy.validation import (
    NumpyFiniteNonnegativeFloat64ArrayValidator,
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)
from projectkoios.numerics.validation import exact_positive_integer
from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesBackendEvidence,
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesTolerance,
)

from .numerics import (
    ResidualReconstruction,
    batch_size,
    first_true_index,
    validation_bound,
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
    The raw frozen constructor accepts only canonical immutable byte-backed arrays.
    Use :meth:`from_arrays` to validate and defensively copy ordinary exact arrays, or
    :meth:`from_immutable_arrays` for explicit zero-copy construction.
    """

    source_matrices: Complex128Array = field(repr=False)
    target_matrices: Complex128Array = field(repr=False)
    tolerance: ComplexUnitaryProcrustesTolerance = (
        DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
    )

    @classmethod
    def from_arrays(
        cls,
        source_matrices: Complex128Array,
        target_matrices: Complex128Array,
        tolerance: ComplexUnitaryProcrustesTolerance = (
            DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
        ),
    ) -> Self:
        """Construct from exact arrays using immutable defensive copies.

        Parameters
        ----------
        source_matrices, target_matrices
            Exact finite complex128 arrays to copy.
        tolerance
            Existing owner numerical-rank and validation policy.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesDefinition
            Frozen definition owning independent immutable arrays.
        """
        return cls(
            NumpyImmutableArrayConstructor(
                source_matrices,
                "source_matrices",
                np.dtype(np.complex128),
            ).execute(),
            NumpyImmutableArrayConstructor(
                target_matrices,
                "target_matrices",
                np.dtype(np.complex128),
            ).execute(),
            tolerance,
        )

    @classmethod
    def from_immutable_arrays(
        cls,
        source_matrices: Complex128Array,
        target_matrices: Complex128Array,
        tolerance: ComplexUnitaryProcrustesTolerance = (
            DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
        ),
    ) -> Self:
        """Construct without copying canonical immutable arrays.

        Parameters
        ----------
        source_matrices, target_matrices
            Canonical finite complex128 arrays with immutable bytes backing.
        tolerance
            Existing owner numerical-rank and validation policy.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesDefinition
            Frozen definition retaining the supplied canonical arrays by identity.
        """
        return cls(source_matrices, target_matrices, tolerance)

    def __post_init__(self) -> None:
        """Validate canonical arrays, shape, finiteness, and nominal tolerance.

        Raises
        ------
        TypeError
            If arrays, dtypes, or tolerance have the wrong exact type.
        ValueError
            If shapes are invalid or an array contains a nonfinite value.
        """
        source = NumpyImmutableArrayValidator(
            self.source_matrices,
            "source_matrices",
            np.dtype(np.complex128),
        ).execute()
        target = NumpyImmutableArrayValidator(
            self.target_matrices,
            "target_matrices",
            np.dtype(np.complex128),
        ).execute()
        if source.ndim < 3:
            raise ValueError(
                "source_matrices must have at least one leading batch dimension"
            )
        if target.shape != source.shape:
            raise ValueError("target_matrices shape must equal source_matrices shape")
        if source.shape[-2] <= 0 or source.shape[-1] <= 0:
            raise ValueError("matrix row and column dimensions must be positive")
        ComplexUnitaryProcrustesSolver.validate_tolerance(self.tolerance)

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
        return batch_size(self.batch_shape)

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
class ResultArrays:
    """Hold normalized result arrays during relational validation.

    Parameters
    ----------
    completions
        Candidate right-unitary completion matrices.
    spectra
        Complete cross singular spectra.
    ranks, full_rank, rank_cutoffs
        Numerical-rank values, uniqueness status, and applied cutoffs.
    source_powers, target_powers
        Squared source and target Frobenius norms.
    squared_residuals, residual_norms, residual_cutoffs
        Optimal residual evidence and squared-unit roundoff cutoffs.

    Notes
    -----
    This encapsulated construction record groups exact references during canonical
    validation. The public result's ``from_arrays`` factory owns defensive copying.
    """

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
    The raw frozen constructor accepts only canonical immutable byte-backed arrays.
    Use :meth:`from_arrays` to make defensive copies or
    :meth:`from_immutable_arrays` for explicit zero-copy construction. Rank-deficient
    completions are diagnostic SVD choices, not admitted unique solutions.
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

    @classmethod
    def from_arrays(
        cls,
        *,
        row_count: int,
        tolerance: ComplexUnitaryProcrustesTolerance,
        right_unitary_completions: Complex128Array,
        cross_singular_values: Float64Array,
        numerical_ranks: IntpArray,
        full_rank: BoolArray,
        applied_rank_cutoffs: Float64Array,
        source_squared_frobenius_norms: Float64Array,
        target_squared_frobenius_norms: Float64Array,
        optimal_squared_residuals: Float64Array,
        optimal_residual_frobenius_norms: Float64Array,
        applied_residual_roundoff_cutoffs: Float64Array,
        backend_identifier: str | None,
        backend_version: str | None,
    ) -> Self:
        """Construct from exact arrays using immutable defensive copies.

        Parameters
        ----------
        row_count
            Shared source and target row count.
        tolerance
            Applied owner numerical software controls.
        right_unitary_completions, cross_singular_values
            Candidate completion matrices and complete singular spectra.
        numerical_ranks, full_rank, applied_rank_cutoffs
            Candidate numerical-rank evidence.
        source_squared_frobenius_norms, target_squared_frobenius_norms
            Candidate input powers.
        optimal_squared_residuals, optimal_residual_frobenius_norms
            Candidate optimal residual evidence.
        applied_residual_roundoff_cutoffs
            Candidate squared-unit cancellation cutoffs.
        backend_identifier, backend_version
            Executed-backend evidence, or absence for an empty batch.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Frozen result owning independent immutable arrays.
        """
        return cls(
            row_count=row_count,
            tolerance=tolerance,
            right_unitary_completions=NumpyImmutableArrayConstructor(
                right_unitary_completions,
                "right_unitary_completions",
                np.dtype(np.complex128),
            ).execute(),
            cross_singular_values=NumpyImmutableArrayConstructor(
                cross_singular_values,
                "cross_singular_values",
                np.dtype(np.float64),
            ).execute(),
            numerical_ranks=NumpyImmutableArrayConstructor(
                numerical_ranks,
                "numerical_ranks",
                np.dtype(np.intp),
            ).execute(),
            full_rank=NumpyImmutableArrayConstructor(
                full_rank,
                "full_rank",
                np.dtype(np.bool_),
            ).execute(),
            applied_rank_cutoffs=NumpyImmutableArrayConstructor(
                applied_rank_cutoffs,
                "applied_rank_cutoffs",
                np.dtype(np.float64),
            ).execute(),
            source_squared_frobenius_norms=NumpyImmutableArrayConstructor(
                source_squared_frobenius_norms,
                "source_squared_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            target_squared_frobenius_norms=NumpyImmutableArrayConstructor(
                target_squared_frobenius_norms,
                "target_squared_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            optimal_squared_residuals=NumpyImmutableArrayConstructor(
                optimal_squared_residuals,
                "optimal_squared_residuals",
                np.dtype(np.float64),
            ).execute(),
            optimal_residual_frobenius_norms=NumpyImmutableArrayConstructor(
                optimal_residual_frobenius_norms,
                "optimal_residual_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            applied_residual_roundoff_cutoffs=NumpyImmutableArrayConstructor(
                applied_residual_roundoff_cutoffs,
                "applied_residual_roundoff_cutoffs",
                np.dtype(np.float64),
            ).execute(),
            backend_identifier=backend_identifier,
            backend_version=backend_version,
        )

    @classmethod
    def from_immutable_arrays(
        cls,
        *,
        row_count: int,
        tolerance: ComplexUnitaryProcrustesTolerance,
        right_unitary_completions: Complex128Array,
        cross_singular_values: Float64Array,
        numerical_ranks: IntpArray,
        full_rank: BoolArray,
        applied_rank_cutoffs: Float64Array,
        source_squared_frobenius_norms: Float64Array,
        target_squared_frobenius_norms: Float64Array,
        optimal_squared_residuals: Float64Array,
        optimal_residual_frobenius_norms: Float64Array,
        applied_residual_roundoff_cutoffs: Float64Array,
        backend_identifier: str | None,
        backend_version: str | None,
    ) -> Self:
        """Construct without copying canonical immutable arrays.

        Parameters
        ----------
        row_count
            Shared source and target row count.
        tolerance
            Applied owner numerical software controls.
        right_unitary_completions, cross_singular_values
            Canonical immutable completion matrices and singular spectra.
        numerical_ranks, full_rank, applied_rank_cutoffs
            Canonical immutable numerical-rank evidence.
        source_squared_frobenius_norms, target_squared_frobenius_norms
            Canonical immutable input powers.
        optimal_squared_residuals, optimal_residual_frobenius_norms
            Canonical immutable residual evidence.
        applied_residual_roundoff_cutoffs
            Canonical immutable squared-unit cancellation cutoffs.
        backend_identifier, backend_version
            Executed-backend evidence, or absence for an empty batch.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Frozen result retaining every supplied canonical array by identity.
        """
        return cls(
            row_count=row_count,
            tolerance=tolerance,
            right_unitary_completions=right_unitary_completions,
            cross_singular_values=cross_singular_values,
            numerical_ranks=numerical_ranks,
            full_rank=full_rank,
            applied_rank_cutoffs=applied_rank_cutoffs,
            source_squared_frobenius_norms=source_squared_frobenius_norms,
            target_squared_frobenius_norms=target_squared_frobenius_norms,
            optimal_squared_residuals=optimal_squared_residuals,
            optimal_residual_frobenius_norms=optimal_residual_frobenius_norms,
            applied_residual_roundoff_cutoffs=applied_residual_roundoff_cutoffs,
            backend_identifier=backend_identifier,
            backend_version=backend_version,
        )

    def __post_init__(self) -> None:
        """Validate result shapes, dtypes, relations, evidence, and immutability.

        Raises
        ------
        TypeError
            If a field has the wrong exact nominal type or dtype.
        ValueError
            If shape, rank, residual, completion, or backend evidence is inconsistent.
        FloatingPointError
            If residual reconstruction or validation bounds become nonfinite.
        """
        row_count = exact_positive_integer(self.row_count, "row_count")
        ComplexUnitaryProcrustesSolver.validate_tolerance(self.tolerance)
        arrays = self._validated_arrays()
        batch_shape, dimension = self._validate_array_shapes(arrays)
        self._validate_rank_evidence(arrays, row_count, dimension)
        self._validate_residual_evidence(arrays)
        self._validate_completions(arrays.completions, batch_shape, dimension)
        self._validated_backend_evidence(batch_shape)

    def _validated_arrays(self) -> ResultArrays:
        """Return validated canonical result arrays without copying.

        Returns
        -------
        ResultArrays
            Named canonical arrays with immutable byte backing.

        Raises
        ------
        TypeError
            If an input is not an exact ndarray or has the wrong dtype.
        ValueError
            If an input is nonfinite or lacks canonical immutable storage.
        """
        return ResultArrays(
            completions=NumpyImmutableArrayValidator(
                self.right_unitary_completions,
                "right_unitary_completions",
                np.dtype(np.complex128),
            ).execute(),
            spectra=NumpyImmutableArrayValidator(
                self.cross_singular_values,
                "cross_singular_values",
                np.dtype(np.float64),
            ).execute(),
            ranks=NumpyImmutableArrayValidator(
                self.numerical_ranks,
                "numerical_ranks",
                np.dtype(np.intp),
            ).execute(),
            full_rank=NumpyImmutableArrayValidator(
                self.full_rank,
                "full_rank",
                np.dtype(np.bool_),
            ).execute(),
            rank_cutoffs=NumpyImmutableArrayValidator(
                self.applied_rank_cutoffs,
                "applied_rank_cutoffs",
                np.dtype(np.float64),
            ).execute(),
            source_powers=NumpyImmutableArrayValidator(
                self.source_squared_frobenius_norms,
                "source_squared_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            target_powers=NumpyImmutableArrayValidator(
                self.target_squared_frobenius_norms,
                "target_squared_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            squared_residuals=NumpyImmutableArrayValidator(
                self.optimal_squared_residuals,
                "optimal_squared_residuals",
                np.dtype(np.float64),
            ).execute(),
            residual_norms=NumpyImmutableArrayValidator(
                self.optimal_residual_frobenius_norms,
                "optimal_residual_frobenius_norms",
                np.dtype(np.float64),
            ).execute(),
            residual_cutoffs=NumpyImmutableArrayValidator(
                self.applied_residual_roundoff_cutoffs,
                "applied_residual_roundoff_cutoffs",
                np.dtype(np.float64),
            ).execute(),
        )

    @staticmethod
    def _validate_array_shapes(
        arrays: ResultArrays,
        /,
    ) -> tuple[tuple[int, ...], int]:
        """Validate result array dimensions and return their shape evidence.

        Parameters
        ----------
        arrays
            Normalized result arrays.

        Returns
        -------
        tuple[tuple[int, ...], int]
            Leading batch shape and positive completion dimension.

        Raises
        ------
        ValueError
            If any array shape or singular-spectrum ordering is inconsistent.
        """
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

    def _validate_rank_evidence(
        self,
        arrays: ResultArrays,
        row_count: int,
        dimension: int,
        /,
    ) -> None:
        """Validate numerical-rank evidence and uniqueness status.

        Parameters
        ----------
        arrays
            Normalized result arrays.
        row_count
            Positive source and target row count.
        dimension
            Positive right-unitary dimension.

        Raises
        ------
        ValueError
            If ranks, cutoffs, or full-rank status are inconsistent.
        """
        if bool((arrays.ranks < 0).any()) or bool((arrays.ranks > dimension).any()):
            raise ValueError("numerical_ranks lie outside the dimension bound")
        if bool((arrays.ranks > row_count).any()):
            raise ValueError("numerical_ranks exceed the matrix row-count bound")
        expected_cutoff = BatchedComplexUnitaryProcrustesSolver.rank_cutoffs(
            self.tolerance,
            arrays.spectra[..., 0],
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

    @staticmethod
    def _validate_residual_evidence(arrays: ResultArrays, /) -> None:
        """Validate powers, residuals, and roundoff evidence.

        Parameters
        ----------
        arrays
            Normalized result arrays.

        Raises
        ------
        ValueError
            If powers, residuals, norms, or cutoffs are inconsistent.
        FloatingPointError
            If independent residual reconstruction becomes nonfinite.
        """
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
        expected = ResidualReconstruction.from_procrustes_evidence(
            arrays.source_powers,
            arrays.target_powers,
            arrays.spectra,
        )
        if not np.array_equal(
            arrays.squared_residuals,
            expected.squared_residuals,
        ):
            raise ValueError("optimal squared residuals are inconsistent")
        if not np.array_equal(arrays.residual_cutoffs, expected.roundoff_cutoffs):
            raise ValueError("residual roundoff cutoffs are inconsistent")

    def _validate_completions(
        self,
        completions: Complex128Array,
        batch_shape: tuple[int, ...],
        dimension: int,
        /,
    ) -> None:
        """Validate right-unitarity of every public completion.

        Parameters
        ----------
        completions
            Normalized completion matrices.
        batch_shape
            Exact leading batch shape.
        dimension
            Positive right-unitary dimension.

        Raises
        ------
        ValueError
            If a completion violates the owner unitarity bound.
        FloatingPointError
            If the validation bound becomes nonfinite.
        """
        identity = np.eye(dimension, dtype=np.complex128)
        gram = np.swapaxes(completions.conjugate(), -1, -2) @ completions
        defects = np.max(np.abs(gram - identity), axis=(-2, -1))
        bound = validation_bound(self.tolerance, 1.0, dimension)
        failures = defects > bound
        if bool(failures.any()):
            index = first_true_index(
                np.asarray(failures, dtype=np.bool_),
                batch_shape,
            )
            raise ValueError(
                "right_unitary_completions violate the unitary bound "
                f"at batch index {index}"
            )

    def _validated_backend_evidence(
        self,
        batch_shape: tuple[int, ...],
        /,
    ) -> ComplexUnitaryProcrustesBackendEvidence | None:
        """Validate coupled backend evidence for empty and nonempty batches.

        Parameters
        ----------
        batch_shape
            Exact leading batch shape.

        Returns
        -------
        ComplexUnitaryProcrustesBackendEvidence or None
            Validated executed-backend evidence, or absence for an empty batch.

        Raises
        ------
        TypeError
            If nonempty backend evidence is not exact string data.
        ValueError
            If evidence is empty or disagrees with empty-batch semantics.
        """
        if batch_size(batch_shape) == 0:
            return ComplexUnitaryProcrustesSolver.validate_absent_backend_evidence(
                self.backend_identifier,
                self.backend_version,
            )
        if self.backend_identifier is None or self.backend_version is None:
            raise ValueError("executed backend must provide complete backend evidence")
        return ComplexUnitaryProcrustesSolver.validate_backend_evidence(
            self.backend_identifier,
            self.backend_version,
        )

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
        return batch_size(self.batch_shape)

    @property
    def column_count(self) -> int:
        """Return the right-unitary dimension.

        Returns
        -------
        int
            Positive completion width.
        """
        return int(self.right_unitary_completions.shape[-1])

    @property
    def all_full_rank(self) -> bool:
        """Return whether every batch item identifies a unique transformation.

        Returns
        -------
        bool
            ``True`` when every item is full rank. Empty batches are vacuously full
            rank.
        """
        return bool(self.full_rank.all())

    @property
    def rank_deficient_count(self) -> int:
        """Return the number of items with nonunique diagnostic completions.

        Returns
        -------
        int
            Count of false entries in :attr:`full_rank`.
        """
        return self.batch_size - int(np.count_nonzero(self.full_rank))

    @property
    def first_rank_deficient_index(self) -> tuple[int, ...] | None:
        """Return the first deficient leading index in deterministic C order.

        Returns
        -------
        tuple[int, ...] or None
            First deficient batch index, or ``None`` when every item is full rank.
        """
        if self.all_full_rank:
            return None
        return first_true_index(
            np.asarray(~self.full_rank, dtype=np.bool_),
            self.batch_shape,
        )

    def require_unique_right_unitary_transformations(self) -> Complex128Array:
        """Return completions only when every item is a unique transformation.

        Returns
        -------
        numpy.ndarray
            The existing immutable :attr:`right_unitary_completions` array.

        Raises
        ------
        ValueError
            If any batch item is rank deficient. The failure identifies the first
            deficient leading index in C order.

        Notes
        -----
        This fail-closed accessor does not alter or hide diagnostic completions. It
        prevents consumers from accidentally admitting a deficient SVD completion as
        a uniquely identified transformation.
        """
        index = self.first_rank_deficient_index
        if index is not None:
            raise ValueError(
                "unique right-unitary transformations are unavailable because "
                f"batch index {index} is rank deficient"
            )
        return self.right_unitary_completions


class BatchedComplexUnitaryProcrustesSolver(ComplexUnitaryProcrustesSolver, ABC):
    """Own policy shared by concrete exact-array batch solvers.

    Notes
    -----
    The nominal batch boundary combines shared Procrustes policy with the exact NumPy
    definition and result contract. It adds vectorized declared-cutoff construction
    while leaving decomposition execution to concrete backends.
    """

    __slots__ = ()

    @classmethod
    @final
    def rank_cutoffs(
        cls,
        tolerance: ComplexUnitaryProcrustesTolerance,
        largest_singular_values: Float64Array,
        /,
    ) -> Float64Array:
        """Return finite vectorized declared rank cutoffs.

        Parameters
        ----------
        tolerance
            Exact shared owner tolerance.
        largest_singular_values
            Per-item largest singular values with exact float64 dtype.

        Returns
        -------
        numpy.ndarray
            Float64 cutoffs with the same batch shape.

        Raises
        ------
        TypeError
            If the tolerance, array type, or array dtype is not exact.
        ValueError
            If a largest singular value is nonfinite or negative.
        FloatingPointError
            If cutoff arithmetic becomes nonfinite.
        """
        validated_tolerance = cls.validate_tolerance(tolerance)
        largest = NumpyFiniteNonnegativeFloat64ArrayValidator(
            largest_singular_values,
            "largest_singular_values",
        ).execute()
        with np.errstate(over="ignore", invalid="ignore"):
            relative_cutoffs = validated_tolerance.rank_relative * largest
            if (
                validated_tolerance.rank_cutoff_rule
                is ComplexUnitaryProcrustesRankCutoffRule.ADDITIVE
            ):
                values = validated_tolerance.rank_absolute + relative_cutoffs
            else:
                values = np.maximum(
                    validated_tolerance.rank_absolute,
                    relative_cutoffs,
                )
        cutoffs = np.asarray(values, dtype=np.float64)
        if not bool(np.isfinite(cutoffs).all()):
            raise FloatingPointError("applied numerical-rank cutoff became nonfinite")
        return cutoffs

    @abstractmethod
    def solve(
        self,
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        /,
    ) -> NumpyBatchedComplexUnitaryProcrustesResult:
        """Solve one complete exact-array batch.

        Parameters
        ----------
        definition
            Immutable exact-array batch definition.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Immutable shape-preserving result and per-item evidence.
        """
        raise NotImplementedError


__all__ = [
    "BatchedComplexUnitaryProcrustesSolver",
    "NumpyBatchedComplexUnitaryProcrustesDefinition",
    "NumpyBatchedComplexUnitaryProcrustesResult",
]
