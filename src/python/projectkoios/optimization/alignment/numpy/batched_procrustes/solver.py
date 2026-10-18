"""Execute batched complex-unitary right Procrustes alignment with NumPy.

Notes
-----
The concrete solver owns cross-matrix construction, one public batched SVD call,
backend validation, deterministic failure isolation, and public result assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import numpy as np

from projectkoios.numerics.numpy.typing import (
    BoolArray,
    Complex128Array,
    Float64Array,
    IntpArray,
)

from .contract import (
    BatchedComplexUnitaryProcrustesSolver,
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
)
from .numerics import (
    ResidualReconstruction,
    batch_index,
    first_true_index,
    validation_bound,
)


@dataclass(frozen=True, slots=True)
class CompletionEvidence:
    """Hold selected completions and their numerical-rank evidence.

    Parameters
    ----------
    completions
        SVD-selected right-unitary matrices.
    ranks
        Numerical ranks under the applied declared cutoffs.
    full_rank
        Per-item unique-transformation status.
    rank_cutoffs
        Exact per-item cutoffs used by strict rank comparison.

    Notes
    -----
    This solver-construction record is not exported by the public façade.
    """

    completions: Complex128Array
    ranks: IntpArray
    full_rank: BoolArray
    rank_cutoffs: Float64Array


@dataclass(frozen=True, slots=True)
class ResidualArrays:
    """Hold per-item input-power and optimal-residual evidence.

    Parameters
    ----------
    source_powers, target_powers
        Squared source and target Frobenius norms.
    squared_residuals
        Reconstructed optimal squared residuals.
    residual_norms
        Nonnegative square roots of ``squared_residuals``.
    residual_cutoffs
        Applied cancellation cutoffs in squared-residual units.

    Notes
    -----
    This solver-construction record is not exported by the public façade.
    """

    source_powers: Float64Array
    target_powers: Float64Array
    squared_residuals: Float64Array
    residual_norms: Float64Array
    residual_cutoffs: Float64Array


@final
class NumpyBatchedComplexUnitaryProcrustesSolver(BatchedComplexUnitaryProcrustesSolver):
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
            index = first_true_index(
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
            index = first_true_index(np.asarray(failure, dtype=np.bool_), batch_shape)
            raise FloatingPointError(
                f"NumPy SVD returned nonfinite singular values at batch index {index}"
            )
        factor_finite = np.isfinite(left).all(axis=(-2, -1)) & np.isfinite(
            right_adjoint
        ).all(axis=(-2, -1))
        if not bool(factor_finite.all()):
            index = first_true_index(
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
        unitary_bound = validation_bound(tolerance, 1.0, dimension)
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
        reconstruction_bound = validation_bound(
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

    @classmethod
    def _completion_evidence(
        cls,
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        completions: Complex128Array,
        singular_values: Float64Array,
        /,
    ) -> CompletionEvidence:
        """Apply the owner rank policy to validated completion evidence.

        Parameters
        ----------
        definition
            Immutable batch definition and shared tolerance.
        completions
            Validated SVD-selected right-unitary matrices.
        singular_values
            Complete nonincreasing cross singular spectra.

        Returns
        -------
        CompletionEvidence
            Completions with ranks, uniqueness status, and applied cutoffs.

        Raises
        ------
        FloatingPointError
            If vectorized cutoff arithmetic becomes nonfinite.
        """
        cutoffs = cls.rank_cutoffs(
            definition.tolerance,
            singular_values[..., 0],
        )
        ranks = np.sum(
            singular_values > cutoffs[..., None],
            axis=-1,
            dtype=np.intp,
        )
        return CompletionEvidence(
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
    ) -> ResidualArrays:
        """Construct and validate per-item power and residual evidence.

        Parameters
        ----------
        definition
            Immutable source-target batch and owner tolerance.
        completions
            Validated SVD-selected right-unitary matrices.
        singular_values
            Complete nonincreasing cross singular spectra.

        Returns
        -------
        ResidualArrays
            Per-item powers, residuals, norms, and roundoff cutoffs.

        Raises
        ------
        FloatingPointError
            If power, residual, or reconstruction arithmetic becomes nonfinite.
        ValueError
            If direct and reconstructed residual evidence disagree.
        """
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
            index = first_true_index(
                np.asarray(~power_finite, dtype=np.bool_),
                definition.batch_shape,
            )
            raise FloatingPointError(
                f"squared Frobenius norm became nonfinite at batch index {index}"
            )
        normalized_source_power = np.asarray(source_power, dtype=np.float64)
        normalized_target_power = np.asarray(target_power, dtype=np.float64)
        reconstruction = ResidualReconstruction.from_procrustes_evidence(
            normalized_source_power,
            normalized_target_power,
            singular_values,
        )
        optimal_squared = reconstruction.squared_residuals

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            direct_squared = np.sum(
                np.abs(source @ completions - target) ** 2,
                axis=(-2, -1),
                dtype=np.float64,
            )
        if not bool(np.isfinite(direct_squared).all()):
            index = first_true_index(
                np.asarray(~np.isfinite(direct_squared), dtype=np.bool_),
                definition.batch_shape,
            )
            raise FloatingPointError(
                f"direct residual became nonfinite at batch index {index}"
            )
        residual_scale = np.maximum(source_power, target_power)
        residual_scale = np.maximum(residual_scale, optimal_squared)
        residual_bound = validation_bound(
            definition.tolerance,
            np.asarray(residual_scale, dtype=np.float64),
            definition.row_count * definition.column_count,
        )
        cls._reject_item_mask(
            np.abs(direct_squared - optimal_squared) > residual_bound,
            definition.batch_shape,
            "SVD-selected completion is inconsistent with the optimal residual",
        )
        return ResidualArrays(
            source_powers=normalized_source_power,
            target_powers=normalized_target_power,
            squared_residuals=optimal_squared,
            residual_norms=np.asarray(np.sqrt(optimal_squared), dtype=np.float64),
            residual_cutoffs=reconstruction.roundoff_cutoffs,
        )

    @staticmethod
    def _result(
        definition: NumpyBatchedComplexUnitaryProcrustesDefinition,
        singular_values: Float64Array,
        completion: CompletionEvidence,
        residuals: ResidualArrays,
        /,
    ) -> NumpyBatchedComplexUnitaryProcrustesResult:
        """Construct a validated nonempty public result from grouped evidence.

        Parameters
        ----------
        definition
            Immutable source-target batch and owner tolerance.
        singular_values
            Complete nonincreasing cross singular spectra.
        completion
            Grouped completion and rank evidence.
        residuals
            Grouped power and residual evidence.

        Returns
        -------
        NumpyBatchedComplexUnitaryProcrustesResult
            Independently validated immutable result.
        """
        return NumpyBatchedComplexUnitaryProcrustesResult.from_arrays(
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
        return NumpyBatchedComplexUnitaryProcrustesResult.from_arrays(
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
            index = first_true_index(normalized, batch_shape)
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
                index = batch_index(flat_index, batch_shape)
                raise RuntimeError(
                    "NumPy singular-value decomposition did not converge "
                    f"at batch index {index}"
                ) from diagnostic_error
        raise RuntimeError(
            "NumPy batched singular-value decomposition did not converge"
        ) from error


__all__ = ["NumpyBatchedComplexUnitaryProcrustesSolver"]
