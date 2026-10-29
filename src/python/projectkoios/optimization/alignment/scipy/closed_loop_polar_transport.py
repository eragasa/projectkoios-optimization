"""SciPy execution for closed-loop polar frame transport."""

from __future__ import annotations

import math
from typing import NoReturn, cast, final

import numpy as np
import scipy
from scipy.linalg import schur

from projectkoios.numerics.numpy.typing import (
    Complex128Array,
    Complex128Matrix,
    Float64Array,
)
from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportBackendFailure,
    ClosedLoopPolarFrameTransportBackendFailureReason,
    ClosedLoopPolarFrameTransportBackendOperation,
    ClosedLoopPolarFrameTransportLinkDiagnostic,
    ClosedLoopPolarFrameTransportLinkKind,
    ClosedLoopPolarFrameTransportLinkStatus,
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
    ClosedLoopPolarFrameTransportStatus,
)
from projectkoios.optimization.alignment.numpy.closed_loop_polar_transport import (
    ClosedLoopPolarFrameInventory,
    ClosedLoopPolarFrameRightActions,
    NumpyClosedLoopPolarFrameTransportDefinition,
    NumpyClosedLoopPolarFrameTransportResult,
)
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesMatrix,
    ComplexUnitaryProcrustesResult,
    ComplexUnitaryProcrustesStatus,
)


@final
class ScipyClosedLoopPolarFrameTransportSolver:
    """Execute sequential NumPy-SVD transport and branch-explicit Schur correction."""

    __slots__ = ("_procrustes_solver",)

    def __init__(
        self,
        procrustes_solver: NumpyComplexUnitaryProcrustesSolver,
    ) -> None:
        if type(procrustes_solver) is not NumpyComplexUnitaryProcrustesSolver:
            raise TypeError(
                "procrustes_solver must be NumpyComplexUnitaryProcrustesSolver"
            )
        self._procrustes_solver = procrustes_solver

    def solve(
        self,
        definition: NumpyClosedLoopPolarFrameTransportDefinition,
        /,
    ) -> NumpyClosedLoopPolarFrameTransportResult:
        """Transport and close one route, or retain its first deficient link."""
        if type(definition) is not NumpyClosedLoopPolarFrameTransportDefinition:
            raise TypeError(
                "definition must be NumpyClosedLoopPolarFrameTransportDefinition"
            )
        preliminary_frames: list[Complex128Matrix] = [
            cast(
                Complex128Matrix,
                np.array(definition.frames[0], dtype=np.complex128, copy=True),
            )
        ]
        primitive_actions: list[Complex128Matrix] = [
            np.eye(definition.rank, dtype=np.complex128)
        ]
        diagnostics: list[ClosedLoopPolarFrameTransportLinkDiagnostic] = []

        for link_index in range(definition.point_count - 1):
            result = self._solve_link(
                definition.frames[link_index + 1],
                preliminary_frames[link_index],
                definition,
                link_index,
            )
            diagnostic = self._diagnostic(
                link_index,
                ClosedLoopPolarFrameTransportLinkKind.NEIGHBOR,
                result,
            )
            diagnostics.append(diagnostic)
            if result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT:
                diagnostics.extend(
                    self._unattempted_diagnostics(
                        link_index + 1,
                        definition.point_count,
                    )
                )
                return self._deficient_result(
                    definition,
                    tuple(diagnostics),
                    result,
                )
            transformation_value = result.transformation
            if transformation_value is None:
                raise ValueError("qualified Procrustes link has no transformation")
            transformation = cast(
                Complex128Matrix,
                np.asarray(transformation_value, dtype=np.complex128),
            )
            primitive_actions.append(transformation)
            preliminary_frames.append(
                cast(
                    Complex128Matrix,
                    np.asarray(
                        definition.frames[link_index + 1] @ transformation,
                        dtype=np.complex128,
                    ),
                )
            )

        closure_index = definition.point_count - 1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sewn_first = cast(
                Complex128Matrix,
                np.asarray(
                    definition.endpoint_sewing_map @ preliminary_frames[0],
                    dtype=np.complex128,
                ),
            )
        if not bool(np.isfinite(sewn_first).all()):
            raise FloatingPointError("endpoint sewing arithmetic became nonfinite")
        closure = self._solve_link(
            preliminary_frames[-1],
            sewn_first,
            definition,
            closure_index,
        )
        diagnostics.append(
            self._diagnostic(
                closure_index,
                ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
                closure,
            )
        )
        if closure.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT:
            return self._deficient_result(
                definition,
                tuple(diagnostics),
                closure,
            )
        closure_transformation = closure.transformation
        if closure_transformation is None:
            raise ValueError("qualified closure link has no transformation")
        closure_action = cast(
            Complex128Matrix,
            np.asarray(closure_transformation, dtype=np.complex128),
        )
        endpoint_mismatch = cast(
            Complex128Matrix,
            np.asarray(closure_action.conjugate().T, dtype=np.complex128),
        )
        phases, schur_vectors = self._closure_schur_evidence(
            endpoint_mismatch,
            definition,
            closure_index,
        )
        phase_tuple = tuple(float(phase) for phase in phases)
        distributed_frames: list[Complex128Matrix] = []
        final_actions: list[Complex128Matrix] = []
        for index, primitive_action in enumerate(primitive_actions):
            correction = self._phase_matrix(
                schur_vectors,
                phases,
                -float(index) / float(definition.point_count),
                closure_index,
            )
            final_action = cast(
                Complex128Matrix,
                np.asarray(primitive_action @ correction, dtype=np.complex128),
            )
            final_actions.append(final_action)
            distributed_frames.append(
                cast(
                    Complex128Matrix,
                    np.asarray(
                        definition.frames[index] @ final_action,
                        dtype=np.complex128,
                    ),
                )
            )
        correction_step = self._phase_matrix(
            schur_vectors,
            phases,
            -1.0 / float(definition.point_count),
            closure_index,
        )
        if (
            self._maximum_absolute_entry_unitarity_defect(correction_step)
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        if (
            self._maximum_absolute_difference(
                np.linalg.matrix_power(correction_step, definition.point_count),
                closure_action,
            )
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        closure_alignment_residual = self._maximum_absolute_difference(
            preliminary_frames[-1] @ closure_action,
            sewn_first,
        )
        closure_alignment_unitarity_defect = (
            self._maximum_absolute_entry_unitarity_defect(closure_action)
        )
        return NumpyClosedLoopPolarFrameTransportResult.from_arrays(
            definition=definition,
            status=ClosedLoopPolarFrameTransportStatus.QUALIFIED,
            link_diagnostics=tuple(diagnostics),
            deficient_link_indices=(),
            unattempted_link_indices=(),
            transported_frames=cast(
                ClosedLoopPolarFrameInventory,
                np.asarray(distributed_frames, dtype=np.complex128),
            ),
            frame_right_actions=cast(
                ClosedLoopPolarFrameRightActions,
                np.asarray(final_actions, dtype=np.complex128),
            ),
            closure_alignment_right_action=closure_action,
            closure_endpoint_mismatch=endpoint_mismatch,
            distributed_closure_correction_step=correction_step,
            closure_schur_vectors=schur_vectors,
            maximum_absolute_entry_closure_alignment_residual=(
                closure_alignment_residual
            ),
            maximum_absolute_entry_closure_alignment_unitarity_defect=(
                closure_alignment_unitarity_defect
            ),
            ordered_closure_endpoint_mismatch_phases=phase_tuple,
            svd_backend_identifier=closure.backend_identifier,
            svd_backend_version=closure.backend_version,
            schur_backend_identifier="scipy.linalg.schur",
            schur_backend_version=str(scipy.__version__),
        )

    def _solve_link(
        self,
        source: Complex128Matrix,
        target: Complex128Matrix,
        definition: NumpyClosedLoopPolarFrameTransportDefinition,
        link_index: int,
    ) -> ComplexUnitaryProcrustesResult:
        problem = ComplexUnitaryProcrustesDefinition(
            self._tuple_matrix(source),
            self._tuple_matrix(target),
            definition.tolerance,
        )
        try:
            return self._procrustes_solver.solve(problem)
        except RuntimeError as error:
            raise ClosedLoopPolarFrameTransportBackendFailure(
                ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD,
                ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE,
                link_index,
            ) from error
        except FloatingPointError as error:
            raise ClosedLoopPolarFrameTransportBackendFailure(
                ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD,
                ClosedLoopPolarFrameTransportBackendFailureReason.NONFINITE_OUTPUT,
                link_index,
            ) from error
        except (TypeError, ValueError) as error:
            raise ClosedLoopPolarFrameTransportBackendFailure(
                ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                link_index,
            ) from error

    def _closure_schur_evidence(
        self,
        endpoint_mismatch: Complex128Matrix,
        definition: NumpyClosedLoopPolarFrameTransportDefinition,
        closure_index: int,
    ) -> tuple[Float64Array, Complex128Matrix]:
        try:
            triangular_value, vectors_value = schur(
                endpoint_mismatch,
                output="complex",
            )
        except np.linalg.LinAlgError as error:
            raise ClosedLoopPolarFrameTransportBackendFailure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE,
                closure_index,
            ) from error
        except (FloatingPointError, TypeError, ValueError) as error:
            raise ClosedLoopPolarFrameTransportBackendFailure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            ) from error
        if (
            type(triangular_value) is not np.ndarray
            or type(vectors_value) is not np.ndarray
            or triangular_value.dtype != np.dtype(np.complex128)
            or vectors_value.dtype != np.dtype(np.complex128)
            or triangular_value.shape != (definition.rank, definition.rank)
            or vectors_value.shape != (definition.rank, definition.rank)
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        triangular = cast(Complex128Matrix, triangular_value)
        vectors = cast(Complex128Matrix, vectors_value)
        if not bool(np.isfinite(triangular).all()) or not bool(
            np.isfinite(vectors).all()
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.NONFINITE_OUTPUT,
                closure_index,
            )
        identity = np.eye(definition.rank, dtype=np.complex128)
        if (
            self._maximum_absolute_difference(
                vectors.conjugate().T @ vectors,
                identity,
            )
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        if (
            self._maximum_absolute_difference(
                vectors @ triangular @ vectors.conjugate().T,
                endpoint_mismatch,
            )
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        off_diagonal = triangular - np.diag(np.diag(triangular))
        if (
            self._maximum_absolute_entry(off_diagonal)
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        diagonal = np.diag(triangular)
        if (
            float(np.max(np.abs(np.abs(diagonal) - 1.0)))
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        phases = np.angle(diagonal).astype(np.float64)
        branch = definition.principal_phase_branch
        if branch is (
            ClosedLoopPolarFrameTransportPrincipalPhaseBranch.negative_pi_open
        ):
            phases[phases == -math.pi] = math.pi
        else:
            phases[phases == math.pi] = -math.pi
        order = np.argsort(phases, kind="stable")
        ordered_phases = phases[order]
        ordered_vectors = cast(
            Complex128Matrix,
            np.asarray(vectors[:, order], dtype=np.complex128),
        )
        reconstructed = self._phase_matrix(
            ordered_vectors,
            ordered_phases,
            1.0,
            closure_index,
        )
        if (
            self._maximum_absolute_difference(
                reconstructed,
                endpoint_mismatch,
            )
            > definition.maximum_absolute_entry_validation_bound
        ):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION,
                closure_index,
            )
        return ordered_phases, ordered_vectors

    def _phase_matrix(
        self,
        schur_vectors: Complex128Matrix,
        phases: Float64Array,
        fraction: float,
        closure_index: int,
    ) -> Complex128Matrix:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            matrix = (
                schur_vectors
                @ np.diag(np.exp(1j * phases * fraction))
                @ schur_vectors.conjugate().T
            )
        normalized = cast(
            Complex128Matrix,
            np.asarray(matrix, dtype=np.complex128),
        )
        if not bool(np.isfinite(normalized).all()):
            self._raise_backend_failure(
                ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR,
                ClosedLoopPolarFrameTransportBackendFailureReason.NONFINITE_OUTPUT,
                closure_index,
            )
        return normalized

    def _diagnostic(
        self,
        index: int,
        kind: ClosedLoopPolarFrameTransportLinkKind,
        result: ComplexUnitaryProcrustesResult,
    ) -> ClosedLoopPolarFrameTransportLinkDiagnostic:
        singular_values = result.cross_singular_values
        cutoff = result.applied_rank_cutoff
        if result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT:
            return ClosedLoopPolarFrameTransportLinkDiagnostic(
                index,
                kind,
                ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT,
                singular_values,
                cutoff,
                singular_values[-1] - cutoff,
                result.numerical_rank,
                None,
            )
        transformation_value = result.transformation
        if transformation_value is None:
            raise ValueError("qualified Procrustes link has no transformation")
        transformation = cast(
            Complex128Matrix,
            np.asarray(transformation_value, dtype=np.complex128),
        )
        return ClosedLoopPolarFrameTransportLinkDiagnostic(
            index,
            kind,
            ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED,
            singular_values,
            cutoff,
            singular_values[-1] - cutoff,
            result.numerical_rank,
            self._maximum_absolute_entry_unitarity_defect(transformation),
        )

    def _unattempted_diagnostics(
        self,
        start: int,
        point_count: int,
    ) -> tuple[ClosedLoopPolarFrameTransportLinkDiagnostic, ...]:
        diagnostics: list[ClosedLoopPolarFrameTransportLinkDiagnostic] = []
        for index in range(start, point_count):
            kind = ClosedLoopPolarFrameTransportLinkKind.NEIGHBOR
            if index == point_count - 1:
                kind = ClosedLoopPolarFrameTransportLinkKind.CLOSURE
            diagnostics.append(
                ClosedLoopPolarFrameTransportLinkDiagnostic(
                    index,
                    kind,
                    ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
        return tuple(diagnostics)

    def _deficient_result(
        self,
        definition: NumpyClosedLoopPolarFrameTransportDefinition,
        diagnostics: tuple[ClosedLoopPolarFrameTransportLinkDiagnostic, ...],
        last_result: ComplexUnitaryProcrustesResult,
    ) -> NumpyClosedLoopPolarFrameTransportResult:
        deficient = tuple(
            item.index
            for item in diagnostics
            if item.status is ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT
        )
        unattempted = tuple(
            item.index
            for item in diagnostics
            if item.status is ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED
        )
        return NumpyClosedLoopPolarFrameTransportResult(
            definition=definition,
            status=ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT,
            link_diagnostics=diagnostics,
            deficient_link_indices=deficient,
            unattempted_link_indices=unattempted,
            transported_frames=None,
            frame_right_actions=None,
            closure_alignment_right_action=None,
            closure_endpoint_mismatch=None,
            distributed_closure_correction_step=None,
            closure_schur_vectors=None,
            maximum_absolute_entry_closure_alignment_residual=None,
            maximum_absolute_entry_closure_alignment_unitarity_defect=None,
            ordered_closure_endpoint_mismatch_phases=None,
            svd_backend_identifier=last_result.backend_identifier,
            svd_backend_version=last_result.backend_version,
            schur_backend_identifier=None,
            schur_backend_version=None,
        )

    def _tuple_matrix(
        self,
        array: Complex128Matrix,
    ) -> ComplexUnitaryProcrustesMatrix:
        return tuple(tuple(complex(value) for value in row) for row in array)

    def _maximum_absolute_entry_unitarity_defect(
        self,
        matrix: Complex128Matrix,
    ) -> float:
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        return self._maximum_absolute_difference(
            matrix.conjugate().T @ matrix,
            identity,
        )

    def _maximum_absolute_entry(self, matrix: Complex128Matrix) -> float:
        magnitude = np.abs(matrix)
        if not bool(np.isfinite(magnitude).all()):
            raise FloatingPointError("matrix magnitude became nonfinite")
        return float(np.max(magnitude, initial=0.0))

    def _maximum_absolute_difference(
        self,
        left: Complex128Array,
        right: Complex128Array,
    ) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            difference = np.abs(left - right)
        if not bool(np.isfinite(difference).all()):
            raise FloatingPointError("array-difference arithmetic became nonfinite")
        return float(np.max(difference, initial=0.0))

    def _raise_backend_failure(
        self,
        operation: ClosedLoopPolarFrameTransportBackendOperation,
        reason: ClosedLoopPolarFrameTransportBackendFailureReason,
        link_index: int,
    ) -> NoReturn:
        raise ClosedLoopPolarFrameTransportBackendFailure(
            operation,
            reason,
            link_index,
        )


__all__ = ["ScipyClosedLoopPolarFrameTransportSolver"]
