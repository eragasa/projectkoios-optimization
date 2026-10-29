"""Immutable NumPy contracts for closed-loop polar frame transport."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self, cast, final

import numpy as np

from projectkoios.numerics.numpy.typing import (
    Complex128Array,
    Complex128Matrix,
    Float64Array,
)
from projectkoios.numerics.numpy.validation import (
    immutable_array,
    validate_immutable_array,
)
from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportLinkDiagnostic,
    ClosedLoopPolarFrameTransportLinkKind,
    ClosedLoopPolarFrameTransportLinkStatus,
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
    ClosedLoopPolarFrameTransportStatus,
)
from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesTolerance,
)
from projectkoios.optimization.validation.validation import text

type ClosedLoopPolarFrameInventory = Complex128Array
type ClosedLoopPolarFrameRightActions = Complex128Array


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyClosedLoopPolarFrameTransportDefinition:
    """Define one exact-array closed-loop frame-transport problem.

    ``endpoint_sewing_map`` acts on the first frame to represent that frame at the
    terminal endpoint. ``principal_phase_branch`` explicitly selects ``(-pi, pi]`` or
    ``[-pi, pi)`` for endpoint-mismatch phases. The raw constructor accepts only
    canonical immutable byte-backed arrays. :meth:`from_arrays` makes defensive
    immutable copies.
    """

    frames: ClosedLoopPolarFrameInventory = field(repr=False)
    endpoint_sewing_map: Complex128Matrix = field(repr=False)
    principal_phase_branch: ClosedLoopPolarFrameTransportPrincipalPhaseBranch
    tolerance: ComplexUnitaryProcrustesTolerance = (
        DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
    )

    @classmethod
    def from_arrays(
        cls,
        frames: ClosedLoopPolarFrameInventory,
        endpoint_sewing_map: Complex128Matrix,
        principal_phase_branch: ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
        tolerance: ComplexUnitaryProcrustesTolerance = (
            DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
        ),
    ) -> Self:
        """Construct from exact complex128 arrays using defensive copies."""
        immutable_frames = cast(
            ClosedLoopPolarFrameInventory,
            immutable_array(frames, "frames", np.dtype(np.complex128)),
        )
        immutable_sewing = cast(
            Complex128Matrix,
            immutable_array(
                endpoint_sewing_map,
                "endpoint_sewing_map",
                np.dtype(np.complex128),
            ),
        )
        return cls(
            immutable_frames,
            immutable_sewing,
            principal_phase_branch,
            tolerance,
        )

    @classmethod
    def from_immutable_arrays(
        cls,
        frames: ClosedLoopPolarFrameInventory,
        endpoint_sewing_map: Complex128Matrix,
        principal_phase_branch: ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
        tolerance: ComplexUnitaryProcrustesTolerance = (
            DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE
        ),
    ) -> Self:
        """Construct without copying canonical immutable arrays."""
        return cls(frames, endpoint_sewing_map, principal_phase_branch, tolerance)

    def __post_init__(self) -> None:
        frames = validate_immutable_array(
            self.frames,
            "frames",
            np.dtype(np.complex128),
        )
        sewing = validate_immutable_array(
            self.endpoint_sewing_map,
            "endpoint_sewing_map",
            np.dtype(np.complex128),
        )
        if frames.ndim != 3:
            raise ValueError("frames must have shape (M, ambient, rank)")
        point_count, ambient_dimension, rank = frames.shape
        if point_count <= 0:
            raise ValueError("frames must contain at least one frame")
        if rank <= 0 or ambient_dimension <= 0 or rank > ambient_dimension:
            raise ValueError("frames must satisfy 0 < rank <= ambient")
        if sewing.shape != (ambient_dimension, ambient_dimension):
            raise ValueError("endpoint_sewing_map shape must equal (ambient, ambient)")
        if (
            type(self.principal_phase_branch)
            is not ClosedLoopPolarFrameTransportPrincipalPhaseBranch
        ):
            raise TypeError(
                "principal_phase_branch must be "
                "ClosedLoopPolarFrameTransportPrincipalPhaseBranch"
            )
        if type(self.tolerance) is not ComplexUnitaryProcrustesTolerance:
            raise TypeError("tolerance must be ComplexUnitaryProcrustesTolerance")
        identity = np.eye(rank, dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            defects = np.max(
                np.abs(np.swapaxes(frames.conjugate(), -1, -2) @ frames - identity),
                axis=(-2, -1),
            )
        if not bool(np.isfinite(defects).all()):
            raise FloatingPointError("frame orthonormality arithmetic became nonfinite")
        failures = defects > self.maximum_absolute_entry_validation_bound
        if bool(failures.any()):
            index = int(np.flatnonzero(failures)[0])
            raise ValueError(
                f"frame violates the orthonormality bound at frame index {index}"
            )

    @property
    def point_count(self) -> int:
        """Return the positive number of ordered frames and loop links."""
        return int(self.frames.shape[0])

    @property
    def ambient_dimension(self) -> int:
        """Return the positive ambient row dimension."""
        return int(self.frames.shape[1])

    @property
    def rank(self) -> int:
        """Return the positive retained frame rank."""
        return int(self.frames.shape[2])

    @property
    def maximum_absolute_entry_validation_bound(self) -> float:
        """Return the finite rank-scaled maximum-absolute-entry validation bound."""
        bound = float(
            self.rank
            * (self.tolerance.validation_absolute + self.tolerance.validation_relative)
        )
        if not math.isfinite(bound):
            raise FloatingPointError("validation bound became nonfinite")
        return bound


@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class NumpyClosedLoopPolarFrameTransportResult:
    """Retain one self-validating exact-array route result.

    Phase order is deterministic storage only. Every reported defect names its
    maximum-absolute-entry metric; no field silently represents a Frobenius value.
    """

    definition: NumpyClosedLoopPolarFrameTransportDefinition
    status: ClosedLoopPolarFrameTransportStatus
    link_diagnostics: tuple[ClosedLoopPolarFrameTransportLinkDiagnostic, ...]
    deficient_link_indices: tuple[int, ...]
    unattempted_link_indices: tuple[int, ...]
    transported_frames: ClosedLoopPolarFrameInventory | None = field(repr=False)
    frame_right_actions: ClosedLoopPolarFrameRightActions | None = field(repr=False)
    closure_alignment_right_action: Complex128Matrix | None = field(repr=False)
    closure_endpoint_mismatch: Complex128Matrix | None = field(repr=False)
    distributed_closure_correction_step: Complex128Matrix | None = field(repr=False)
    closure_schur_vectors: Complex128Matrix | None = field(repr=False)
    maximum_absolute_entry_closure_alignment_residual: float | None
    maximum_absolute_entry_closure_alignment_unitarity_defect: float | None
    ordered_closure_endpoint_mismatch_phases: tuple[float, ...] | None
    svd_backend_identifier: str
    svd_backend_version: str
    schur_backend_identifier: str | None
    schur_backend_version: str | None

    @classmethod
    def from_arrays(
        cls,
        *,
        definition: NumpyClosedLoopPolarFrameTransportDefinition,
        status: ClosedLoopPolarFrameTransportStatus,
        link_diagnostics: tuple[ClosedLoopPolarFrameTransportLinkDiagnostic, ...],
        deficient_link_indices: tuple[int, ...],
        unattempted_link_indices: tuple[int, ...],
        transported_frames: ClosedLoopPolarFrameInventory,
        frame_right_actions: ClosedLoopPolarFrameRightActions,
        closure_alignment_right_action: Complex128Matrix,
        closure_endpoint_mismatch: Complex128Matrix,
        distributed_closure_correction_step: Complex128Matrix,
        closure_schur_vectors: Complex128Matrix,
        maximum_absolute_entry_closure_alignment_residual: float,
        maximum_absolute_entry_closure_alignment_unitarity_defect: float,
        ordered_closure_endpoint_mismatch_phases: tuple[float, ...],
        svd_backend_identifier: str,
        svd_backend_version: str,
        schur_backend_identifier: str,
        schur_backend_version: str,
    ) -> Self:
        """Construct a qualified result using immutable defensive array copies."""
        return cls(
            definition=definition,
            status=status,
            link_diagnostics=link_diagnostics,
            deficient_link_indices=deficient_link_indices,
            unattempted_link_indices=unattempted_link_indices,
            transported_frames=cast(
                ClosedLoopPolarFrameInventory,
                immutable_array(
                    transported_frames,
                    "transported_frames",
                    np.dtype(np.complex128),
                ),
            ),
            frame_right_actions=cast(
                ClosedLoopPolarFrameRightActions,
                immutable_array(
                    frame_right_actions,
                    "frame_right_actions",
                    np.dtype(np.complex128),
                ),
            ),
            closure_alignment_right_action=cast(
                Complex128Matrix,
                immutable_array(
                    closure_alignment_right_action,
                    "closure_alignment_right_action",
                    np.dtype(np.complex128),
                ),
            ),
            closure_endpoint_mismatch=cast(
                Complex128Matrix,
                immutable_array(
                    closure_endpoint_mismatch,
                    "closure_endpoint_mismatch",
                    np.dtype(np.complex128),
                ),
            ),
            distributed_closure_correction_step=cast(
                Complex128Matrix,
                immutable_array(
                    distributed_closure_correction_step,
                    "distributed_closure_correction_step",
                    np.dtype(np.complex128),
                ),
            ),
            closure_schur_vectors=cast(
                Complex128Matrix,
                immutable_array(
                    closure_schur_vectors,
                    "closure_schur_vectors",
                    np.dtype(np.complex128),
                ),
            ),
            maximum_absolute_entry_closure_alignment_residual=(
                maximum_absolute_entry_closure_alignment_residual
            ),
            maximum_absolute_entry_closure_alignment_unitarity_defect=(
                maximum_absolute_entry_closure_alignment_unitarity_defect
            ),
            ordered_closure_endpoint_mismatch_phases=(
                ordered_closure_endpoint_mismatch_phases
            ),
            svd_backend_identifier=svd_backend_identifier,
            svd_backend_version=svd_backend_version,
            schur_backend_identifier=schur_backend_identifier,
            schur_backend_version=schur_backend_version,
        )

    def __post_init__(self) -> None:  # noqa: C901
        if type(self.definition) is not NumpyClosedLoopPolarFrameTransportDefinition:
            raise TypeError(
                "definition must be NumpyClosedLoopPolarFrameTransportDefinition"
            )
        if type(self.status) is not ClosedLoopPolarFrameTransportStatus:
            raise TypeError("status must be ClosedLoopPolarFrameTransportStatus")
        text(self.svd_backend_identifier, "svd_backend_identifier")
        text(self.svd_backend_version, "svd_backend_version")
        self._validate_link_progression()
        qualified_values = (
            self.transported_frames,
            self.frame_right_actions,
            self.closure_alignment_right_action,
            self.closure_endpoint_mismatch,
            self.distributed_closure_correction_step,
            self.closure_schur_vectors,
            self.maximum_absolute_entry_closure_alignment_residual,
            self.maximum_absolute_entry_closure_alignment_unitarity_defect,
            self.ordered_closure_endpoint_mismatch_phases,
            self.schur_backend_identifier,
            self.schur_backend_version,
        )
        if self.status is ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT:
            if any(value is not None for value in qualified_values):
                raise ValueError("rank-deficient routes must not expose closure output")
            return
        if any(value is None for value in qualified_values):
            raise ValueError("qualified routes require complete closure output")
        self._validate_qualified_output()

    def _validate_link_progression(self) -> None:
        diagnostics = self.link_diagnostics
        if type(diagnostics) is not tuple or len(diagnostics) != self.point_count:
            raise ValueError("link_diagnostics must contain exactly point_count links")
        for index, diagnostic in enumerate(diagnostics):
            if type(diagnostic) is not ClosedLoopPolarFrameTransportLinkDiagnostic:
                raise TypeError("link_diagnostics entries have the wrong type")
            if diagnostic.index != index:
                raise ValueError("link_diagnostics must be ordered by index")
            expected_kind = (
                ClosedLoopPolarFrameTransportLinkKind.CLOSURE
                if index == self.point_count - 1
                else ClosedLoopPolarFrameTransportLinkKind.NEIGHBOR
            )
            if diagnostic.kind is not expected_kind:
                raise ValueError("link diagnostic kind is inconsistent with index")
            if diagnostic.singular_values is not None:
                if len(diagnostic.singular_values) != self.rank:
                    raise ValueError("link singular-value count must equal route rank")
                expected_cutoff = ComplexUnitaryProcrustesSolver.rank_cutoff(
                    self.definition.tolerance,
                    diagnostic.singular_values[0],
                )
                if diagnostic.applied_rank_cutoff != expected_cutoff:
                    raise ValueError("link applied rank cutoff is inconsistent")
        if self.status is ClosedLoopPolarFrameTransportStatus.QUALIFIED:
            if any(
                item.status is not ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED
                for item in diagnostics
            ):
                raise ValueError("qualified routes require every link to qualify")
            if self.deficient_link_indices != ():
                raise ValueError("qualified routes have no deficient links")
            if self.unattempted_link_indices != ():
                raise ValueError("qualified routes have no unattempted links")
            return
        deficient = tuple(
            item.index
            for item in diagnostics
            if item.status is ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT
        )
        if len(deficient) != 1:
            raise ValueError("rank-deficient routes require exactly one deficient link")
        deficient_index = deficient[0]
        for index, diagnostic in enumerate(diagnostics):
            expected_status = ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED
            if index == deficient_index:
                expected_status = ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT
            elif index > deficient_index:
                expected_status = ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED
            if diagnostic.status is not expected_status:
                raise ValueError("link progression is inconsistent")
        expected_unattempted = tuple(range(deficient_index + 1, self.point_count))
        if self.deficient_link_indices != (deficient_index,):
            raise ValueError("deficient_link_indices are inconsistent")
        if self.unattempted_link_indices != expected_unattempted:
            raise ValueError("unattempted_link_indices are inconsistent")

    def _validate_qualified_output(self) -> None:
        transported = self.transported_frames
        actions = self.frame_right_actions
        closure_action = self.closure_alignment_right_action
        endpoint_mismatch = self.closure_endpoint_mismatch
        correction_step = self.distributed_closure_correction_step
        schur_vectors = self.closure_schur_vectors
        phases = self.ordered_closure_endpoint_mismatch_phases
        if (
            transported is None
            or actions is None
            or closure_action is None
            or endpoint_mismatch is None
            or correction_step is None
            or schur_vectors is None
            or phases is None
        ):
            raise ValueError("qualified routes require complete array output")
        transported = cast(
            ClosedLoopPolarFrameInventory,
            validate_immutable_array(
                transported,
                "transported_frames",
                np.dtype(np.complex128),
            ),
        )
        actions = cast(
            ClosedLoopPolarFrameRightActions,
            validate_immutable_array(
                actions,
                "frame_right_actions",
                np.dtype(np.complex128),
            ),
        )
        closure_action = cast(
            Complex128Matrix,
            validate_immutable_array(
                closure_action,
                "closure_alignment_right_action",
                np.dtype(np.complex128),
            ),
        )
        endpoint_mismatch = cast(
            Complex128Matrix,
            validate_immutable_array(
                endpoint_mismatch,
                "closure_endpoint_mismatch",
                np.dtype(np.complex128),
            ),
        )
        correction_step = cast(
            Complex128Matrix,
            validate_immutable_array(
                correction_step,
                "distributed_closure_correction_step",
                np.dtype(np.complex128),
            ),
        )
        schur_vectors = cast(
            Complex128Matrix,
            validate_immutable_array(
                schur_vectors,
                "closure_schur_vectors",
                np.dtype(np.complex128),
            ),
        )
        if transported.shape != self.definition.frames.shape:
            raise ValueError("transported_frames shape is inconsistent")
        action_shape = (self.point_count, self.rank, self.rank)
        if actions.shape != action_shape:
            raise ValueError("frame_right_actions shape is inconsistent")
        matrix_shape = (self.rank, self.rank)
        for name, matrix in (
            ("closure_alignment_right_action", closure_action),
            ("closure_endpoint_mismatch", endpoint_mismatch),
            ("distributed_closure_correction_step", correction_step),
            ("closure_schur_vectors", schur_vectors),
        ):
            if matrix.shape != matrix_shape:
                raise ValueError(f"{name} shape is inconsistent")
        self._validate_phases(phases)
        schur_identifier = self.schur_backend_identifier
        schur_version = self.schur_backend_version
        if schur_identifier is None or schur_version is None:
            raise ValueError("qualified routes require Schur backend evidence")
        text(schur_identifier, "schur_backend_identifier")
        text(schur_version, "schur_backend_version")
        self._validate_closure_matrices(
            closure_action,
            endpoint_mismatch,
            correction_step,
            schur_vectors,
            phases,
        )
        self._validate_transported_frames(
            transported,
            actions,
            closure_action,
            schur_vectors,
            phases,
        )

    def _validate_phases(self, phases: tuple[float, ...]) -> None:
        if type(phases) is not tuple or len(phases) != self.rank:
            raise ValueError(
                "ordered_closure_endpoint_mismatch_phases length must equal rank"
            )
        if any(
            type(phase) is not float or not math.isfinite(phase) for phase in phases
        ):
            raise ValueError(
                "ordered_closure_endpoint_mismatch_phases must contain finite floats"
            )
        branch = self.definition.principal_phase_branch
        if branch is (
            ClosedLoopPolarFrameTransportPrincipalPhaseBranch.negative_pi_open
        ):
            phase_outside_branch = any(
                phase <= -math.pi or phase > math.pi for phase in phases
            )
            branch_text = "(-pi, pi]"
        else:
            phase_outside_branch = any(
                phase < -math.pi or phase >= math.pi for phase in phases
            )
            branch_text = "[-pi, pi)"
        if phase_outside_branch:
            raise ValueError(
                "ordered_closure_endpoint_mismatch_phases must be finite phases in "
                f"{branch_text}"
            )
        if phases != tuple(sorted(phases)):
            raise ValueError(
                "ordered_closure_endpoint_mismatch_phases must use ascending "
                "storage order"
            )

    def _validate_closure_matrices(
        self,
        closure_action: Complex128Matrix,
        endpoint_mismatch: Complex128Matrix,
        correction_step: Complex128Matrix,
        schur_vectors: Complex128Matrix,
        phases: tuple[float, ...],
    ) -> None:
        bound = self.definition.maximum_absolute_entry_validation_bound
        if self._maximum_absolute_entry_unitarity_defect(schur_vectors) > bound:
            raise ValueError("closure_schur_vectors violate the unitary bound")
        expected_endpoint_mismatch = self._phase_matrix(
            schur_vectors,
            phases,
            1.0,
        )
        expected_correction_step = self._phase_matrix(
            schur_vectors,
            phases,
            -1.0 / float(self.point_count),
        )
        if (
            self._maximum_absolute_difference(
                endpoint_mismatch,
                expected_endpoint_mismatch,
            )
            > bound
        ):
            raise ValueError(
                "closure_endpoint_mismatch is inconsistent with Schur evidence"
            )
        if (
            self._maximum_absolute_difference(
                correction_step,
                expected_correction_step,
            )
            > bound
        ):
            raise ValueError(
                "distributed_closure_correction_step is inconsistent with Schur "
                "evidence"
            )
        if (
            self._maximum_absolute_difference(
                closure_action,
                endpoint_mismatch.conjugate().T,
            )
            > bound
        ):
            raise ValueError(
                "closure alignment and endpoint-mismatch orientations are inconsistent"
            )
        if self._maximum_absolute_entry_unitarity_defect(closure_action) > bound:
            raise ValueError(
                "closure_alignment_right_action violates the unitary bound"
            )
        if self._maximum_absolute_entry_unitarity_defect(endpoint_mismatch) > bound:
            raise ValueError("closure_endpoint_mismatch violates the unitary bound")
        if self._maximum_absolute_entry_unitarity_defect(correction_step) > bound:
            raise ValueError(
                "distributed_closure_correction_step violates the unitary bound"
            )
        if (
            self._maximum_absolute_difference(
                np.linalg.matrix_power(correction_step, self.point_count),
                closure_action,
            )
            > bound
        ):
            raise ValueError(
                "distributed closure correction does not reconstruct the inverse "
                "endpoint mismatch"
            )
        expected_unitarity = self._maximum_absolute_entry_unitarity_defect(
            closure_action
        )
        reported_unitarity = (
            self.maximum_absolute_entry_closure_alignment_unitarity_defect
        )
        if (
            type(reported_unitarity) is not float
            or not math.isfinite(reported_unitarity)
            or reported_unitarity < 0.0
        ):
            raise ValueError(
                "maximum_absolute_entry_closure_alignment_unitarity_defect must be "
                "finite and nonnegative"
            )
        if reported_unitarity != expected_unitarity:
            raise ValueError(
                "maximum_absolute_entry_closure_alignment_unitarity_defect is "
                "inconsistent"
            )

    def _validate_transported_frames(
        self,
        transported: ClosedLoopPolarFrameInventory,
        actions: ClosedLoopPolarFrameRightActions,
        closure_action: Complex128Matrix,
        schur_vectors: Complex128Matrix,
        phases: tuple[float, ...],
    ) -> None:
        bound = self.definition.maximum_absolute_entry_validation_bound
        identity = np.eye(self.rank, dtype=np.complex128)
        action_grams = np.swapaxes(actions.conjugate(), -1, -2) @ actions
        action_defects = np.max(
            np.abs(action_grams - identity),
            axis=(-2, -1),
        )
        if not bool(np.isfinite(action_defects).all()):
            raise FloatingPointError("frame-action validation became nonfinite")
        if bool((action_defects > bound).any()):
            raise ValueError("frame_right_actions violate the unitary bound")
        expected_transported = self.definition.frames @ actions
        if (
            self._maximum_absolute_difference(
                transported,
                expected_transported,
            )
            > bound
        ):
            raise ValueError("transported_frames are inconsistent with right actions")
        primitive_frames: list[Complex128Matrix] = []
        for index in range(self.point_count):
            correction = self._phase_matrix(
                schur_vectors,
                phases,
                -float(index) / float(self.point_count),
            )
            primitive_action = actions[index] @ correction.conjugate().T
            primitive_frames.append(
                cast(
                    Complex128Matrix,
                    self.definition.frames[index] @ primitive_action,
                )
            )
            if index > 0:
                diagnostic = self.link_diagnostics[index - 1]
                expected_defect = self._maximum_absolute_entry_unitarity_defect(
                    primitive_action
                )
                reported_defect = diagnostic.maximum_absolute_entry_unitarity_defect
                if (
                    reported_defect is None
                    or abs(reported_defect - expected_defect) > bound
                ):
                    raise ValueError(
                        "neighbor maximum-absolute-entry unitary defect is inconsistent"
                    )
        closure_diagnostic = self.link_diagnostics[-1]
        reported_closure_link_defect = (
            closure_diagnostic.maximum_absolute_entry_unitarity_defect
        )
        if (
            reported_closure_link_defect is None
            or abs(
                reported_closure_link_defect
                - self._maximum_absolute_entry_unitarity_defect(closure_action)
            )
            > bound
        ):
            raise ValueError(
                "closure maximum-absolute-entry unitary defect is inconsistent"
            )
        sewn_first = self.definition.endpoint_sewing_map @ primitive_frames[0]
        expected_closure_residual = self._maximum_absolute_difference(
            primitive_frames[-1] @ closure_action,
            sewn_first,
        )
        reported_closure_residual = (
            self.maximum_absolute_entry_closure_alignment_residual
        )
        if (
            type(reported_closure_residual) is not float
            or not math.isfinite(reported_closure_residual)
            or reported_closure_residual < 0.0
        ):
            raise ValueError(
                "maximum_absolute_entry_closure_alignment_residual must be finite "
                "and nonnegative"
            )
        if abs(reported_closure_residual - expected_closure_residual) > bound:
            raise ValueError(
                "maximum_absolute_entry_closure_alignment_residual is inconsistent"
            )

    def _phase_matrix(
        self,
        schur_vectors: Complex128Matrix,
        phases: tuple[float, ...],
        fraction: float,
    ) -> Complex128Matrix:
        phase_array = cast(Float64Array, np.asarray(phases, dtype=np.float64))
        matrix = (
            schur_vectors
            @ np.diag(np.exp(1j * phase_array * fraction))
            @ schur_vectors.conjugate().T
        )
        normalized = cast(Complex128Matrix, np.asarray(matrix, dtype=np.complex128))
        if not bool(np.isfinite(normalized).all()):
            raise FloatingPointError("phase-matrix arithmetic became nonfinite")
        return normalized

    def _maximum_absolute_entry_unitarity_defect(
        self,
        matrix: Complex128Matrix,
    ) -> float:
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        return self._maximum_absolute_difference(
            matrix.conjugate().T @ matrix,
            identity,
        )

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

    @property
    def point_count(self) -> int:
        """Return the definition point count."""
        return self.definition.point_count

    @property
    def ambient_dimension(self) -> int:
        """Return the definition ambient dimension."""
        return self.definition.ambient_dimension

    @property
    def rank(self) -> int:
        """Return the definition frame rank."""
        return self.definition.rank


__all__ = [
    "ClosedLoopPolarFrameInventory",
    "ClosedLoopPolarFrameRightActions",
    "NumpyClosedLoopPolarFrameTransportDefinition",
    "NumpyClosedLoopPolarFrameTransportResult",
]
