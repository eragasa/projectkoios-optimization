"""Unit verification for exact-array closed-loop transport results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportLinkDiagnostic,
    ClosedLoopPolarFrameTransportLinkKind,
    ClosedLoopPolarFrameTransportLinkStatus,
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
    ClosedLoopPolarFrameTransportStatus,
)
from projectkoios.optimization.alignment.numpy.closed_loop_polar_transport import (
    NumpyClosedLoopPolarFrameTransportDefinition,
    NumpyClosedLoopPolarFrameTransportResult,
)

pytestmark = pytest.mark.unit


def _definition(
    rank: int = 1,
    point_count: int = 1,
    sewing_sign: float = 1.0,
    principal_phase_branch: ClosedLoopPolarFrameTransportPrincipalPhaseBranch = (
        ClosedLoopPolarFrameTransportPrincipalPhaseBranch.negative_pi_open
    ),
) -> NumpyClosedLoopPolarFrameTransportDefinition:
    frames = np.broadcast_to(
        np.eye(rank, dtype=np.complex128),
        (point_count, rank, rank),
    ).copy()
    sewing = sewing_sign * np.eye(rank, dtype=np.complex128)
    return NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        sewing,
        principal_phase_branch,
    )


def _qualified_link(
    rank: int,
    index: int,
    kind: ClosedLoopPolarFrameTransportLinkKind,
) -> ClosedLoopPolarFrameTransportLinkDiagnostic:
    singular_values = tuple(1.0 for _ in range(rank))
    cutoff = 1.0e-12
    return ClosedLoopPolarFrameTransportLinkDiagnostic(
        index=index,
        kind=kind,
        status=ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED,
        singular_values=singular_values,
        applied_rank_cutoff=cutoff,
        rank_margin=1.0 - cutoff,
        numerical_rank=rank,
        maximum_absolute_entry_unitarity_defect=0.0,
    )


def _qualified_result(
    definition: NumpyClosedLoopPolarFrameTransportDefinition,
    endpoint_mismatch: np.ndarray,
    correction_step: np.ndarray,
    phases: tuple[float, ...],
) -> NumpyClosedLoopPolarFrameTransportResult:
    actions = np.asarray(
        [
            np.linalg.matrix_power(correction_step, index)
            for index in range(definition.point_count)
        ],
        dtype=np.complex128,
    )
    transported = definition.frames @ actions
    closure_action = endpoint_mismatch.conjugate().T
    closure_residual = float(
        np.max(
            np.abs(
                definition.frames[-1] @ closure_action
                - definition.endpoint_sewing_map @ definition.frames[0]
            )
        )
    )
    closure_unitarity = float(
        np.max(
            np.abs(
                closure_action.conjugate().T @ closure_action
                - np.eye(definition.rank, dtype=np.complex128)
            )
        )
    )
    diagnostics = tuple(
        _qualified_link(
            definition.rank,
            index,
            (
                ClosedLoopPolarFrameTransportLinkKind.CLOSURE
                if index == definition.point_count - 1
                else ClosedLoopPolarFrameTransportLinkKind.NEIGHBOR
            ),
        )
        for index in range(definition.point_count)
    )
    return NumpyClosedLoopPolarFrameTransportResult.from_arrays(
        definition=definition,
        status=ClosedLoopPolarFrameTransportStatus.QUALIFIED,
        link_diagnostics=diagnostics,
        deficient_link_indices=(),
        unattempted_link_indices=(),
        transported_frames=transported,
        frame_right_actions=actions,
        closure_alignment_right_action=closure_action,
        closure_endpoint_mismatch=endpoint_mismatch,
        distributed_closure_correction_step=correction_step,
        closure_schur_vectors=np.eye(definition.rank, dtype=np.complex128),
        maximum_absolute_entry_closure_alignment_residual=closure_residual,
        maximum_absolute_entry_closure_alignment_unitarity_defect=(closure_unitarity),
        ordered_closure_endpoint_mismatch_phases=phases,
        svd_backend_identifier="test.svd",
        svd_backend_version="1",
        schur_backend_identifier="test.schur",
        schur_backend_version="1",
    )


def test_qualified_result_defensively_owns_correlated_arrays() -> None:
    definition = _definition()
    endpoint_mismatch = np.eye(1, dtype=np.complex128)
    correction_step = np.eye(1, dtype=np.complex128)

    result = _qualified_result(
        definition,
        endpoint_mismatch,
        correction_step,
        (0.0,),
    )
    endpoint_mismatch[0, 0] = 0.0
    correction_step[0, 0] = 0.0

    assert result.closure_endpoint_mismatch is not None
    assert result.distributed_closure_correction_step is not None
    assert result.closure_endpoint_mismatch[0, 0] == 1.0 + 0.0j
    assert result.distributed_closure_correction_step[0, 0] == 1.0 + 0.0j
    with pytest.raises(ValueError, match="read-only"):
        result.closure_endpoint_mismatch[0, 0] = 0.0


def test_result_rejects_root_that_does_not_reconstruct_inverse_mismatch() -> None:
    definition = _definition(sewing_sign=-1.0)

    with pytest.raises(ValueError, match="distributed_closure_correction_step"):
        _qualified_result(
            definition,
            -np.eye(1, dtype=np.complex128),
            np.eye(1, dtype=np.complex128),
            (math.pi,),
        )


def test_result_accepts_positive_pi_in_negative_open_positive_closed_branch() -> None:
    definition = _definition(sewing_sign=-1.0)

    result = _qualified_result(
        definition,
        -np.eye(1, dtype=np.complex128),
        -np.eye(1, dtype=np.complex128),
        (math.pi,),
    )

    assert result.ordered_closure_endpoint_mismatch_phases == (math.pi,)


def test_result_rejects_negative_pi_in_negative_open_positive_closed_branch() -> None:
    definition = _definition(sewing_sign=-1.0)

    with pytest.raises(ValueError, match=r"\(-pi, pi\]"):
        _qualified_result(
            definition,
            -np.eye(1, dtype=np.complex128),
            -np.eye(1, dtype=np.complex128),
            (-math.pi,),
        )


def test_result_accepts_negative_pi_in_negative_closed_positive_open_branch() -> None:
    definition = _definition(
        sewing_sign=-1.0,
        principal_phase_branch=(
            ClosedLoopPolarFrameTransportPrincipalPhaseBranch.positive_pi_open
        ),
    )

    result = _qualified_result(
        definition,
        -np.eye(1, dtype=np.complex128),
        -np.eye(1, dtype=np.complex128),
        (-math.pi,),
    )

    assert result.ordered_closure_endpoint_mismatch_phases == (-math.pi,)


def test_result_rejects_link_spectrum_length_different_from_rank() -> None:
    definition = _definition(rank=2)
    diagnostic = ClosedLoopPolarFrameTransportLinkDiagnostic(
        index=0,
        kind=ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
        status=ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT,
        singular_values=(0.0,),
        applied_rank_cutoff=0.0,
        rank_margin=0.0,
        numerical_rank=0,
        maximum_absolute_entry_unitarity_defect=None,
    )

    with pytest.raises(ValueError, match="count must equal route rank"):
        NumpyClosedLoopPolarFrameTransportResult(
            definition=definition,
            status=ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT,
            link_diagnostics=(diagnostic,),
            deficient_link_indices=(0,),
            unattempted_link_indices=(),
            transported_frames=None,
            frame_right_actions=None,
            closure_alignment_right_action=None,
            closure_endpoint_mismatch=None,
            distributed_closure_correction_step=None,
            closure_schur_vectors=None,
            maximum_absolute_entry_closure_alignment_residual=None,
            maximum_absolute_entry_closure_alignment_unitarity_defect=None,
            ordered_closure_endpoint_mismatch_phases=None,
            svd_backend_identifier="test.svd",
            svd_backend_version="1",
            schur_backend_identifier=None,
            schur_backend_version=None,
        )


def test_result_rejects_unattempted_link_before_deficiency() -> None:
    definition = _definition(point_count=2)
    unattempted = ClosedLoopPolarFrameTransportLinkDiagnostic(
        0,
        ClosedLoopPolarFrameTransportLinkKind.NEIGHBOR,
        ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED,
        None,
        None,
        None,
        None,
        None,
    )
    deficient = ClosedLoopPolarFrameTransportLinkDiagnostic(
        1,
        ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
        ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT,
        (0.0,),
        0.0,
        0.0,
        0,
        None,
    )

    with pytest.raises(ValueError, match="progression"):
        NumpyClosedLoopPolarFrameTransportResult(
            definition=definition,
            status=ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT,
            link_diagnostics=(unattempted, deficient),
            deficient_link_indices=(1,),
            unattempted_link_indices=(),
            transported_frames=None,
            frame_right_actions=None,
            closure_alignment_right_action=None,
            closure_endpoint_mismatch=None,
            distributed_closure_correction_step=None,
            closure_schur_vectors=None,
            maximum_absolute_entry_closure_alignment_residual=None,
            maximum_absolute_entry_closure_alignment_unitarity_defect=None,
            ordered_closure_endpoint_mismatch_phases=None,
            svd_backend_identifier="test.svd",
            svd_backend_version="1",
            schur_backend_identifier=None,
            schur_backend_version=None,
        )
