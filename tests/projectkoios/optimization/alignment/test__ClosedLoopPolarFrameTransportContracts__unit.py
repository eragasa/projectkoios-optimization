"""Unit verification for dependency-free closed-loop transport contracts."""

from __future__ import annotations

import pytest

from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportBackendFailure,
    ClosedLoopPolarFrameTransportBackendFailureReason,
    ClosedLoopPolarFrameTransportBackendOperation,
    ClosedLoopPolarFrameTransportLinkDiagnostic,
    ClosedLoopPolarFrameTransportLinkKind,
    ClosedLoopPolarFrameTransportLinkStatus,
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
)

pytestmark = pytest.mark.unit


def test_link_diagnostic_retains_ordered_full_rank_evidence() -> None:
    diagnostic = ClosedLoopPolarFrameTransportLinkDiagnostic(
        index=0,
        kind=ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
        status=ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED,
        singular_values=(1.0, 0.5),
        applied_rank_cutoff=0.1,
        rank_margin=0.4,
        numerical_rank=2,
        maximum_absolute_entry_unitarity_defect=0.0,
    )

    assert diagnostic.singular_values == (1.0, 0.5)
    assert diagnostic.numerical_rank == 2


def test_link_diagnostic_rejects_increasing_singular_spectrum() -> None:
    with pytest.raises(ValueError, match="nonincreasing"):
        ClosedLoopPolarFrameTransportLinkDiagnostic(
            index=0,
            kind=ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
            status=ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED,
            singular_values=(0.5, 1.0),
            applied_rank_cutoff=0.1,
            rank_margin=0.9,
            numerical_rank=2,
            maximum_absolute_entry_unitarity_defect=0.0,
        )


def test_unattempted_link_cannot_claim_numerical_evidence() -> None:
    with pytest.raises(ValueError, match="must not claim"):
        ClosedLoopPolarFrameTransportLinkDiagnostic(
            index=1,
            kind=ClosedLoopPolarFrameTransportLinkKind.CLOSURE,
            status=ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED,
            singular_values=(0.0,),
            applied_rank_cutoff=None,
            rank_margin=None,
            numerical_rank=None,
            maximum_absolute_entry_unitarity_defect=None,
        )


def test_principal_phase_branches_are_explicit_dependency_free_conventions() -> None:
    assert (
        ClosedLoopPolarFrameTransportPrincipalPhaseBranch.negative_pi_open.value
        == "(-pi, pi]"
    )
    assert (
        ClosedLoopPolarFrameTransportPrincipalPhaseBranch.positive_pi_open.value
        == "[-pi, pi)"
    )


def test_backend_failure_retains_typed_operation_reason_and_link() -> None:
    failure = ClosedLoopPolarFrameTransportBackendFailure(
        ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD,
        ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE,
        3,
    )

    assert (
        failure.operation
        is ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD
    )
    assert (
        failure.reason
        is ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE
    )
    assert failure.link_index == 3
