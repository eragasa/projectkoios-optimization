"""Integration verification for closed-loop polar frame transport."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportBackendFailure,
    ClosedLoopPolarFrameTransportBackendFailureReason,
    ClosedLoopPolarFrameTransportBackendOperation,
    ClosedLoopPolarFrameTransportLinkKind,
    ClosedLoopPolarFrameTransportLinkStatus,
    ClosedLoopPolarFrameTransportStatus,
)
from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch as PhaseBranch,
)
from projectkoios.optimization.alignment.numpy.closed_loop_polar_transport import (
    NumpyClosedLoopPolarFrameTransportDefinition,
)
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesTolerance,
)
from projectkoios.optimization.alignment.scipy import (
    closed_loop_polar_transport as transport_module,
)
from projectkoios.optimization.alignment.scipy.closed_loop_polar_transport import (
    ScipyClosedLoopPolarFrameTransportSolver,
)

pytestmark = pytest.mark.integration

_PHASE_BRANCH = PhaseBranch.negative_pi_open


def _scalar_frames(phases: tuple[float, ...]) -> np.ndarray:
    frames = np.zeros((len(phases), 2, 1), dtype=np.complex128)
    frames[:, 0, 0] = np.exp(1j * np.asarray(phases, dtype=np.float64))
    return frames


def test_scalar_transport_removes_local_phases_and_distributes_sewn_closure() -> None:
    frames = _scalar_frames((0.2, -1.1, 2.4, -0.7))
    closure_phase = 0.8
    sewing = np.diag(np.asarray([np.exp(1j * closure_phase), 1.0], dtype=np.complex128))
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        sewing,
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.status is ClosedLoopPolarFrameTransportStatus.QUALIFIED
    assert result.deficient_link_indices == ()
    assert result.unattempted_link_indices == ()
    assert result.transported_frames is not None
    assert result.closure_alignment_right_action is not None
    assert result.closure_endpoint_mismatch is not None
    assert result.distributed_closure_correction_step is not None
    assert result.ordered_closure_endpoint_mismatch_phases == pytest.approx(
        (-closure_phase,)
    )
    np.testing.assert_allclose(
        result.closure_alignment_right_action,
        [[np.exp(1j * closure_phase)]],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.closure_endpoint_mismatch,
        [[np.exp(-1j * closure_phase)]],
        atol=1.0e-14,
    )
    expected = np.asarray(
        [
            [
                [
                    np.exp(
                        1j
                        * (0.2 + closure_phase * float(index) / float(frames.shape[0]))
                    )
                ],
                [0.0],
            ]
            for index in range(frames.shape[0])
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(result.transported_frames, expected, atol=1.0e-14)
    for diagnostic in result.link_diagnostics:
        assert diagnostic.status is ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED
        assert diagnostic.singular_values == pytest.approx((1.0,))
        assert diagnostic.numerical_rank == 1
        assert diagnostic.maximum_absolute_entry_unitarity_defect is not None
        assert diagnostic.maximum_absolute_entry_unitarity_defect <= 1.0e-14
    assert (
        result.link_diagnostics[-1].kind
        is ClosedLoopPolarFrameTransportLinkKind.CLOSURE
    )
    assert result.maximum_absolute_entry_closure_alignment_residual is not None
    assert result.maximum_absolute_entry_closure_alignment_residual <= 1.0e-14
    assert result.maximum_absolute_entry_closure_alignment_unitarity_defect is not None
    assert result.maximum_absolute_entry_closure_alignment_unitarity_defect <= 1.0e-14


def test_composite_transport_resolves_noncommuting_local_right_gauges() -> None:
    first = np.eye(2, dtype=np.complex128)
    second = np.asarray(
        [[0.0, np.exp(0.3j)], [np.exp(-0.2j), 0.0]],
        dtype=np.complex128,
    )
    angle = 0.4
    third = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.complex128,
    )
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        np.asarray([first, second, third], dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.transported_frames is not None
    for frame in result.transported_frames:
        np.testing.assert_allclose(frame, first, atol=1.0e-14)
    assert result.ordered_closure_endpoint_mismatch_phases is not None
    np.testing.assert_allclose(
        np.asarray(result.ordered_closure_endpoint_mismatch_phases),
        [0.0, 0.0],
        atol=1.0e-14,
    )


def test_composite_transport_distributes_nondiagonal_holonomy_directly() -> None:
    point_count = 3
    frames = np.broadcast_to(
        np.eye(2, dtype=np.complex128),
        (point_count, 2, 2),
    ).copy()
    mixing = np.asarray(
        [[1.0, 1.0], [1.0j, -1.0j]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    phases = np.asarray([-0.7, 1.1], dtype=np.float64)
    sewing = mixing @ np.diag(np.exp(1j * phases)) @ mixing.conjugate().T
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        sewing,
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.transported_frames is not None
    assert result.closure_alignment_right_action is not None
    assert result.closure_endpoint_mismatch is not None
    assert result.distributed_closure_correction_step is not None
    np.testing.assert_allclose(
        result.closure_alignment_right_action,
        sewing,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.closure_endpoint_mismatch,
        sewing.conjugate().T,
        atol=1.0e-14,
    )
    assert result.ordered_closure_endpoint_mismatch_phases is not None
    np.testing.assert_allclose(
        np.asarray(result.ordered_closure_endpoint_mismatch_phases),
        np.sort(-phases),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.linalg.matrix_power(
            result.distributed_closure_correction_step,
            point_count,
        ),
        result.closure_alignment_right_action,
        atol=1.0e-14,
    )
    for index in range(point_count):
        np.testing.assert_allclose(
            result.transported_frames[index],
            np.linalg.matrix_power(
                result.distributed_closure_correction_step,
                index,
            ),
            atol=1.0e-14,
        )
    closure_overlap = (
        result.transported_frames[-1].conjugate().T
        @ sewing
        @ result.transported_frames[0]
    )
    np.testing.assert_allclose(
        closure_overlap,
        result.distributed_closure_correction_step,
        atol=1.0e-14,
    )


def test_neighbor_rank_deficiency_stops_progression_without_completion() -> None:
    frames = np.asarray(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
            [[0.0], [1.0]],
        ],
        dtype=np.complex128,
    )
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.status is ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT
    assert result.deficient_link_indices == (0,)
    assert result.unattempted_link_indices == (1, 2)
    assert result.link_diagnostics[0].singular_values == (0.0,)
    assert result.link_diagnostics[0].maximum_absolute_entry_unitarity_defect is None
    assert all(
        item.status is ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED
        for item in result.link_diagnostics[1:]
    )
    assert result.transported_frames is None
    assert result.closure_alignment_right_action is None
    assert result.closure_endpoint_mismatch is None
    assert result.ordered_closure_endpoint_mismatch_phases is None


def test_rank_cutoff_equality_stops_before_dependent_links() -> None:
    frames = np.asarray(
        [
            [[1.0], [0.0]],
            [[0.5], [np.sqrt(0.75)]],
        ],
        dtype=np.complex128,
    )
    tolerance = ComplexUnitaryProcrustesTolerance(
        rank_absolute=0.5,
        rank_relative=0.0,
        validation_absolute=1.0e-12,
        validation_relative=1.0e-10,
    )
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
        tolerance,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.deficient_link_indices == (0,)
    assert result.unattempted_link_indices == (1,)
    assert result.link_diagnostics[0].singular_values == (0.5,)
    assert result.link_diagnostics[0].applied_rank_cutoff == 0.5
    assert result.link_diagnostics[0].rank_margin == 0.0


def test_closure_rank_deficiency_retains_all_attempted_link_evidence() -> None:
    frames = _scalar_frames((0.0, 0.0))
    sewing = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        sewing,
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.status is ClosedLoopPolarFrameTransportStatus.RANK_DEFICIENT
    assert result.deficient_link_indices == (1,)
    assert result.unattempted_link_indices == ()
    assert (
        result.link_diagnostics[0].status
        is ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED
    )
    assert (
        result.link_diagnostics[1].status
        is ClosedLoopPolarFrameTransportLinkStatus.RANK_DEFICIENT
    )
    assert result.link_diagnostics[1].singular_values == (0.0,)
    assert result.transported_frames is None


def test_closure_alignment_residual_uses_maximum_entry_metric() -> None:
    angle = 0.5
    frame = np.asarray(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=np.complex128,
    )
    sewing = np.asarray(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.complex128,
    )
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frame,
        sewing,
        _PHASE_BRANCH,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    expected_residual = frame[0] - sewing @ frame[0]
    expected_maximum_entry = float(np.max(np.abs(expected_residual)))
    normalized_frobenius = float(
        np.linalg.norm(expected_residual, ord="fro") / np.sqrt(2.0)
    )
    assert result.maximum_absolute_entry_closure_alignment_residual == pytest.approx(
        expected_maximum_entry,
        abs=1.0e-15,
    )
    assert expected_maximum_entry != pytest.approx(
        normalized_frobenius,
        abs=1.0e-3,
    )


def test_svd_nonconvergence_maps_to_typed_link_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_svd(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(np.linalg, "svd", fail_svd)
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        _scalar_frames((0.0, 0.0)),
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    with pytest.raises(ClosedLoopPolarFrameTransportBackendFailure) as captured:
        ScipyClosedLoopPolarFrameTransportSolver(
            NumpyComplexUnitaryProcrustesSolver()
        ).solve(definition)

    assert (
        captured.value.operation
        is ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD
    )
    assert (
        captured.value.reason
        is ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE
    )
    assert captured.value.link_index == 0


def test_closure_svd_nonconvergence_retains_closure_link_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_svd = np.linalg.svd
    call_count = 0

    def fail_second_svd(
        matrix: np.ndarray,
        *,
        full_matrices: bool,
        compute_uv: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("injected closure failure")
        result = real_svd(
            matrix,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        return result.U, result.S, result.Vh

    monkeypatch.setattr(np.linalg, "svd", fail_second_svd)
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        _scalar_frames((0.0, 0.0)),
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    with pytest.raises(ClosedLoopPolarFrameTransportBackendFailure) as captured:
        ScipyClosedLoopPolarFrameTransportSolver(
            NumpyComplexUnitaryProcrustesSolver()
        ).solve(definition)

    assert (
        captured.value.operation
        is ClosedLoopPolarFrameTransportBackendOperation.PROCRUSTES_SVD
    )
    assert (
        captured.value.reason
        is ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE
    )
    assert captured.value.link_index == 1


def test_schur_nonconvergence_maps_to_typed_operation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_schur(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(transport_module, "schur", fail_schur)
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        _scalar_frames((0.0, 0.0)),
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    with pytest.raises(ClosedLoopPolarFrameTransportBackendFailure) as captured:
        ScipyClosedLoopPolarFrameTransportSolver(
            NumpyComplexUnitaryProcrustesSolver()
        ).solve(definition)

    assert (
        captured.value.operation
        is ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR
    )
    assert (
        captured.value.reason
        is ClosedLoopPolarFrameTransportBackendFailureReason.NONCONVERGENCE
    )
    assert captured.value.link_index == 1


def test_invalid_schur_evidence_maps_to_typed_operation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_schur(
        matrix: np.ndarray,
        *,
        output: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        del output
        dimension = matrix.shape[0]
        return (
            2.0 * np.eye(dimension, dtype=np.complex128),
            np.eye(dimension, dtype=np.complex128),
        )

    monkeypatch.setattr(transport_module, "schur", invalid_schur)
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        _scalar_frames((0.0, 0.0)),
        np.eye(2, dtype=np.complex128),
        _PHASE_BRANCH,
    )

    with pytest.raises(ClosedLoopPolarFrameTransportBackendFailure) as captured:
        ScipyClosedLoopPolarFrameTransportSolver(
            NumpyComplexUnitaryProcrustesSolver()
        ).solve(definition)

    assert (
        captured.value.operation
        is ClosedLoopPolarFrameTransportBackendOperation.CLOSURE_SCHUR
    )
    assert (
        captured.value.reason
        is ClosedLoopPolarFrameTransportBackendFailureReason.INVALID_DECOMPOSITION
    )
    assert captured.value.link_index == 1


@pytest.mark.parametrize(
    ("branch", "expected_phase", "expected_step"),
    [
        pytest.param(
            PhaseBranch.negative_pi_open,
            np.pi,
            np.exp(-1j * np.pi / 4.0),
            id="negative-open-positive-closed",
        ),
        pytest.param(
            PhaseBranch.positive_pi_open,
            -np.pi,
            np.exp(1j * np.pi / 4.0),
            id="negative-closed-positive-open",
        ),
    ],
)
def test_negative_one_endpoint_mismatch_uses_selected_branch_and_root(
    branch: PhaseBranch,
    expected_phase: float,
    expected_step: complex,
) -> None:
    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        _scalar_frames((0.0, 0.0, 0.0, 0.0)),
        np.diag(np.asarray([-1.0, 1.0], dtype=np.complex128)),
        branch,
    )

    result = ScipyClosedLoopPolarFrameTransportSolver(
        NumpyComplexUnitaryProcrustesSolver()
    ).solve(definition)

    assert result.ordered_closure_endpoint_mismatch_phases == (expected_phase,)
    assert result.distributed_closure_correction_step is not None
    np.testing.assert_allclose(
        result.distributed_closure_correction_step,
        [[expected_step]],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.linalg.matrix_power(result.distributed_closure_correction_step, 4),
        [[-1.0]],
        atol=1.0e-14,
    )
