"""Unit verification for exact-array closed-loop transport definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch as PhaseBranch,
)
from projectkoios.optimization.alignment.numpy.closed_loop_polar_transport import (
    NumpyClosedLoopPolarFrameTransportDefinition,
)

pytestmark = pytest.mark.unit

_PHASE_BRANCH = PhaseBranch.negative_pi_open


def _scalar_frames(phases: tuple[float, ...]) -> np.ndarray:
    frames = np.zeros((len(phases), 2, 1), dtype=np.complex128)
    frames[:, 0, 0] = np.exp(1j * np.asarray(phases, dtype=np.float64))
    return frames


def test_definition_defensively_owns_exact_immutable_frame_inventory() -> None:
    frames = _scalar_frames((0.0, 0.2))
    sewing = np.eye(2, dtype=np.complex128)

    definition = NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
        frames,
        sewing,
        _PHASE_BRANCH,
    )
    frames[0, 0, 0] = 0.0
    sewing[0, 0] = 0.0

    assert definition.frames[0, 0, 0] == 1.0 + 0.0j
    assert definition.endpoint_sewing_map[0, 0] == 1.0 + 0.0j
    with pytest.raises(ValueError, match="read-only"):
        definition.frames[0, 0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        definition.tolerance = definition.tolerance  # type: ignore[misc]


def test_definition_requires_explicit_principal_phase_branch() -> None:
    with pytest.raises(TypeError, match="principal_phase_branch"):
        NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(  # type: ignore[call-arg]
            _scalar_frames((0.0,)),
            np.eye(2, dtype=np.complex128),
        )


@pytest.mark.parametrize(
    ("frames", "sewing", "error", "match"),
    [
        pytest.param(
            np.ones((2, 2, 1), dtype=np.complex64),
            np.eye(2, dtype=np.complex128),
            TypeError,
            "complex128",
            id="frame-dtype",
        ),
        pytest.param(
            np.ones((2, 2), dtype=np.complex128),
            np.eye(2, dtype=np.complex128),
            ValueError,
            "shape",
            id="frame-shape",
        ),
        pytest.param(
            np.ones((2, 1, 2), dtype=np.complex128),
            np.eye(1, dtype=np.complex128),
            ValueError,
            "rank <= ambient",
            id="rank-bound",
        ),
        pytest.param(
            np.ones((2, 2, 1), dtype=np.complex128),
            np.eye(3, dtype=np.complex128),
            ValueError,
            "endpoint_sewing_map shape",
            id="sewing-shape",
        ),
        pytest.param(
            2.0 * np.ones((2, 2, 1), dtype=np.complex128),
            np.eye(2, dtype=np.complex128),
            ValueError,
            "orthonormality",
            id="frame-orthonormality",
        ),
    ],
)
def test_definition_rejects_malformed_inventory(
    frames: np.ndarray,
    sewing: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        NumpyClosedLoopPolarFrameTransportDefinition.from_arrays(
            frames,
            sewing,
            _PHASE_BRANCH,
        )
