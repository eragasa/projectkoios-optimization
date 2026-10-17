from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import projectkoios.optimization.alignment.numpy.batched_procrustes as facade
from projectkoios.optimization.alignment.numpy.batched_procrustes.numerics import (
    ResidualReconstruction,
)
from tests.projectkoios.optimization.alignment.fixtures import (
    BatchedProcrustesEvidenceFixture,
)


@pytest.mark.parametrize(
    ("source", "target", "spectra", "expected"),
    [
        pytest.param(
            np.array([4.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            id="positive-residual",
        ),
        pytest.param(
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            id="multidimensional-zero-scale",
        ),
        pytest.param(
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array(
                [[np.nextafter(np.float64(1.0), np.float64(0.0))]],
                dtype=np.float64,
            ),
            np.array([0.0], dtype=np.float64),
            id="cancellation-clamped-to-zero",
        ),
    ],
)
def test_reconstruction_returns_expected_float64_evidence(
    source: np.ndarray,
    target: np.ndarray,
    spectra: np.ndarray,
    expected: np.ndarray,
) -> None:
    reconstruction = ResidualReconstruction.from_procrustes_evidence(
        source,
        target,
        spectra,
    )

    np.testing.assert_array_equal(reconstruction.squared_residuals, expected)
    assert reconstruction.squared_residuals.shape == source.shape
    assert reconstruction.roundoff_cutoffs.shape == source.shape
    assert reconstruction.squared_residuals.dtype == np.dtype(np.float64)
    assert reconstruction.roundoff_cutoffs.dtype == np.dtype(np.float64)
    assert bool((reconstruction.roundoff_cutoffs >= 0.0).all())


@pytest.mark.parametrize(
    ("source", "target", "spectra", "match"),
    [
        pytest.param(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float64),
            r"batch index \(0, 1\)",
            id="first-materially-negative-c-order-index",
        ),
        pytest.param(
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([[np.finfo(np.float64).max]], dtype=np.float64),
            "cross nuclear-norm contribution became nonfinite",
            id="nonfinite-nuclear-norm-contribution",
        ),
    ],
)
def test_reconstruction_rejects_invalid_arithmetic(
    source: np.ndarray,
    target: np.ndarray,
    spectra: np.ndarray,
    match: str,
) -> None:
    with (
        np.errstate(over="ignore"),
        pytest.raises(FloatingPointError, match=match),
    ):
        ResidualReconstruction.from_procrustes_evidence(source, target, spectra)


@pytest.fixture
def zero_reconstruction(
    procrustes_evidence: BatchedProcrustesEvidenceFixture,
) -> ResidualReconstruction:
    """Return a valid exact-zero reconstruction from shared evidence."""
    return ResidualReconstruction.from_procrustes_evidence(
        procrustes_evidence.source_powers,
        procrustes_evidence.target_powers,
        procrustes_evidence.spectra,
    )


def test_reconstruction_has_frozen_slots_and_is_not_a_facade_export(
    zero_reconstruction: ResidualReconstruction,
) -> None:
    assert not hasattr(zero_reconstruction, "__dict__")
    assert not hasattr(facade, "ResidualReconstruction")
    with pytest.raises(FrozenInstanceError):
        zero_reconstruction.squared_residuals = np.array(  # type: ignore[misc]
            [1.0], dtype=np.float64
        )
