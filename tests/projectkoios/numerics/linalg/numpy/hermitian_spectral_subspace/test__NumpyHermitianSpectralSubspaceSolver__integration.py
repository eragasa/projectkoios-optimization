"""Integration tests for dense NumPy Hermitian spectral subspaces."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceTolerance,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectral_subspace import (
    NumpyHermitianSpectralSubspaceSolver,
)

pytestmark = pytest.mark.integration


def _tolerance(
    *,
    hermiticity_absolute: float = 1.0e-12,
    boundary_absolute: float = 1.0e-12,
) -> HermitianSpectralSubspaceTolerance:
    return HermitianSpectralSubspaceTolerance(
        hermiticity_absolute,
        1.0e-14,
        boundary_absolute,
        1.0e-14,
        2.0e-14,
        2.0e-14,
        2.0e-14,
    )


def test_complex_hermitian_matrix_returns_expected_spectrum_and_projector() -> None:
    definition = HermitianSpectralSubspaceDefinition(
        ((2.0 + 0.0j, 0.0 + 1.0j), (0.0 - 1.0j, 2.0 + 0.0j)),
        0,
        1,
        _tolerance(),
    )

    result = NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert result.eigenvalues == pytest.approx((1.0, 3.0), abs=2.0e-15)
    expected = np.array(
        [[0.5 + 0.0j, 0.0 - 0.5j], [0.0 + 0.5j, 0.5 + 0.0j]],
        dtype=np.complex128,
    )
    assert np.asarray(result.projector) == pytest.approx(expected, abs=2.0e-15)
    assert result.lower_boundary_gap is None
    assert result.upper_boundary_gap == pytest.approx(2.0)
    assert result.backend_identifier == "numpy.linalg.eigh"
    assert result.backend_version == np.__version__


def test_approximately_hermitian_input_uses_explicit_shared_representative() -> None:
    matrix = (
        (1.0 + 2.0e-13j, 0.2 + 0.3j),
        (0.2 - 0.3j + 4.0e-13, 3.0 - 1.0e-13j),
    )
    definition = HermitianSpectralSubspaceDefinition(
        matrix,
        0,
        1,
        _tolerance(hermiticity_absolute=1.0e-12),
    )

    result = NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert result.diagnostics.input_hermiticity_maximum_absolute_defect > 0.0
    assert result.diagnostics.representative_adjustment_maximum_absolute_defect > 0.0
    assert (
        result.diagnostics.input_hermiticity_maximum_absolute_defect
        <= result.diagnostics.applied_hermiticity_cutoff
    )
    projector = np.asarray(result.projector)
    assert projector == pytest.approx(projector.conjugate().T, abs=1.0e-14)


def test_complete_range_returns_identity_without_boundary_gaps() -> None:
    definition = HermitianSpectralSubspaceDefinition(
        (
            (2.0 + 0.0j, 0.5 + 0.25j, 0.0 + 0.0j),
            (0.5 - 0.25j, 1.0 + 0.0j, 0.3 + 0.0j),
            (0.0 + 0.0j, 0.3 + 0.0j, 4.0 + 0.0j),
        ),
        0,
        3,
        _tolerance(),
    )

    result = NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert np.asarray(result.projector) == pytest.approx(np.eye(3), abs=1.0e-14)
    assert result.lower_boundary_gap is None
    assert result.upper_boundary_gap is None


def test_every_diagnostic_field_matches_independent_array_recomputation() -> None:
    original = np.array(
        [
            [0.5 + 2.0e-13j, 0.2 + 0.1j, -0.15j],
            [0.2 - 0.1j + 2.0e-13, 2.0 - 1.0e-13j, 0.3 + 0.05j],
            [0.15j, 0.3 - 0.05j - 1.0e-13, 4.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    tolerance = _tolerance(hermiticity_absolute=1.0e-12)
    result = NumpyHermitianSpectralSubspaceSolver().solve(
        HermitianSpectralSubspaceDefinition(_as_tuple(original), 0, 2, tolerance)
    )
    representative = (original + original.conjugate().T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(representative, UPLO="L")
    selected = eigenvectors[:, :2]
    projector = selected @ selected.conjugate().T
    input_scale = float(np.max(np.abs(original)))
    representative_scale = float(np.max(np.abs(representative)))
    hermiticity_cutoff = (
        tolerance.hermiticity_absolute + tolerance.hermiticity_relative * input_scale
    )
    boundary_cutoff = tolerance.boundary_absolute + tolerance.boundary_relative * float(
        eigenvalues[-1] - eigenvalues[0]
    )
    spectral_cutoff = (
        tolerance.spectral_validation_absolute
        + tolerance.spectral_validation_relative * representative_scale
    )
    expected = {
        "input_hermiticity_maximum_absolute_defect": float(
            np.max(np.abs(original - original.conjugate().T))
        ),
        "representative_adjustment_maximum_absolute_defect": float(
            np.max(np.abs(representative - original))
        ),
        "eigenvector_orthonormality_maximum_absolute_defect": float(
            np.max(np.abs(eigenvectors.conjugate().T @ eigenvectors - np.eye(3)))
        ),
        "eigensystem_residual_maximum_absolute_defect": float(
            np.max(
                np.abs(
                    representative @ eigenvectors - eigenvectors @ np.diag(eigenvalues)
                )
            )
        ),
        "representative_reconstruction_maximum_absolute_defect": float(
            np.max(
                np.abs(
                    representative
                    - eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conjugate().T
                )
            )
        ),
        "projector_hermiticity_maximum_absolute_defect": float(
            np.max(np.abs(projector - projector.conjugate().T))
        ),
        "projector_idempotence_maximum_absolute_defect": float(
            np.max(np.abs(projector @ projector - projector))
        ),
        "projector_trace_absolute_defect": float(abs(np.trace(projector) - 2.0)),
        "projector_commutator_maximum_absolute_defect": float(
            np.max(np.abs(representative @ projector - projector @ representative))
        ),
        "selected_spectral_trace_absolute_defect": float(
            abs(np.trace(representative @ projector) - np.sum(eigenvalues[:2]))
        ),
        "full_spectral_trace_absolute_defect": float(
            abs(np.trace(representative) - np.sum(eigenvalues))
        ),
        "input_matrix_maximum_absolute_entry": input_scale,
        "representative_matrix_maximum_absolute_entry": representative_scale,
        "applied_hermiticity_cutoff": hermiticity_cutoff,
        "spectral_diameter": float(eigenvalues[-1] - eigenvalues[0]),
        "applied_boundary_cutoff": boundary_cutoff,
        "applied_projector_validation_cutoff": tolerance.projector_validation_absolute,
        "applied_spectral_validation_cutoff": spectral_cutoff,
    }

    assert set(expected) == {field.name for field in fields(result.diagnostics)}
    for name, expected_value in expected.items():
        assert getattr(result.diagnostics, name) == pytest.approx(
            expected_value, abs=2.0e-14
        )


def _as_tuple(matrix: np.ndarray) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(value) for value in row) for row in matrix)


def test_coordinate_permutation_covariantly_permutates_projector() -> None:
    original = np.array(
        [
            [0.0, 0.2 + 0.1j, -0.3j],
            [0.2 - 0.1j, 2.0, 0.4],
            [0.3j, 0.4, 5.0],
        ],
        dtype=np.complex128,
    )
    permutation = np.eye(3, dtype=np.complex128)[[2, 0, 1]]
    permuted = permutation @ original @ permutation.conjugate().T
    solver = NumpyHermitianSpectralSubspaceSolver()

    first = solver.solve(
        HermitianSpectralSubspaceDefinition(
            tuple(tuple(complex(value) for value in row) for row in original),
            0,
            2,
            _tolerance(),
        )
    )
    second = solver.solve(
        HermitianSpectralSubspaceDefinition(
            tuple(tuple(complex(value) for value in row) for row in permuted),
            0,
            2,
            _tolerance(),
        )
    )

    expected = permutation @ np.asarray(first.projector) @ permutation.conjugate().T
    assert np.asarray(second.projector) == pytest.approx(expected, abs=2.0e-14)


def test_repeated_solves_of_same_definition_are_equal() -> None:
    definition = HermitianSpectralSubspaceDefinition(
        (
            (0.5 + 0.0j, 0.2 + 0.1j, 0.0 + 0.0j),
            (0.2 - 0.1j, 2.0 + 0.0j, 0.3 + 0.0j),
            (0.0 + 0.0j, 0.3 + 0.0j, 4.0 + 0.0j),
        ),
        0,
        2,
        _tolerance(),
    )
    solver = NumpyHermitianSpectralSubspaceSolver()

    assert solver.solve(definition) == solver.solve(definition)


def test_ksd_shaped_lowest_retained_group_returns_full_spectrum_and_projector() -> None:
    # This sanitized unit-free fixture has the same owner-facing shape as KSD M2:
    # one dense complex Hermitian matrix and a lowest retained-rank range [0, r).
    definition = HermitianSpectralSubspaceDefinition(
        (
            (-1.5 + 0.0j, 0.1 + 0.05j, 0.0 + 0.02j, 0.0 + 0.0j),
            (0.1 - 0.05j, -0.5 + 0.0j, 0.08 + 0.0j, 0.0 - 0.01j),
            (0.0 - 0.02j, 0.08 + 0.0j, 1.2 + 0.0j, 0.12 + 0.03j),
            (0.0 + 0.0j, 0.0 + 0.01j, 0.12 - 0.03j, 2.5 + 0.0j),
        ),
        0,
        2,
        _tolerance(),
    )

    result = NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert len(result.eigenvalues) == 4
    assert result.eigenvalues == tuple(sorted(result.eigenvalues))
    assert result.lower_boundary_gap is None
    assert result.upper_boundary_gap is not None
    assert result.upper_boundary_gap > result.diagnostics.applied_boundary_cutoff
    assert np.trace(np.asarray(result.projector)).real == pytest.approx(
        2.0, abs=1.0e-14
    )


def test_numpy_convergence_failure_is_translated_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 0.0j,),),
        0,
        1,
        _tolerance(),
    )

    with pytest.raises(RuntimeError, match="did not converge") as captured:
        NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert isinstance(captured.value.__cause__, np.linalg.LinAlgError)


def test_nonfinite_numpy_output_fails_as_floating_point_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        return np.array([np.nan]), np.array([[1.0 + 0.0j]])

    monkeypatch.setattr(np.linalg, "eigh", nonfinite)
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 0.0j,),),
        0,
        1,
        _tolerance(),
    )

    with pytest.raises(FloatingPointError, match="nonfinite eigenvalues"):
        NumpyHermitianSpectralSubspaceSolver().solve(definition)


def test_undeclared_numpy_backend_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise KeyError("programming defect")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 0.0j,),),
        0,
        1,
        _tolerance(),
    )

    with pytest.raises(KeyError, match="programming defect"):
        NumpyHermitianSpectralSubspaceSolver().solve(definition)
