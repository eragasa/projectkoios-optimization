"""Bounded numerical verification for dense Hermitian spectral subspaces."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, cast

import mpmath
import numpy as np
import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralMatrix,
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceTolerance,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectral_subspace import (
    NumpyHermitianSpectralSubspaceSolver,
)

pytestmark = pytest.mark.numerical


def _exact_mpf(value: float) -> mpmath.mpf:
    numerator, denominator = value.as_integer_ratio()
    return mpmath.mpf(numerator) / denominator


def _mp_matrix(matrix: HermitianSpectralMatrix) -> mpmath.matrix:
    dimension = len(matrix)
    result = mpmath.matrix(dimension, dimension)
    for row in range(dimension):
        for column in range(dimension):
            value = matrix[row][column]
            result[row, column] = mpmath.mpc(
                _exact_mpf(value.real),
                _exact_mpf(value.imag),
            )
    return result


def _mp_maximum_absolute(matrix: mpmath.matrix) -> mpmath.mpf:
    return max(
        (
            abs(matrix[row, column])
            for row in range(matrix.rows)
            for column in range(matrix.cols)
        ),
        default=mpmath.mpf("0"),
    )


def _reference(
    matrix: HermitianSpectralMatrix,
    start: int,
    stop: int,
    dps: int,
) -> tuple[tuple[mpmath.mpf, ...], tuple[tuple[mpmath.mpc, ...], ...]]:
    with mpmath.workdps(dps):
        high_matrix = _mp_matrix(matrix)
        eigenvalues, eigenvectors = mpmath.eigh(high_matrix)
        diagonal = mpmath.diag(eigenvalues)
        identity = mpmath.eye(len(matrix))
        scale = max(
            mpmath.mpf("1"),
            mpmath.sqrt(
                mpmath.fsum(
                    abs(high_matrix[row, column]) ** 2
                    for row in range(len(matrix))
                    for column in range(len(matrix))
                )
            ),
        )
        residual = _mp_maximum_absolute(
            high_matrix * eigenvectors - eigenvectors * diagonal
        )
        orthonormality = _mp_maximum_absolute(eigenvectors.H * eigenvectors - identity)
        reconstruction = _mp_maximum_absolute(
            high_matrix - eigenvectors * diagonal * eigenvectors.H
        )
        assert residual <= mpmath.mpf("1e-80") * scale
        assert orthonormality <= mpmath.mpf("1e-80") * scale
        assert reconstruction <= mpmath.mpf("1e-80") * scale

        selected = eigenvectors[:, start:stop]
        projector = selected * selected.H
        projector_hermiticity = _mp_maximum_absolute(projector - projector.H)
        projector_idempotence = _mp_maximum_absolute(projector * projector - projector)
        projector_rank = abs(
            mpmath.fsum(projector[index, index] for index in range(len(matrix)))
            - (stop - start)
        )
        assert projector_hermiticity <= mpmath.mpf("1e-80")
        assert projector_idempotence <= mpmath.mpf("1e-80")
        assert projector_rank <= mpmath.mpf("1e-80")
        return (
            tuple(+eigenvalues[index] for index in range(len(matrix))),
            tuple(
                tuple(+projector[row, column] for column in range(len(matrix)))
                for row in range(len(matrix))
            ),
        )


def _as_tuple(matrix: np.ndarray) -> HermitianSpectralMatrix:
    return tuple(tuple(complex(value) for value in row) for row in matrix)


def _tolerance(scale: float) -> HermitianSpectralSubspaceTolerance:
    epsilon = math.ulp(1.0)
    absolute = 128.0 * epsilon * max(1.0, scale)
    return HermitianSpectralSubspaceTolerance(
        absolute,
        64.0 * epsilon,
        absolute,
        64.0 * epsilon,
        128.0 * epsilon,
        absolute,
        128.0 * epsilon,
    )


def _frobenius(matrix: HermitianSpectralMatrix) -> float:
    return math.sqrt(math.fsum(abs(value) ** 2 for row in matrix for value in row))


def _projector_error(
    actual: HermitianSpectralMatrix,
    expected: tuple[tuple[mpmath.mpc, ...], ...],
) -> float:
    return math.sqrt(
        math.fsum(
            abs(actual[row][column] - complex(expected[row][column])) ** 2
            for row in range(len(actual))
            for column in range(len(actual))
        )
    )


def test_exact_degenerate_selected_block_matches_analytic_projector() -> None:
    # Every Hadamard entry is +/- 1/2, so the unitary identities, integer-eigenvalue
    # transformation, matrix entries, and selected projector are exactly representable
    # in binary64. This exercises a non-coordinate exact degeneracy through NumPy.
    unitary = 0.5 * np.array(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=np.complex128,
    )
    eigenvalues = np.array((-2.0, 1.0, 1.0, 4.0))
    matrix = unitary @ np.diag(eigenvalues) @ unitary.conjugate().T
    expected = unitary[:, 1:3] @ unitary[:, 1:3].conjugate().T
    definition = HermitianSpectralSubspaceDefinition(
        _as_tuple(matrix),
        1,
        3,
        _tolerance(float(np.max(np.abs(matrix)))),
    )

    result = NumpyHermitianSpectralSubspaceSolver().solve(definition)

    assert result.eigenvalues == pytest.approx(eigenvalues, abs=4.0e-15)
    assert result.eigenvalues[1] == pytest.approx(result.eigenvalues[2], abs=2.0e-15)
    assert np.asarray(result.projector) == pytest.approx(expected, abs=2.0e-14)


def test_shift_and_positive_scaling_preserve_selected_projector() -> None:
    matrix = np.array(
        [[0.5, 0.2 + 0.1j, -0.1j], [0.2 - 0.1j, 2.0, 0.3], [0.1j, 0.3, 4.0]],
        dtype=np.complex128,
    )
    transformed = 3.5 * matrix + 7.0 * np.eye(3)
    solver = NumpyHermitianSpectralSubspaceSolver()

    first = solver.solve(
        HermitianSpectralSubspaceDefinition(
            _as_tuple(matrix),
            0,
            2,
            _tolerance(float(np.max(np.abs(matrix)))),
        )
    )
    second = solver.solve(
        HermitianSpectralSubspaceDefinition(
            _as_tuple(transformed),
            0,
            2,
            _tolerance(float(np.max(np.abs(transformed)))),
        )
    )

    assert np.asarray(second.projector) == pytest.approx(
        np.asarray(first.projector), abs=3.0e-13
    )
    assert second.diagnostics.spectral_diameter == pytest.approx(
        3.5 * first.diagnostics.spectral_diameter,
        rel=3.0e-15,
    )


def test_256_seeded_cases_agree_with_converged_mpmath_oracle() -> None:
    generator = random.Random(20261014)
    numpy_generator = np.random.default_rng(20261014)
    solver = NumpyHermitianSpectralSubspaceSolver()
    epsilon = math.ulp(1.0)
    maximum_input_entry_magnitude = 0.0
    maximum_eigenvalue_error = 0.0
    maximum_eigenvalue_bound = 0.0
    maximum_eigenvalue_ratio = 0.0
    maximum_projector_error = 0.0
    maximum_projector_bound = 0.0
    maximum_projector_ratio = 0.0

    for case in range(256):
        dimension = 1 + case % 8
        raw = numpy_generator.normal(
            size=(dimension, dimension)
        ) + 1j * numpy_generator.normal(size=(dimension, dimension))
        unitary, _ = np.linalg.qr(raw)
        shift = generator.uniform(-1.0, 1.0)
        spacing = 0.75 + generator.uniform(0.0, 0.25)
        known_eigenvalues = np.array(
            [
                0.25 * shift + spacing * (index - (dimension - 1) / 2.0)
                for index in range(dimension)
            ],
            dtype=np.float64,
        )
        dense = unitary @ np.diag(known_eigenvalues) @ unitary.conjugate().T
        dense = (dense + dense.conjugate().T) * 0.5
        matrix = _as_tuple(dense)
        input_entry_magnitude = float(np.max(np.abs(dense)))
        assert input_entry_magnitude <= 4.0
        maximum_input_entry_magnitude = max(
            maximum_input_entry_magnitude,
            input_entry_magnitude,
        )
        start = case % dimension
        stop = start + 1 + (case // 8) % (dimension - start)

        reference_100 = _reference(matrix, start, stop, 100)
        reference_160 = _reference(matrix, start, stop, 160)
        scale = max(1.0, _frobenius(matrix))
        present_gaps = []
        if start > 0:
            present_gaps.append(
                float(reference_160[0][start] - reference_160[0][start - 1])
            )
        if stop < dimension:
            present_gaps.append(
                float(reference_160[0][stop] - reference_160[0][stop - 1])
            )
        gap = min(present_gaps, default=scale)
        assert gap >= 1.0e-3 * scale

        with mpmath.workdps(180):
            eigenvalue_convergence = max(
                (
                    abs(reference_100[0][index] - reference_160[0][index])
                    for index in range(dimension)
                ),
                default=mpmath.mpf("0"),
            )
            projector_convergence = mpmath.sqrt(
                mpmath.fsum(
                    abs(reference_100[1][row][column] - reference_160[1][row][column])
                    ** 2
                    for row in range(dimension)
                    for column in range(dimension)
                )
            )
            assert eigenvalue_convergence <= mpmath.mpf("1e-80") * scale
            assert projector_convergence <= (
                mpmath.mpf("1e-75") * max(1.0, scale / gap)
            )

        result = solver.solve(
            HermitianSpectralSubspaceDefinition(
                matrix,
                start,
                stop,
                _tolerance(float(np.max(np.abs(dense)))),
            )
        )
        eigenvalue_error = max(
            abs(result.eigenvalues[index] - float(reference_160[0][index]))
            for index in range(dimension)
        )
        eigenvalue_bound = 512.0 * epsilon * dimension * scale
        projector_error = _projector_error(result.projector, reference_160[1])
        projector_bound = (
            512.0 * epsilon * dimension
            if start == 0 and stop == dimension
            else 4096.0 * epsilon * dimension * max(1.0, scale / gap)
        )
        maximum_eigenvalue_error = max(maximum_eigenvalue_error, eigenvalue_error)
        maximum_eigenvalue_bound = max(maximum_eigenvalue_bound, eigenvalue_bound)
        maximum_eigenvalue_ratio = max(
            maximum_eigenvalue_ratio,
            eigenvalue_error / eigenvalue_bound,
        )
        maximum_projector_error = max(maximum_projector_error, projector_error)
        maximum_projector_bound = max(maximum_projector_bound, projector_bound)
        maximum_projector_ratio = max(
            maximum_projector_ratio,
            projector_error / projector_bound,
        )
        assert eigenvalue_error <= eigenvalue_bound
        assert projector_error <= projector_bound

    # Retain meaningful stress assertions even when individual errors are exact zeros.
    assert maximum_eigenvalue_ratio < 1.0
    assert maximum_projector_ratio < 1.0

    output = os.environ.get("PROJECTKOIOS_HERMITIAN_CALIBRATION_OUTPUT")
    if output is not None:
        repository_root = Path(__file__).resolve().parents[6]
        implementation_path = repository_root / (
            "src/python/projectkoios/numerics/linalg/hermitian_spectral_subspace.py"
        )
        numpy_backend_path = repository_root / (
            "src/python/projectkoios/numerics/linalg/numpy/"
            "hermitian_spectral_subspace.py"
        )
        numerical_test_path = Path(__file__).resolve()
        numpy_configuration = cast(dict[str, Any], np.show_config(mode="dicts"))
        python_information = numpy_configuration.get("Python Information")
        if isinstance(python_information, dict):
            python_information.pop("path", None)
        evidence = {
            "calibration_schema": 2,
            "case_count": 256,
            "dimension_minimum": 1,
            "dimension_maximum": 8,
            "mpmath_decimal_precisions": [100, 160],
            "implementation_sha256": hashlib.sha256(
                implementation_path.read_bytes()
            ).hexdigest(),
            "mpmath_version": mpmath.__version__,
            "numerical_test_sha256": hashlib.sha256(
                numerical_test_path.read_bytes()
            ).hexdigest(),
            "numpy_backend_sha256": hashlib.sha256(
                numpy_backend_path.read_bytes()
            ).hexdigest(),
            "numpy_configuration": numpy_configuration,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_random_seed": 20261014,
            "numpy_random_seed": 20261014,
            "observed": {
                "maximum_input_entry_magnitude": maximum_input_entry_magnitude,
                "maximum_eigenvalue_absolute_error": maximum_eigenvalue_error,
                "maximum_eigenvalue_acceptance_bound": maximum_eigenvalue_bound,
                "maximum_eigenvalue_bound_ratio": maximum_eigenvalue_ratio,
                "maximum_projector_frobenius_error": maximum_projector_error,
                "maximum_projector_acceptance_bound": maximum_projector_bound,
                "maximum_projector_bound_ratio": maximum_projector_ratio,
            },
        }
        Path(output).write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
