"""Bounded numerical references for complex affine residual quadratics."""

from __future__ import annotations

from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.affine.numpy.complex_quadratic import (  # noqa: E402
    NumpyComplexAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.numerical


def _matrix(values: Any) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(value) for value in row) for row in values)


def _vector(values: Any) -> tuple[complex, ...]:
    return tuple(complex(value) for value in values)


@pytest.mark.parametrize("seed", (101, 202, 303, 404, 505))
def test_seeded_system_matches_independent_stacked_real_reference(seed: int) -> None:
    generator = np.random.default_rng(seed)
    design = generator.normal(size=(9, 3)) + 1.0j * generator.normal(size=(9, 3))
    targets = generator.normal(size=9) + 1.0j * generator.normal(size=9)
    weights = np.exp(generator.normal(size=9))
    stacked_design = np.concatenate((design.real, design.imag), axis=0)
    stacked_targets = np.concatenate((targets.real, targets.imag), axis=0)
    scales = np.concatenate((np.sqrt(weights), np.sqrt(weights)))
    expected_center, _, rank, _ = np.linalg.lstsq(
        scales[:, None] * stacked_design,
        scales * stacked_targets,
        rcond=None,
    )
    expected_matrix = np.real(design.conj().T @ (weights[:, None] * design))
    expected_residuals = design @ expected_center - targets
    expected_minimum = float(np.sum(weights * np.abs(expected_residuals) ** 2))

    result = NumpyComplexAffineResidualQuadraticConstructor().construct(
        _matrix(design),
        _vector(targets),
        weights=tuple(float(value) for value in weights),
    )

    assert int(rank) == 3
    assert result.center == pytest.approx(tuple(expected_center), abs=1.0e-12)
    assert np.asarray(result.matrix) == pytest.approx(expected_matrix, abs=1.0e-12)
    assert result.minimum_squared_loss == pytest.approx(
        expected_minimum,
        abs=1.0e-12,
    )
    for coordinates in (
        expected_center,
        expected_center + np.asarray((0.1, -0.2, 0.3)),
        expected_center + np.asarray((-0.4, 0.25, 0.05)),
    ):
        residuals = design @ coordinates - targets
        expected_loss = float(np.sum(weights * np.abs(residuals) ** 2))
        assert result.squared_loss(tuple(coordinates)) == pytest.approx(
            expected_loss,
            abs=1.0e-10,
        )


def test_flattened_complex_observations_preserve_declared_order_independently() -> None:
    features = np.asarray(
        (
            ((1.0 + 0.0j, 0.5 + 0.25j), (1.0 + 0.0j, -0.5 + 0.75j)),
            ((1.0 + 0.0j, 0.2 - 0.1j), (1.0 + 0.0j, -0.2 - 0.4j)),
        )
    )
    offsets = np.asarray(((0.25 + 0.1j, -0.5j), (-0.1 + 0.2j, 0.3 - 0.4j)))
    design = features.reshape((-1, 2))
    targets = -offsets.reshape(-1)
    normalization = float(offsets.size) * 4.0
    weight = 1.0 / normalization

    result = NumpyComplexAffineResidualQuadraticConstructor().construct(
        _matrix(design),
        _vector(targets),
        weights=(weight,) * len(targets),
    )

    expected_matrix = np.real(design.conj().T @ design) / normalization
    expected_linear = np.real(design.conj().T @ targets) / normalization
    expected_center = np.linalg.solve(expected_matrix, expected_linear)
    expected_residual = design @ expected_center - targets
    expected_minimum = float(np.vdot(expected_residual, expected_residual).real)
    expected_minimum /= normalization
    assert result.center == pytest.approx(tuple(expected_center), abs=1.0e-12)
    assert np.asarray(result.matrix) == pytest.approx(expected_matrix, abs=1.0e-12)
    assert result.minimum_squared_loss == pytest.approx(
        expected_minimum,
        abs=1.0e-12,
    )
