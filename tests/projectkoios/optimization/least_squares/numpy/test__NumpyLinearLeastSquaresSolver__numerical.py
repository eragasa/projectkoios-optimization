from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.least_squares.numpy.linear import (  # noqa: E402
    NumpyLinearLeastSquaresSolver,
)

pytestmark = pytest.mark.numerical


def _matrix(values: Any) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(value) for value in row) for row in values)


def test_weighted_complex_solutions_match_normal_equation_and_qr_references() -> None:
    generator = np.random.default_rng(20261013)
    solver = NumpyLinearLeastSquaresSolver()

    for _ in range(8):
        design = generator.normal(size=(8, 3)) + 1.0j * generator.normal(size=(8, 3))
        targets = generator.normal(size=(8, 4)) + 1.0j * generator.normal(size=(8, 4))
        weights_array = generator.uniform(0.25, 3.0, size=8)
        weights = tuple(float(value) for value in weights_array)
        square_root_weights = np.sqrt(weights_array)
        weighted_design = square_root_weights[:, None] * design
        weighted_targets = square_root_weights[:, None] * targets

        normal_reference = np.linalg.solve(
            weighted_design.conj().T @ weighted_design,
            weighted_design.conj().T @ weighted_targets,
        )
        orthogonal, triangular = np.linalg.qr(weighted_design, mode="reduced")
        qr_reference = np.linalg.solve(
            triangular,
            orthogonal.conj().T @ weighted_targets,
        )
        result = solver.solve(
            _matrix(design),
            _matrix(targets),
            weights=weights,
        )
        coefficients = np.asarray(result.coefficients)

        assert coefficients == pytest.approx(normal_reference, rel=1.0e-10, abs=1.0e-11)
        assert coefficients == pytest.approx(qr_reference, rel=1.0e-10, abs=1.0e-11)
        weighted_residuals = square_root_weights[:, None] * (
            design @ coefficients - targets
        )
        stationarity = weighted_design.conj().T @ weighted_residuals
        scale = np.linalg.norm(weighted_design) * np.linalg.norm(weighted_residuals)
        assert np.linalg.norm(stationarity) <= 1.0e-11 + 1.0e-10 * scale
