"""NumPy integration tests for complex affine residual quadratics."""

from __future__ import annotations

from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")

from projectkoios.optimization.affine.numpy.complex_quadratic import (  # noqa: E402
    NumpyComplexAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.integration


def _matrix(values: Any) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(value) for value in row) for row in values)


def _vector(values: Any) -> tuple[complex, ...]:
    return tuple(complex(value) for value in values)


def test_complex_observation_can_identify_two_real_coordinates() -> None:
    result = NumpyComplexAffineResidualQuadraticConstructor().construct(
        ((1.0, 1.0j),),
        (1.0 + 2.0j,),
    )

    assert result.center == pytest.approx((1.0, 2.0))
    assert result.matrix == ((1.0, 0.0), (0.0, 1.0))
    assert result.minimum_squared_loss == pytest.approx(0.0, abs=1.0e-15)
    assert result.squared_loss((2.0, 4.0)) == pytest.approx(5.0)


def test_positive_weights_are_objective_multipliers() -> None:
    design = np.asarray(
        (
            (1.0 + 1.0j, 0.5 - 0.25j),
            (0.5 - 0.5j, -1.0 + 0.25j),
            (2.0 + 0.0j, 0.25 + 1.0j),
            (-0.5 + 0.25j, 1.5 - 0.5j),
        ),
        dtype=np.complex128,
    )
    targets = np.asarray((1.0 + 0.5j, -0.25j, 2.0 - 1.0j, 0.75 + 0.25j))
    weights = np.asarray((0.5, 2.0, 1.5, 3.0))
    stacked_design = np.concatenate((design.real, design.imag), axis=0)
    stacked_targets = np.concatenate((targets.real, targets.imag), axis=0)
    scales = np.concatenate((np.sqrt(weights), np.sqrt(weights)))
    expected_center, _, expected_rank, _ = np.linalg.lstsq(
        scales[:, None] * stacked_design,
        scales * stacked_targets,
        rcond=None,
    )
    expected_residual = design @ expected_center - targets
    expected_minimum = float(np.sum(weights * np.abs(expected_residual) ** 2))
    expected_matrix = np.real(design.conj().T @ (weights[:, None] * design))

    result = NumpyComplexAffineResidualQuadraticConstructor().construct(
        _matrix(design),
        _vector(targets),
        weights=tuple(float(value) for value in weights),
    )

    assert int(expected_rank) == 2
    assert result.center == pytest.approx(tuple(expected_center))
    assert np.asarray(result.matrix) == pytest.approx(expected_matrix)
    assert result.minimum_squared_loss == pytest.approx(expected_minimum)
    assert np.array_equal(np.asarray(result.matrix), np.asarray(result.matrix).T)


def test_none_weights_equal_explicit_unit_weights() -> None:
    design = ((1.0 + 1.0j,), (2.0 - 0.5j,), (-1.0 + 0.25j,))
    targets = (2.0 + 0.0j, 1.0 - 1.0j, -0.5 + 0.25j)
    constructor = NumpyComplexAffineResidualQuadraticConstructor()

    ordinary = constructor.construct(design, targets)
    explicit = constructor.construct(design, targets, weights=(1.0, 1.0, 1.0))

    assert ordinary.center == explicit.center
    assert ordinary.matrix == explicit.matrix
    assert ordinary.minimum_squared_loss == explicit.minimum_squared_loss


def test_minimum_is_evaluated_from_residual_without_cancellation() -> None:
    result = NumpyComplexAffineResidualQuadraticConstructor().construct(
        ((1.0,), (1.0,)),
        (1.0e8, 1.0e8 + 1.0),
    )

    assert result.center == pytest.approx((1.0e8 + 0.5,))
    assert result.matrix == ((2.0,),)
    assert result.minimum_squared_loss == pytest.approx(0.5)


def test_real_coordinate_rank_deficiency_is_rejected() -> None:
    with pytest.raises(ValueError, match="full real-coordinate rank"):
        NumpyComplexAffineResidualQuadraticConstructor().construct(
            ((1.0 + 1.0j, 2.0 + 2.0j), (2.0, 4.0)),
            (1.0, 2.0),
        )


def test_numpy_lstsq_failure_is_translated_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = np.linalg.LinAlgError("forced complex affine failure")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(np.linalg, "lstsq", fail)

    with pytest.raises(
        ValueError,
        match="NumPy complex affine least-squares solve failed",
    ) as caught:
        NumpyComplexAffineResidualQuadraticConstructor().construct(
            ((1.0,),),
            (2.0,),
        )

    assert caught.value.__cause__ is failure
