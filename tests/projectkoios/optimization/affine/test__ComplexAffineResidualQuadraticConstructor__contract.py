"""Contract tests for complex-observation affine quadratic construction."""

from __future__ import annotations

from typing import cast

import pytest

from projectkoios.optimization.affine.complex_quadratic import (
    ComplexAffineResidualMatrix,
    ComplexAffineResidualQuadraticConstructor,
    ComplexAffineResidualQuadraticTolerance,
    ComplexAffineResidualVector,
)
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)

pytestmark = pytest.mark.unit


class _ForgedConstructor(ComplexAffineResidualQuadraticConstructor):
    __slots__ = ("_result",)

    def __init__(self, result: object) -> None:
        super().__init__()
        self._result = result

    def _construct_validated(
        self,
        design_matrix: ComplexAffineResidualMatrix,
        target_vector: ComplexAffineResidualVector,
        weights: tuple[float, ...] | None,
        /,
    ) -> PositiveDefiniteQuadraticLoss:
        del design_matrix, target_vector, weights
        return cast(PositiveDefiniteQuadraticLoss, self._result)


def _valid_result() -> PositiveDefiniteQuadraticLoss:
    return PositiveDefiniteQuadraticLoss(
        (1.0, 2.0),
        ((1.0, 0.0), (0.0, 1.0)),
        0.0,
    )


def test_nominal_constructor_is_abstract_and_public_template_is_final() -> None:
    with pytest.raises(TypeError):
        ComplexAffineResidualQuadraticConstructor()

    assert ComplexAffineResidualQuadraticConstructor.construct.__final__ is True


@pytest.mark.parametrize(
    ("absolute", "relative", "error"),
    (
        (-1.0, 1.0e-10, ValueError),
        (1.0e-12, -1.0, ValueError),
        (0.0, 0.0, ValueError),
        (True, 1.0e-10, TypeError),
    ),
)
def test_tolerance_rejects_invalid_values(
    absolute: object,
    relative: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ComplexAffineResidualQuadraticTolerance(absolute, relative)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("design", "targets", "weights", "error", "message"),
    (
        ([], (1.0,), None, TypeError, "design_matrix"),
        (((1.0,), (1.0, 2.0)), (1.0, 2.0), None, ValueError, "common length"),
        (((True,),), (1.0,), None, TypeError, "excluding bool"),
        (((1.0,),), (), None, TypeError, "target_vector"),
        (((1.0,),), (1.0, 2.0), None, ValueError, "length"),
        (((1.0,),), (1.0,), [1.0], TypeError, "weights"),
        (((1.0,),), (1.0,), (0.0,), ValueError, "strictly positive"),
        (((1.0,),), (1.0,), (float("inf"),), ValueError, "finite"),
    ),
)
def test_public_template_rejects_invalid_inputs(
    design: object,
    targets: object,
    weights: object,
    error: type[Exception],
    message: str,
) -> None:
    constructor = _ForgedConstructor(_valid_result())

    with pytest.raises(error, match=message):
        constructor.construct(  # type: ignore[arg-type]
            design,
            targets,
            weights=weights,
        )


def test_public_template_accepts_consistent_backend_result() -> None:
    result = _valid_result()

    actual = _ForgedConstructor(result).construct(
        ((1.0, 0.0), (0.0, 1.0)),
        (1.0, 2.0),
    )

    assert actual is result


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (object(), "PositiveDefiniteQuadraticLoss"),
        (
            PositiveDefiniteQuadraticLoss(
                (1.0,),
                ((1.0,),),
                0.0,
            ),
            "dimension",
        ),
        (
            PositiveDefiniteQuadraticLoss(
                (1.0, 2.0),
                ((2.0, 0.0), (0.0, 1.0)),
                0.0,
            ),
            "matrix",
        ),
        (
            PositiveDefiniteQuadraticLoss(
                (0.0, 2.0),
                ((1.0, 0.0), (0.0, 1.0)),
                1.0,
            ),
            "stationary",
        ),
        (
            PositiveDefiniteQuadraticLoss(
                (1.0, 2.0),
                ((1.0, 0.0), (0.0, 1.0)),
                1.0,
            ),
            "minimum",
        ),
    ),
)
def test_public_template_rejects_inconsistent_backend_result(
    result: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _ForgedConstructor(result).construct(
            ((1.0, 0.0), (0.0, 1.0)),
            (1.0, 2.0),
        )
