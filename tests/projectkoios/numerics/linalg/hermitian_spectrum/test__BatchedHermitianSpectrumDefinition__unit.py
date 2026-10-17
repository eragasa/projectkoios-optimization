"""Unit tests for canonical batched Hermitian-spectrum definitions."""

from __future__ import annotations

import math
from typing import cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)

pytestmark = pytest.mark.unit


def _tolerance() -> BatchedHermitianSpectrumTolerance:
    return BatchedHermitianSpectrumTolerance(1.0e-12, 1.0e-10)


def test_constructor_retains_canonical_matrix_storage() -> None:
    matrices = (
        (
            (1.0 + 0.0j, 2.0 - 1.0j),
            (2.0 + 1.0j, 3.0 + 0.0j),
        ),
        (
            (4.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 5.0 + 0.0j),
        ),
    )
    tolerance = _tolerance()

    definition = BatchedHermitianSpectrumDefinition(matrices, tolerance)

    assert definition.dimension == 2
    assert definition.batch_size == 2
    assert definition.matrices is matrices
    assert definition.tolerance is tolerance


def test_empty_constructor_retains_explicit_dimension() -> None:
    definition = BatchedHermitianSpectrumDefinition.empty(4, _tolerance())

    assert definition.dimension == 4
    assert definition.batch_size == 0
    assert definition.matrices == ()


def test_constructor_requires_exact_positive_dimension() -> None:
    with pytest.raises(TypeError, match="dimension"):
        BatchedHermitianSpectrumDefinition.empty(True, _tolerance())
    with pytest.raises(ValueError, match="dimension"):
        BatchedHermitianSpectrumDefinition.empty(0, _tolerance())


def test_constructor_requires_exact_canonical_matrix_storage() -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        BatchedHermitianSpectrumDefinition(
            cast(tuple[tuple[tuple[complex, ...], ...], ...], []),
            _tolerance(),
        )
    with pytest.raises(TypeError, match="exact built-in complex"):
        BatchedHermitianSpectrumDefinition(
            cast(
                tuple[tuple[tuple[complex, ...], ...], ...],
                (((1.0,),),),
            ),
            _tolerance(),
        )
    with pytest.raises(ValueError, match="row count"):
        BatchedHermitianSpectrumDefinition(
            (
                ((1.0 + 0.0j,),),
                (
                    (1.0 + 0.0j, 0.0 + 0.0j),
                    (0.0 + 0.0j, 1.0 + 0.0j),
                ),
            ),
            _tolerance(),
        )
    with pytest.raises(ValueError, match="finite"):
        BatchedHermitianSpectrumDefinition(
            (((complex(math.nan, 0.0),),),),
            _tolerance(),
        )


@pytest.mark.parametrize(
    ("absolute", "relative", "error"),
    (
        (True, 0.0, TypeError),
        (0.0, True, TypeError),
        (complex(1.0, 0.0), 0.0, TypeError),
        (-1.0, 0.0, ValueError),
        (math.inf, 0.0, ValueError),
        (0.0, math.nan, ValueError),
        (0.0, 1.0, ValueError),
    ),
)
def test_tolerance_rejects_noncanonical_or_out_of_range_values(
    absolute: object,
    relative: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        BatchedHermitianSpectrumTolerance(
            cast(float, absolute),
            cast(float, relative),
        )


def test_constructor_requires_nominal_tolerance() -> None:
    with pytest.raises(TypeError, match="BatchedHermitianSpectrumTolerance"):
        BatchedHermitianSpectrumDefinition(
            (((1.0 + 0.0j,),),),
            cast(BatchedHermitianSpectrumTolerance, object()),
        )
