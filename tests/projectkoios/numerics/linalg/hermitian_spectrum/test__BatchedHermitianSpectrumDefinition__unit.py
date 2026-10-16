"""Unit tests for ergonomic batched Hermitian-spectrum definitions."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)

pytestmark = pytest.mark.unit


def _tolerance() -> BatchedHermitianSpectrumTolerance:
    return BatchedHermitianSpectrumTolerance(1.0e-12, 1.0e-10)


def test_nonempty_batch_infers_dimension_and_normalizes_inputs() -> None:
    definition = BatchedHermitianSpectrumDefinition(
        [
            [[1, 2.0 - 1.0j], [2.0 + 1.0j, 3]],
            [[4.0, 0.0], [0.0, 5.0]],
        ],
        _tolerance(),
    )

    assert definition.dimension == 2
    assert definition.batch_size == 2
    assert definition.matrices == (
        ((1.0 + 0.0j, 2.0 - 1.0j), (2.0 + 1.0j, 3.0 + 0.0j)),
        ((4.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 5.0 + 0.0j)),
    )


def test_empty_constructor_requires_dimension_only_when_shape_cannot_be_inferred() -> (
    None
):
    definition = BatchedHermitianSpectrumDefinition.empty(4, _tolerance())

    assert definition.dimension == 4
    assert definition.batch_size == 0
    assert definition.matrices == ()

    with pytest.raises(ValueError, match="require an explicit dimension"):
        BatchedHermitianSpectrumDefinition((), _tolerance())
    with pytest.raises(ValueError, match="must be omitted"):
        BatchedHermitianSpectrumDefinition(
            (((1.0,),),),
            _tolerance(),
            dimension=1,
        )


@pytest.mark.parametrize("dimension", [True, 1.5, 0, -1])
def test_empty_constructor_rejects_invalid_dimension(dimension: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dimension"):
        BatchedHermitianSpectrumDefinition.empty(
            cast(Any, dimension),
            _tolerance(),
        )


@pytest.mark.parametrize(
    "matrices, error, match",
    [
        (1, TypeError, "matrices must be iterable"),
        ("matrix", TypeError, "non-string iterable"),
        ([[]], ValueError, "nonempty"),
        ([[[1.0, 2.0]]], ValueError, "square"),
        ([[[1.0]], [[1.0, 0.0], [0.0, 1.0]]], ValueError, "dimension"),
        ([[[True]]], TypeError, "numeric scalars"),
        ([[[object()]]], TypeError, "numeric scalars"),
        ([[[complex(math.nan, 0.0)]]], ValueError, "finite"),
    ],
)
def test_definition_rejects_malformed_matrix_batches(
    matrices: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        BatchedHermitianSpectrumDefinition(matrices, _tolerance())


@pytest.mark.parametrize(
    "absolute, relative, error",
    [
        (True, 0.0, TypeError),
        (0.0, True, TypeError),
        (complex(1.0, 0.0), 0.0, TypeError),
        (-1.0, 0.0, ValueError),
        (math.inf, 0.0, ValueError),
        (0.0, math.nan, ValueError),
        (0.0, 1.0, ValueError),
    ],
)
def test_tolerance_rejects_invalid_values(
    absolute: object,
    relative: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        BatchedHermitianSpectrumTolerance(
            cast(Any, absolute),
            cast(Any, relative),
        )


def test_definition_requires_nominal_tolerance() -> None:
    with pytest.raises(TypeError, match="BatchedHermitianSpectrumTolerance"):
        BatchedHermitianSpectrumDefinition(
            (((1.0,),),),
            cast(Any, object()),
        )
