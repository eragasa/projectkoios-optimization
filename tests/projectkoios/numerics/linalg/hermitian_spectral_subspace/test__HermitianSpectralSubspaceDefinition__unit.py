"""Unit tests for Hermitian spectral-subspace definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceTolerance,
)

pytestmark = pytest.mark.unit


def _tolerance() -> HermitianSpectralSubspaceTolerance:
    return HermitianSpectralSubspaceTolerance(*(1.0e-12,) * 7)


def test_definition_retains_matrix_range_and_derived_dimensions() -> None:
    matrix = (
        (1.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j),
        (0.0 - 1.0j, 2.0 + 0.0j, 0.5 + 0.0j),
        (0.0 + 0.0j, 0.5 + 0.0j, 3.0 + 0.0j),
    )
    definition = HermitianSpectralSubspaceDefinition(matrix, 1, 3, _tolerance())

    assert definition.matrix == matrix
    assert definition.dimension == 3
    assert definition.selected_rank == 2
    assert definition.start_index == 1
    assert definition.stop_index == 3
    with pytest.raises(FrozenInstanceError):
        definition.start_index = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    (
        ([], TypeError, "matrix"),
        ((), ValueError, "nonempty"),
        (([],), TypeError, "rows"),
        (((1.0 + 0.0j, 0.0 + 0.0j),), ValueError, "square"),
        (((1.0,),), TypeError, "complex"),
        (((complex(float("nan"), 0.0),),), ValueError, "finite"),
        (((complex(0.0, float("inf")),),), ValueError, "finite"),
    ),
)
def test_definition_rejects_invalid_matrix_structure(
    matrix: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        HermitianSpectralSubspaceDefinition(
            cast(Any, matrix),
            0,
            1,
            _tolerance(),
        )


@pytest.mark.parametrize(
    ("start", "stop", "error", "message"),
    (
        (True, 1, TypeError, "start_index"),
        (0.0, 1, TypeError, "start_index"),
        (-1, 1, ValueError, "start_index"),
        (0, True, TypeError, "stop_index"),
        (0, 1.0, TypeError, "stop_index"),
        (1, 1, ValueError, "less"),
        (1, 0, ValueError, "less"),
        (0, 2, ValueError, "dimension"),
    ),
)
def test_definition_rejects_invalid_range(
    start: object,
    stop: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        HermitianSpectralSubspaceDefinition(
            ((1.0 + 0.0j,),),
            cast(Any, start),
            cast(Any, stop),
            _tolerance(),
        )


def test_definition_requires_exact_tolerance_record() -> None:
    with pytest.raises(TypeError, match="HermitianSpectralSubspaceTolerance"):
        HermitianSpectralSubspaceDefinition(
            ((1.0 + 0.0j,),),
            0,
            1,
            cast(Any, object()),
        )


def test_definition_defers_hermiticity_policy_to_solver() -> None:
    definition = HermitianSpectralSubspaceDefinition(
        ((1.0 + 1.0j,),),
        0,
        1,
        _tolerance(),
    )

    assert definition.matrix == ((1.0 + 1.0j,),)
