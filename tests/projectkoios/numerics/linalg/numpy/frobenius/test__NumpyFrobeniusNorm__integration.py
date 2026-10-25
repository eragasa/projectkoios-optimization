"""Integration tests for dense NumPy Frobenius norms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.numpy.frobenius import NumpyFrobeniusNorm
from projectkoios.numerics.numpy.typing import DenseMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64), 5.0),
        (
            np.asarray(
                (
                    (3.0 + 4.0j, 0.0 + 0.0j),
                    (1.0 - 1.0j, 0.0 + 1.0j),
                ),
                dtype=np.complex128,
            ),
            math.sqrt(28.0),
        ),
    ],
)
def test_evaluation_returns_scalar_norm(
    matrix: DenseMatrix,
    expected: float,
) -> None:
    evaluation = NumpyFrobeniusNorm(matrix)

    result = evaluation.result()

    assert evaluation.matrix_norm is MatrixNormEnum.frobenius
    assert result == expected
    assert type(result) is float


def test_evaluation_propagates_dense_representation_validation() -> None:
    matrix: DenseMatrix = np.asarray(((float("nan"),),), dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        NumpyFrobeniusNorm(matrix).result()
