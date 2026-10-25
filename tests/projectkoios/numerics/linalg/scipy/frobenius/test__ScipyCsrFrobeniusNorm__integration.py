"""Integration tests for SciPy CSR Frobenius norms."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from projectkoios.numerics.linalg.matrix_norm import MatrixNormEnum
from projectkoios.numerics.linalg.scipy.frobenius import ScipyCsrFrobeniusNorm
from projectkoios.numerics.scipy.typing import CsrMatrix

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (
            sp.csr_matrix(np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64)),
            5.0,
        ),
        (
            sp.csr_array(
                np.asarray(
                    (
                        (3.0 + 4.0j, 0.0 + 0.0j),
                        (1.0 - 1.0j, 0.0 + 1.0j),
                    ),
                    dtype=np.complex128,
                )
            ),
            math.sqrt(28.0),
        ),
    ],
)
def test_evaluation_returns_scalar_norm(
    matrix: CsrMatrix,
    expected: float,
) -> None:
    evaluation = ScipyCsrFrobeniusNorm(matrix)

    result = evaluation.result()

    assert evaluation.matrix_norm is MatrixNormEnum.frobenius
    assert result == expected
    assert type(result) is float


def test_evaluation_propagates_csr_representation_validation() -> None:
    matrix: CsrMatrix = sp.csr_array(np.asarray(((float("nan"),),), dtype=np.float64))

    with pytest.raises(ValueError, match="finite"):
        ScipyCsrFrobeniusNorm(matrix).result()
