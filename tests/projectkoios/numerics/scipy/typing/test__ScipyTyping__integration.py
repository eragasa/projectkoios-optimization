"""Integration checks for shared SciPy object interfaces."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import optimize, stats

from projectkoios.numerics.scipy.typing import (
    Float64SparseMatrix,
    GaussianKde,
    OptimizeResult,
)

pytestmark = pytest.mark.integration


def test_float64_sparse_matrix_interface_matches_csr_array_operations() -> None:
    matrix = cast(
        Float64SparseMatrix,
        sp.csr_array(np.asarray(((2.0, 0.0), (0.0, 3.0)), dtype=np.float64)),
    )
    vector = np.asarray((1.0, 2.0), dtype=np.float64)

    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.dtype(np.float64)
    assert tuple(float(value) for value in matrix @ vector) == (2.0, 6.0)
    assert float(matrix[0, 0]) == 2.0
    assert matrix.copy().nnz == 2
    assert (matrix - matrix.transpose()).nnz == 0


def test_gaussian_kde_interface_matches_consumed_operations() -> None:
    source = np.asarray(((-1.0, 0.0, 1.0),), dtype=np.float64)
    kde = cast(GaussianKde, stats.gaussian_kde(source, "silverman"))
    points = np.asarray(((0.0, 0.5),), dtype=np.float64)

    assert kde.factor > 0.0
    assert kde(points).shape == (2,)
    assert kde.logpdf(points).shape == (2,)
    assert kde.integrate_kde(kde) > 0.0
    assert kde.resample(2, seed=np.random.RandomState(7)).shape == (1, 2)


def test_optimize_result_interface_matches_consumed_fields() -> None:
    result = cast(
        OptimizeResult,
        optimize.minimize(
            lambda values: float(values @ values),
            np.asarray((1.0, -1.0), dtype=np.float64),
        ),
    )

    assert result.success
    assert result.x.shape == (2,)
    assert float(result.fun) >= 0.0
    assert isinstance(result.message, str)
