"""Integration checks for shared NumPy array aliases."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import (
    BoolArray,
    Complex128Array,
    Complex128Matrix,
    DenseMatrix,
    Float64Array,
    Float64Matrix,
    Int64Array,
    IntpArray,
    NumpyArrayLike,
)

pytestmark = pytest.mark.integration


def test_aliases_accept_the_declared_concrete_numpy_dtypes() -> None:
    float_values: Float64Array = np.asarray((1.0,), dtype=np.float64)
    complex_values: Complex128Array = np.asarray((1.0 + 2.0j,), dtype=np.complex128)
    float_matrix: Float64Matrix = np.asarray(((1.0, 0.0),), dtype=np.float64)
    complex_matrix: Complex128Matrix = np.asarray(
        ((1.0 + 2.0j, 0.0 + 0.0j),), dtype=np.complex128
    )
    dense_float_matrix: DenseMatrix = float_matrix
    dense_complex_matrix: DenseMatrix = complex_matrix
    int64_values: Int64Array = np.asarray((1,), dtype=np.int64)
    intp_values: IntpArray = np.asarray((1,), dtype=np.intp)
    bool_values: BoolArray = np.asarray((True,), dtype=np.bool_)
    array_like: NumpyArrayLike = (1, 2)
    converted: Float64Array = np.asarray(array_like, dtype=np.float64)

    assert float_values.dtype == np.dtype(np.float64)
    assert complex_values.dtype == np.dtype(np.complex128)
    assert float_matrix.shape == (1, 2)
    assert complex_matrix.shape == (1, 2)
    assert dense_float_matrix.dtype == np.dtype(np.float64)
    assert dense_complex_matrix.dtype == np.dtype(np.complex128)
    assert int64_values.dtype == np.dtype(np.int64)
    assert intp_values.dtype == np.dtype(np.intp)
    assert bool_values.dtype == np.dtype(np.bool_)
    assert converted.dtype == np.dtype(np.float64)
