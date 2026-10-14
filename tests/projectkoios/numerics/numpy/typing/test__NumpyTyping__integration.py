"""Integration checks for shared NumPy array aliases."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import (
    BoolArray,
    Complex128Array,
    Float64Array,
    Int64Array,
    IntpArray,
    NumpyArrayLike,
)

pytestmark = pytest.mark.integration


def _float64_identity(values: Float64Array) -> Float64Array:
    return values


def _complex128_identity(values: Complex128Array) -> Complex128Array:
    return values


def _int64_identity(values: Int64Array) -> Int64Array:
    return values


def _intp_identity(values: IntpArray) -> IntpArray:
    return values


def _bool_identity(values: BoolArray) -> BoolArray:
    return values


def _as_float64(values: NumpyArrayLike) -> Float64Array:
    return np.asarray(values, dtype=np.float64)


def test_aliases_accept_the_declared_concrete_numpy_dtypes() -> None:
    float_values = _float64_identity(np.asarray((1.0,), dtype=np.float64))
    complex_values = _complex128_identity(
        np.asarray((1.0 + 2.0j,), dtype=np.complex128)
    )
    int64_values = _int64_identity(np.asarray((1,), dtype=np.int64))
    intp_values = _intp_identity(np.asarray((1,), dtype=np.intp))
    bool_values = _bool_identity(np.asarray((True,), dtype=np.bool_))

    assert float_values.dtype == np.dtype(np.float64)
    assert complex_values.dtype == np.dtype(np.complex128)
    assert int64_values.dtype == np.dtype(np.int64)
    assert intp_values.dtype == np.dtype(np.intp)
    assert bool_values.dtype == np.dtype(np.bool_)
    assert _as_float64((1, 2)).dtype == np.dtype(np.float64)
