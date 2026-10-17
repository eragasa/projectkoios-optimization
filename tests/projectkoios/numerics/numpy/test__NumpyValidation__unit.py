from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.numerics.numpy.validation import (
    finite_nonnegative_float64_array,
    immutable_array,
    validate_immutable_array,
)


@dataclass(frozen=True, slots=True)
class Float64ArrayFailureCase:
    """Name one invalid runtime representation and its expected failure."""

    value: Float64Array
    error: type[Exception]
    message_pattern: str


@pytest.fixture
def mutable_complex_array() -> Complex128Array:
    """Return one mutable exact complex128 array."""
    original_value = complex(1.0, 2.0)
    return np.array([[original_value]], dtype=np.complex128)


def test_immutable_array_returns_independent_byte_backed_copy(
    mutable_complex_array: Complex128Array,
) -> None:
    expected = mutable_complex_array.copy()
    replacement_value = -complex(mutable_complex_array[0, 0])

    result = immutable_array(
        mutable_complex_array,
        "source",
        np.dtype(np.complex128),
    )
    mutable_complex_array[0, 0] = replacement_value

    np.testing.assert_array_equal(result, expected)
    assert not result.flags.writeable
    assert validate_immutable_array(result, "result", np.dtype(np.complex128)) is result
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_validate_immutable_array_rejects_non_canonical_order() -> None:
    dimension = 2
    backing = np.arange(dimension * dimension, dtype=np.float64).tobytes()
    fortran_order = np.frombuffer(backing, dtype=np.float64).reshape(
        (dimension, dimension),
        order="F",
    )

    assert fortran_order.flags.f_contiguous
    assert not fortran_order.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_immutable_array(
            fortran_order,
            "values",
            np.dtype(np.float64),
        )


def test_validate_immutable_array_rejects_read_only_mutable_backing() -> None:
    owner = np.ones((2, 2), dtype=np.float64)
    read_only_view = owner.view()
    read_only_view.setflags(write=False)

    with pytest.raises(ValueError, match="immutable bytes backing"):
        validate_immutable_array(
            read_only_view,
            "values",
            np.dtype(np.float64),
        )


@pytest.fixture(
    params=[
        pytest.param(
            Float64ArrayFailureCase(
                value=cast(Float64Array, [[1.0]]),
                error=TypeError,
                message_pattern="exact numpy.ndarray",
            ),
            id="array-like",
        ),
        pytest.param(
            Float64ArrayFailureCase(
                value=cast(
                    Float64Array,
                    np.array([1.0], dtype=np.float32),
                ),
                error=TypeError,
                message_pattern="exact dtype float64",
            ),
            id="wrong-dtype",
        ),
        pytest.param(
            Float64ArrayFailureCase(
                value=np.array([np.nan], dtype=np.float64),
                error=ValueError,
                message_pattern="finite and nonnegative",
            ),
            id="nonfinite",
        ),
        pytest.param(
            Float64ArrayFailureCase(
                value=np.array([-1.0], dtype=np.float64),
                error=ValueError,
                message_pattern="finite and nonnegative",
            ),
            id="negative",
        ),
    ]
)
def float64_array_failure_case(
    request: pytest.FixtureRequest,
) -> Float64ArrayFailureCase:
    """Return one named invalid float64-array validation case."""
    return cast(Float64ArrayFailureCase, request.param)


def test_finite_nonnegative_float64_array_rejects_invalid_values(
    float64_array_failure_case: Float64ArrayFailureCase,
) -> None:
    case = float64_array_failure_case

    with pytest.raises(case.error, match=case.message_pattern):
        finite_nonnegative_float64_array(case.value, "values")


def test_finite_nonnegative_float64_array_returns_original_array() -> None:
    lower_boundary = 0.0
    positive_value = 1.0
    values = np.array([lower_boundary, positive_value], dtype=np.float64)

    assert finite_nonnegative_float64_array(values, "values") is values
