from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.numerics.numpy.validation import (
    finite_nonnegative_float64_array,
    immutable_array,
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
    with pytest.raises(ValueError):
        result.setflags(write=True)


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
