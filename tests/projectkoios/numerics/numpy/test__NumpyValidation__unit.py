from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import cast

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.numerics.numpy.validation import (
    NumpyFiniteArrayValidator,
    NumpyFiniteNonnegativeFloat64ArrayValidator,
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)


@dataclass(frozen=True, slots=True)
class Float64ArrayFailureCase:
    """Name one invalid runtime representation and its expected failure."""

    value: Float64Array
    error: type[Exception]
    message_pattern: str


def test_validation_module_exposes_nominal_owners_without_function_facades() -> None:
    import projectkoios.numerics.numpy.validation as module

    assert module.__all__ == [
        "NumpyFiniteArrayValidator",
        "NumpyFiniteNonnegativeFloat64ArrayValidator",
        "NumpyImmutableArrayConstructor",
        "NumpyImmutableArrayValidator",
    ]
    for legacy_name in (
        "finite_nonnegative_float64_array",
        "immutable_array",
        "immutable_complex128_array",
        "immutable_float64_array",
        "immutable_int64_array",
        "validate_immutable_array",
        "validate_immutable_complex128_array",
        "validate_immutable_float64_array",
        "validate_immutable_int64_array",
    ):
        assert not hasattr(module, legacy_name)


@pytest.fixture
def mutable_complex_array() -> Complex128Array:
    """Return one mutable exact complex128 array."""
    original_value = complex(1.0, 2.0)
    return np.array([[original_value]], dtype=np.complex128)


def test_immutable_array_owner_binding_is_frozen(
    mutable_complex_array: Complex128Array,
) -> None:
    owner = NumpyImmutableArrayConstructor(
        mutable_complex_array,
        "source",
        np.dtype(np.complex128),
    )

    with pytest.raises(FrozenInstanceError):
        owner.name = "replacement"  # type: ignore[misc]


def test_immutable_array_constructor_returns_independent_byte_backed_copy(
    mutable_complex_array: Complex128Array,
) -> None:
    expected = mutable_complex_array.copy()
    replacement_value = -complex(mutable_complex_array[0, 0])

    result = NumpyImmutableArrayConstructor(
        mutable_complex_array,
        "source",
        np.dtype(np.complex128),
    ).execute()
    mutable_complex_array[0, 0] = replacement_value

    np.testing.assert_array_equal(result, expected)
    assert not result.flags.writeable
    assert (
        NumpyImmutableArrayValidator(
            result,
            "result",
            np.dtype(np.complex128),
        ).execute()
        is result
    )
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        pytest.param(
            np.array([1.0 + 2.0j], dtype=np.complex128),
            np.dtype(np.complex128),
            id="complex128",
        ),
        pytest.param(
            np.array([1.0], dtype=np.float64),
            np.dtype(np.float64),
            id="float64",
        ),
        pytest.param(
            np.array([1], dtype=np.int64),
            np.dtype(np.int64),
            id="int64",
        ),
    ],
)
def test_immutable_array_owners_preserve_exact_dtype(
    values: np.ndarray[tuple[int], np.dtype[np.generic]],
    dtype: np.dtype[np.generic],
) -> None:
    immutable_copy = NumpyImmutableArrayConstructor(
        values,
        "values",
        dtype,
    ).execute()

    assert immutable_copy.dtype == values.dtype
    assert not immutable_copy.flags.writeable
    assert (
        NumpyImmutableArrayValidator(
            immutable_copy,
            "immutable_copy",
            dtype,
        ).execute()
        is immutable_copy
    )


def test_immutable_array_validator_rejects_non_canonical_order() -> None:
    dimension = 2
    backing = np.arange(dimension * dimension, dtype=np.float64).tobytes()
    fortran_order = np.frombuffer(backing, dtype=np.float64).reshape(
        (dimension, dimension),
        order="F",
    )

    assert fortran_order.flags.f_contiguous
    assert not fortran_order.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        NumpyImmutableArrayValidator(
            fortran_order,
            "values",
            np.dtype(np.float64),
        ).execute()


def test_immutable_array_validator_rejects_read_only_mutable_backing() -> None:
    owner = np.ones((2, 2), dtype=np.float64)
    read_only_view = owner.view()
    read_only_view.setflags(write=False)

    with pytest.raises(ValueError, match="immutable bytes backing"):
        NumpyImmutableArrayValidator(
            read_only_view,
            "values",
            np.dtype(np.float64),
        ).execute()


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


def test_finite_nonnegative_float64_array_validator_rejects_invalid_values(
    float64_array_failure_case: Float64ArrayFailureCase,
) -> None:
    case = float64_array_failure_case

    with pytest.raises(case.error, match=case.message_pattern):
        NumpyFiniteNonnegativeFloat64ArrayValidator(
            case.value,
            "values",
        ).execute()


def test_finite_nonnegative_float64_array_validator_returns_original_array() -> None:
    lower_boundary = 0.0
    positive_value = 1.0
    values = np.array([lower_boundary, positive_value], dtype=np.float64)

    assert (
        NumpyFiniteNonnegativeFloat64ArrayValidator(values, "values").execute()
        is values
    )
    assert (
        NumpyFiniteArrayValidator(
            values,
            "values",
            np.dtype(np.float64),
        ).execute()
        is values
    )
