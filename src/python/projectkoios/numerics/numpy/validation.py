"""Validate exact NumPy array representations used by numerical capabilities.

Notes
-----
NumPy remains capability-local. These functions validate software representation and
finite arithmetic only; they assign no optimization or scientific meaning.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .typing import Complex128Array, Float64Array, Int64Array


def _validate_exact_finite_array(
    value: np.ndarray,
    name: str,
    dtype: np.dtype[np.generic],
    /,
) -> np.ndarray:
    """Validate one exact finite NumPy array and dtype.

    Parameters
    ----------
    value
        Candidate exact NumPy array.
    name
        Field name included in failure messages.
    dtype
        Required exact NumPy dtype.

    Returns
    -------
    numpy.ndarray
        Original validated array.

    Raises
    ------
    TypeError
        If the value is not an exact ndarray or has the wrong exact dtype.
    ValueError
        If any entry is nonfinite.
    """
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have exact dtype {dtype.name}")
    finite = np.isfinite(value)
    if not bool(finite.all()):
        first = tuple(int(component) for component in np.argwhere(~finite)[0])
        raise ValueError(f"{name} contains a nonfinite entry at index {first}")
    return value


def _has_immutable_bytes_backing(value: np.ndarray, /) -> bool:
    """Return whether an array's ultimate backing owner is immutable bytes.

    Parameters
    ----------
    value
        Exact NumPy array whose base chain is inspected.

    Returns
    -------
    bool
        Whether the base chain terminates in exact built-in ``bytes``.
    """
    backing = cast(np.ndarray | bytes | None, value.base)
    while type(backing) is np.ndarray:
        backing = cast(np.ndarray | bytes | None, backing.base)
    return type(backing) is bytes


def immutable_array(
    value: np.ndarray,
    name: str,
    dtype: np.dtype[np.generic],
    /,
) -> np.ndarray:
    """Return an immutable byte-backed copy of one exact NumPy array.

    Parameters
    ----------
    value
        Candidate exact NumPy array.
    name
        Field name included in failure messages.
    dtype
        Required exact NumPy dtype.

    Returns
    -------
    numpy.ndarray
        C-order array whose immutable bytes backing prevents write re-enablement.

    Raises
    ------
    TypeError
        If the value is not an exact ndarray or has the wrong exact dtype.
    ValueError
        If any entry is nonfinite.
    """
    validated = _validate_exact_finite_array(value, name, dtype)
    backing = validated.tobytes(order="C")
    return np.frombuffer(backing, dtype=dtype).reshape(validated.shape)


def validate_immutable_array(
    value: np.ndarray,
    name: str,
    dtype: np.dtype[np.generic],
    /,
) -> np.ndarray:
    """Validate one canonical immutable byte-backed NumPy array.

    Parameters
    ----------
    value
        Candidate exact NumPy array.
    name
        Field name included in failure messages.
    dtype
        Required exact NumPy dtype.

    Returns
    -------
    numpy.ndarray
        Original validated array without copying.

    Raises
    ------
    TypeError
        If the value is not an exact ndarray or has the wrong exact dtype.
    ValueError
        If entries are nonfinite or storage is not canonical immutable C order.
    """
    validated = _validate_exact_finite_array(value, name, dtype)
    if not validated.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if validated.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    if not _has_immutable_bytes_backing(validated):
        raise ValueError(f"{name} must have immutable bytes backing")
    return validated


def immutable_complex128_array(
    value: Complex128Array,
    name: str,
    /,
) -> Complex128Array:
    """Return a canonical immutable copy of an exact complex128 array.

    Parameters
    ----------
    value
        Candidate exact finite complex128 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Immutable byte-backed complex128 copy.
    """
    return cast(
        Complex128Array,
        immutable_array(value, name, np.dtype(np.complex128)),
    )


def immutable_float64_array(
    value: Float64Array,
    name: str,
    /,
) -> Float64Array:
    """Return a canonical immutable copy of an exact float64 array.

    Parameters
    ----------
    value
        Candidate exact finite float64 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Immutable byte-backed float64 copy.
    """
    return cast(Float64Array, immutable_array(value, name, np.dtype(np.float64)))


def immutable_int64_array(
    value: Int64Array,
    name: str,
    /,
) -> Int64Array:
    """Return a canonical immutable copy of an exact int64 array.

    Parameters
    ----------
    value
        Candidate exact int64 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Immutable byte-backed int64 copy.
    """
    return cast(Int64Array, immutable_array(value, name, np.dtype(np.int64)))


def validate_immutable_complex128_array(
    value: Complex128Array,
    name: str,
    /,
) -> Complex128Array:
    """Validate canonical immutable complex128 storage without copying.

    Parameters
    ----------
    value
        Candidate exact finite complex128 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Original validated complex128 array.
    """
    return cast(
        Complex128Array,
        validate_immutable_array(value, name, np.dtype(np.complex128)),
    )


def validate_immutable_float64_array(
    value: Float64Array,
    name: str,
    /,
) -> Float64Array:
    """Validate canonical immutable float64 storage without copying.

    Parameters
    ----------
    value
        Candidate exact finite float64 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Original validated float64 array.
    """
    return cast(
        Float64Array,
        validate_immutable_array(value, name, np.dtype(np.float64)),
    )


def validate_immutable_int64_array(
    value: Int64Array,
    name: str,
    /,
) -> Int64Array:
    """Validate canonical immutable int64 storage without copying.

    Parameters
    ----------
    value
        Candidate exact int64 array.
    name
        Field name included in failures.

    Returns
    -------
    numpy.ndarray
        Original validated int64 array.
    """
    return cast(
        Int64Array,
        validate_immutable_array(value, name, np.dtype(np.int64)),
    )


def finite_nonnegative_float64_array(
    value: Float64Array,
    name: str,
    /,
) -> Float64Array:
    """Validate one exact finite nonnegative float64 array.

    Parameters
    ----------
    value
        Candidate exact float64 ndarray.
    name
        Field name included in failure messages.

    Returns
    -------
    numpy.ndarray
        Original validated array without copying.

    Raises
    ------
    TypeError
        If the value is not an exact ndarray or does not have exact float64 dtype.
    ValueError
        If any entry is nonfinite or negative.
    """
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have exact dtype float64")
    if not bool(np.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


__all__ = [
    "finite_nonnegative_float64_array",
    "immutable_array",
    "immutable_complex128_array",
    "immutable_float64_array",
    "immutable_int64_array",
    "validate_immutable_array",
    "validate_immutable_complex128_array",
    "validate_immutable_float64_array",
    "validate_immutable_int64_array",
]
