"""Validate exact NumPy array representations used by numerical capabilities.

Notes
-----
NumPy remains capability-local. These functions validate software representation and
finite arithmetic only; they assign no optimization or scientific meaning.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .typing import Float64Array


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
    "validate_immutable_array",
]
