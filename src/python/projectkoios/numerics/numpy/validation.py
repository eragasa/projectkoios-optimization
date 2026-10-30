"""Nominal owners for exact NumPy array construction and validation.

Notes
-----
NumPy remains capability-local. These actions own software representation and finite
arithmetic only; they assign no optimization or scientific meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final, overload

import numpy as np
import numpy.typing as npt

from .typing import BoolArray, Complex128Array, Float64Array, Int64Array

type NumpyArray = npt.NDArray[np.generic]


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyFiniteArrayValidator[ArrayT: np.ndarray]:
    """Validate one exact finite NumPy array representation without copying."""

    value: ArrayT
    name: str
    dtype: np.dtype[np.generic]

    def __post_init__(self) -> None:
        """Validate immutable action metadata without inspecting array values."""
        if type(self.name) is not str:
            raise TypeError("name must be a built-in string")
        if not self.name:
            raise ValueError("name must be nonempty")
        if not isinstance(self.dtype, np.dtype):
            raise TypeError("dtype must be numpy.dtype")

    @overload
    def execute(self: NumpyFiniteArrayValidator[BoolArray]) -> BoolArray: ...

    @overload
    def execute(
        self: NumpyFiniteArrayValidator[Complex128Array],
    ) -> Complex128Array: ...

    @overload
    def execute(self: NumpyFiniteArrayValidator[Float64Array]) -> Float64Array: ...

    @overload
    def execute(self: NumpyFiniteArrayValidator[Int64Array]) -> Int64Array: ...

    @overload
    def execute(self) -> NumpyArray: ...

    def execute(self) -> NumpyArray:
        """Return the original array after exact dtype and finiteness validation.

        Raises
        ------
        TypeError
            The value is not an exact ndarray or has the wrong exact dtype.
        ValueError
            An entry is nonfinite.
        """
        if type(self.value) is not np.ndarray:
            raise TypeError(f"{self.name} must be an exact numpy.ndarray")
        if self.value.dtype != self.dtype:
            raise TypeError(f"{self.name} must have exact dtype {self.dtype.name}")
        finite = np.isfinite(self.value)
        if not bool(finite.all()):
            first = tuple(int(component) for component in np.argwhere(~finite)[0])
            raise ValueError(f"{self.name} contains a nonfinite entry at index {first}")
        return self.value


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyImmutableArrayConstructor[ArrayT: np.ndarray]:
    """Construct canonical immutable byte-backed storage for one exact array."""

    value: ArrayT
    name: str
    dtype: np.dtype[np.generic]

    def __post_init__(self) -> None:
        """Validate immutable action metadata without copying the candidate."""
        NumpyFiniteArrayValidator(self.value, self.name, self.dtype)

    @overload
    def execute(self: NumpyImmutableArrayConstructor[BoolArray]) -> BoolArray: ...

    @overload
    def execute(
        self: NumpyImmutableArrayConstructor[Complex128Array],
    ) -> Complex128Array: ...

    @overload
    def execute(self: NumpyImmutableArrayConstructor[Float64Array]) -> Float64Array: ...

    @overload
    def execute(self: NumpyImmutableArrayConstructor[Int64Array]) -> Int64Array: ...

    @overload
    def execute(self) -> NumpyArray: ...

    def execute(self) -> NumpyArray:
        """Return an independent C-order array with immutable bytes backing."""
        validated = NumpyFiniteArrayValidator(
            self.value,
            self.name,
            self.dtype,
        ).execute()
        backing = validated.tobytes(order="C")
        return np.frombuffer(backing, dtype=self.dtype).reshape(validated.shape)


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyImmutableArrayValidator[ArrayT: np.ndarray]:
    """Validate canonical immutable byte-backed storage without copying."""

    value: ArrayT
    name: str
    dtype: np.dtype[np.generic]

    def __post_init__(self) -> None:
        """Validate immutable action metadata without inspecting array storage."""
        NumpyFiniteArrayValidator(self.value, self.name, self.dtype)

    @overload
    def execute(self: NumpyImmutableArrayValidator[BoolArray]) -> BoolArray: ...

    @overload
    def execute(
        self: NumpyImmutableArrayValidator[Complex128Array],
    ) -> Complex128Array: ...

    @overload
    def execute(self: NumpyImmutableArrayValidator[Float64Array]) -> Float64Array: ...

    @overload
    def execute(self: NumpyImmutableArrayValidator[Int64Array]) -> Int64Array: ...

    @overload
    def execute(self) -> NumpyArray: ...

    def execute(self) -> NumpyArray:
        """Return the original canonical immutable array without copying.

        Raises
        ------
        TypeError
            The value is not an exact ndarray or has the wrong exact dtype.
        ValueError
            Entries are nonfinite or storage is not canonical immutable C order.
        """
        validated = NumpyFiniteArrayValidator(
            self.value,
            self.name,
            self.dtype,
        ).execute()
        if not validated.flags.c_contiguous:
            raise ValueError(f"{self.name} must be C-contiguous")
        if validated.flags.writeable:
            raise ValueError(f"{self.name} must be read-only")
        backing: np.ndarray | bytes | None = validated.base
        while type(backing) is np.ndarray:
            backing = backing.base
        if type(backing) is not bytes:
            raise ValueError(f"{self.name} must have immutable bytes backing")
        return validated


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyFiniteNonnegativeFloat64ArrayValidator:
    """Validate one exact finite nonnegative float64 array without copying."""

    value: Float64Array
    name: str

    def __post_init__(self) -> None:
        """Validate immutable action metadata without inspecting array values."""
        if type(self.name) is not str:
            raise TypeError("name must be a built-in string")
        if not self.name:
            raise ValueError("name must be nonempty")

    def execute(self) -> Float64Array:
        """Return the original array after finite nonnegative validation."""
        if type(self.value) is not np.ndarray:
            raise TypeError(f"{self.name} must be an exact numpy.ndarray")
        if self.value.dtype != np.dtype(np.float64):
            raise TypeError(f"{self.name} must have exact dtype float64")
        if not bool(np.logical_and(np.isfinite(self.value), self.value >= 0.0).all()):
            raise ValueError(f"{self.name} must be finite and nonnegative")
        return self.value


__all__ = [
    "NumpyFiniteArrayValidator",
    "NumpyFiniteNonnegativeFloat64ArrayValidator",
    "NumpyImmutableArrayConstructor",
    "NumpyImmutableArrayValidator",
]
