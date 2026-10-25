"""Shared precise array types for optional NumPy capability modules.

Importing this module imports NumPy. Dependency-free contracts must therefore use
built-in immutable records instead of these aliases; concrete NumPy implementations and
their tests may import the aliases explicitly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type NumpyArrayLike = npt.ArrayLike
type BoolArray = npt.NDArray[np.bool_]
type Complex128Array = npt.NDArray[np.complex128]
type Complex128Matrix = npt.NDArray[np.complex128]
type Float64Array = npt.NDArray[np.float64]
type Float64Matrix = npt.NDArray[np.float64]
type DenseMatrix = Float64Matrix | Complex128Matrix
type Int64Array = npt.NDArray[np.int64]
type IntpArray = npt.NDArray[np.intp]

__all__ = [
    "BoolArray",
    "Complex128Array",
    "Complex128Matrix",
    "DenseMatrix",
    "Float64Array",
    "Float64Matrix",
    "Int64Array",
    "IntpArray",
    "NumpyArrayLike",
]
