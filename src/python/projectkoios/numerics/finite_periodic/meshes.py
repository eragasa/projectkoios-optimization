"""Dependency-free records for finite periodic meshes.

The records in this module describe numerical mesh geometry only. They attach no
reciprocal-lattice units, physical interpretation, shell policy, or scientific
acceptance criterion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

from projectkoios.numerics.validation import exact_positive_integer


@final
@dataclass(frozen=True, slots=True)
class CenteredUniformMesh:
    """Describe a one-, two-, or three-dimensional centered uniform tensor mesh.

    Parameters
    ----------
    dimension
        Exact spatial dimension. This capability supports one, two, and three
        dimensions.
    size
        Positive number of points on every tensor axis.
    offsets
        Exact built-in finite offsets, one per axis. Axis ``a`` contains coordinates
        ``-1/2 + (i + offsets[a]) / size`` for ``i = 0, ..., size - 1``.

    Notes
    -----
    The centered integer inventory is ``-size//2, ..., (size - 1)//2``. Even sizes have
    one lower Nyquist boundary; odd sizes have ordinary inversion pairs and no Nyquist
    mode. Offsets are not reduced modulo one because the supplied origin is part of the
    transform convention and retained evidence.
    """

    dimension: int
    size: int
    offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate exact dimensions, positive size, and finite axis offsets.

        Raises
        ------
        TypeError
            If a field does not use its exact canonical built-in representation.
        ValueError
            If the dimension, size, offset count, or finiteness is invalid.
        """
        dimension = exact_positive_integer(self.dimension, "dimension")
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be one, two, or three")
        exact_positive_integer(self.size, "size")
        if type(self.offsets) is not tuple:
            raise TypeError("offsets must be an exact tuple")
        if len(self.offsets) != dimension:
            raise ValueError("offsets length must equal dimension")
        for offset in self.offsets:
            if type(offset) is not float:
                raise TypeError("offsets must contain exact built-in floats")
            if not math.isfinite(offset):
                raise ValueError("offsets must be finite")

    @property
    def point_count(self) -> int:
        """Return the tensor-product sample count ``size**dimension``.

        Returns
        -------
        int
            Positive number of mesh points.
        """
        return int(self.size**self.dimension)

    @property
    def coordinate_origin(self) -> tuple[float, ...]:
        """Return the first reduced coordinate on every tensor axis.

        Returns
        -------
        tuple of float
            Coordinates ``-1/2 + offsets[a] / size`` in declared axis order.
        """
        return tuple(-0.5 + offset / self.size for offset in self.offsets)


__all__ = ["CenteredUniformMesh"]
