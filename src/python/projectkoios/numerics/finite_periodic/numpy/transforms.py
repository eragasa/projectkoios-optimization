"""NumPy forward transforms for centered uniform matrix meshes.

The capability implements complete one-, two-, and three-dimensional tensor-product
forward transforms with explicit mesh offsets, negative exponent, and ``1 / N**d``
normalization. Matrix-series evaluation and diagnostics live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast, final

import numpy as np

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    _backend_evidence,
)
from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.numpy.typing import Complex128Array, Int64Array
from projectkoios.numerics.numpy.validation import (
    immutable_complex128_array,
    validate_immutable_complex128_array,
)

from .matrix_series import NumpyFiniteFourierMatrixSeries


def _centered_integer_modes(mesh: CenteredUniformMesh, /) -> Int64Array:
    """Return mutable lexicographically ordered centered modes for one mesh.

    Parameters
    ----------
    mesh
        Validated centered uniform mesh.

    Returns
    -------
    numpy.ndarray
        Int64 array with shape ``(mesh.point_count, mesh.dimension)``.
    """
    axis = np.arange(-(mesh.size // 2), mesh.size // 2, dtype=np.int64)
    components = np.meshgrid(*(axis,) * mesh.dimension, indexing="ij")
    return cast(
        Int64Array,
        np.stack(components, axis=-1).reshape((-1, mesh.dimension)),
    )


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyCenteredUniformMeshMatrixFourierTransformDefinition:
    """Define matrix samples on one complete centered uniform tensor mesh.

    Parameters
    ----------
    mesh
        Exact centered uniform mesh metadata.
    matrix_samples
        Canonical immutable complex128 array with shape
        ``(mesh.size,) * mesh.dimension + (n, n)`` for positive ``n``.

    Notes
    -----
    Use :meth:`from_array` for a defensive copy or :meth:`from_immutable_array` for an
    explicit zero-copy boundary.
    """

    mesh: CenteredUniformMesh
    matrix_samples: Complex128Array = field(repr=False)

    @classmethod
    def from_array(
        cls,
        mesh: CenteredUniformMesh,
        matrix_samples: Complex128Array,
        /,
    ) -> Self:
        """Construct from an exact complex128 array using a defensive copy.

        Parameters
        ----------
        mesh
            Centered uniform mesh metadata.
        matrix_samples
            Exact finite complex128 tensor of square matrices.

        Returns
        -------
        NumpyCenteredUniformMeshMatrixFourierTransformDefinition
            Canonical independently owned transform definition.
        """
        return cls(
            mesh,
            immutable_complex128_array(matrix_samples, "matrix_samples"),
        )

    @classmethod
    def from_immutable_array(
        cls,
        mesh: CenteredUniformMesh,
        matrix_samples: Complex128Array,
        /,
    ) -> Self:
        """Construct without copying a canonical immutable sample array.

        Parameters
        ----------
        mesh
            Centered uniform mesh metadata.
        matrix_samples
            Canonical immutable finite complex128 tensor of square matrices.

        Returns
        -------
        NumpyCenteredUniformMeshMatrixFourierTransformDefinition
            Definition retaining the supplied array by identity.
        """
        return cls(mesh, matrix_samples)

    def __post_init__(self) -> None:
        """Validate the nominal mesh and complete tensor sample shape.

        Raises
        ------
        TypeError
            If the mesh or sample representation has the wrong exact type.
        ValueError
            If sample rank, tensor shape, matrix shape, or finiteness is invalid.
        """
        if type(self.mesh) is not CenteredUniformMesh:
            raise TypeError("mesh must be CenteredUniformMesh")
        samples = validate_immutable_complex128_array(
            self.matrix_samples,
            "matrix_samples",
        )
        if samples.ndim != self.mesh.dimension + 2:
            raise ValueError("matrix_samples rank is inconsistent with mesh dimension")
        if (
            samples.shape[: self.mesh.dimension]
            != (self.mesh.size,) * self.mesh.dimension
        ):
            raise ValueError("matrix_samples must contain the complete tensor mesh")
        if samples.shape[-2] <= 0 or samples.shape[-1] != samples.shape[-2]:
            raise ValueError("matrix samples must be nonempty and square")

    @property
    def matrix_size(self) -> int:
        """Return the shared square-matrix dimension.

        Returns
        -------
        int
            Positive matrix dimension.
        """
        return int(self.matrix_samples.shape[-1])


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyCenteredUniformMeshMatrixFourierTransformResult:
    """Record one complete normalized centered-mesh Fourier transform.

    Parameters
    ----------
    mesh
        Source mesh, including all axis offsets.
    series
        Complete centered integer-mode matrix series in tensor lexicographic order.
    backend_identifier, backend_version
        Exact executed NumPy backend evidence.
    """

    mesh: CenteredUniformMesh
    series: NumpyFiniteFourierMatrixSeries
    backend_identifier: str
    backend_version: str

    def __post_init__(self) -> None:
        """Validate complete inventory, mesh correlation, and backend evidence.

        Raises
        ------
        TypeError
            If a field has the wrong exact nominal representation.
        ValueError
            If dimension, count, or centered mode ordering is inconsistent.
        """
        if type(self.mesh) is not CenteredUniformMesh:
            raise TypeError("mesh must be CenteredUniformMesh")
        if type(self.series) is not NumpyFiniteFourierMatrixSeries:
            raise TypeError("series must be NumpyFiniteFourierMatrixSeries")
        if self.series.dimension != self.mesh.dimension:
            raise ValueError("series dimension must equal mesh dimension")
        if self.series.term_count != self.mesh.point_count:
            raise ValueError("series must contain one coefficient per mesh point")
        expected_modes = _centered_integer_modes(self.mesh)
        if not bool(np.array_equal(self.series.integer_modes, expected_modes)):
            raise ValueError("series modes must equal the centered mesh inventory")
        _backend_evidence(self.backend_identifier, self.backend_version)


@final
class NumpyCenteredUniformMeshMatrixFourierTransformer:
    """Transform complete centered uniform matrix samples into integer modes.

    Notes
    -----
    The forward convention is

    ``C_R = N**(-d) * sum_i F(q_i) * exp(-2*pi*1j*dot(q_i, R))``

    with ``q_i[a] = -1/2 + (i[a] + offsets[a]) / N``. NumPy FFT axes are
    shifted into the centered half-open inventory before the exact origin phase is
    applied.
    """

    __slots__ = ()

    def transform(
        self,
        definition: NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
        /,
    ) -> NumpyCenteredUniformMeshMatrixFourierTransformResult:
        """Execute one complete normalized forward transform.

        Parameters
        ----------
        definition
            Canonical complete mesh and matrix samples.

        Returns
        -------
        NumpyCenteredUniformMeshMatrixFourierTransformResult
            Complete centered modes, immutable matrix coefficients, and backend
            evidence.

        Raises
        ------
        TypeError
            If ``definition`` has the wrong exact nominal type.
        FloatingPointError
            If FFT or origin-phase arithmetic becomes nonfinite.
        """
        if (
            type(definition)
            is not NumpyCenteredUniformMeshMatrixFourierTransformDefinition
        ):
            raise TypeError(
                "definition must be "
                "NumpyCenteredUniformMeshMatrixFourierTransformDefinition"
            )
        mesh = definition.mesh
        axes = tuple(range(mesh.dimension))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = np.fft.fftn(
                definition.matrix_samples,
                axes=axes,
                norm="forward",
            )
            centered = np.fft.fftshift(transformed, axes=axes)
            modes = _centered_integer_modes(mesh)
            origin = mesh.coordinate_origin
            phase_arguments = np.zeros(mesh.point_count, dtype=np.float64)
            for component in range(mesh.dimension):
                phase_arguments = (
                    phase_arguments + modes[:, component] * origin[component]
                )
            phases = np.exp(
                np.complex128(-2.0j * np.pi) * phase_arguments.astype(np.complex128)
            )
            coefficients = (
                centered.reshape(
                    (mesh.point_count, definition.matrix_size, definition.matrix_size)
                )
                * phases[:, None, None]
            )
        if not bool(np.isfinite(coefficients).all()):
            raise FloatingPointError("centered uniform mesh transform became nonfinite")
        series = NumpyFiniteFourierMatrixSeries.from_arrays(
            mesh.dimension,
            modes,
            coefficients,
        )
        return NumpyCenteredUniformMeshMatrixFourierTransformResult(
            mesh,
            series,
            "numpy.fft.fftn",
            np.__version__,
        )


__all__ = [
    "NumpyCenteredUniformMeshMatrixFourierTransformDefinition",
    "NumpyCenteredUniformMeshMatrixFourierTransformer",
    "NumpyCenteredUniformMeshMatrixFourierTransformResult",
]
