"""NumPy diagnostics for complete centered uniform matrix transforms."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from projectkoios.numerics.finite_periodic.transforms import (
    FinitePeriodicHermiticityDiagnostics,
    FinitePeriodicParsevalDiagnostics,
)

from .transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformResult,
)


@final
class NumpyFinitePeriodicParsevalAnalyzer:
    """Measure Parseval identity for one constructor-bound complete transform."""

    __slots__ = ("_definition", "_transform_result")

    def __init__(
        self,
        *,
        definition: NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
        transform_result: NumpyCenteredUniformMeshMatrixFourierTransformResult,
    ) -> None:
        if (
            type(definition)
            is not NumpyCenteredUniformMeshMatrixFourierTransformDefinition
        ):
            raise TypeError(
                "definition must be "
                "NumpyCenteredUniformMeshMatrixFourierTransformDefinition"
            )
        if (
            type(transform_result)
            is not NumpyCenteredUniformMeshMatrixFourierTransformResult
        ):
            raise TypeError(
                "transform_result must be "
                "NumpyCenteredUniformMeshMatrixFourierTransformResult"
            )
        if transform_result.mesh != definition.mesh:
            raise ValueError("transform_result mesh must equal definition mesh")
        if transform_result.series.matrix_size != definition.matrix_size:
            raise ValueError(
                "transform_result matrix size must equal definition matrix size"
            )
        self._definition = definition
        self._transform_result = transform_result

    def result(self) -> FinitePeriodicParsevalDiagnostics:
        """Compare sample-mean and complete-coefficient squared Frobenius energies.

        Returns
        -------
        FinitePeriodicParsevalDiagnostics
            Threshold-free represented energies and absolute defect.

        Raises
        ------
        FloatingPointError
            If norm arithmetic becomes nonfinite.
        """
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sample_energy = float(
                np.sum(np.abs(self._definition.matrix_samples) ** 2)
                / self._definition.mesh.point_count
            )
            coefficient_energy = float(
                np.sum(np.abs(self._transform_result.series.matrix_coefficients) ** 2)
            )
            defect = abs(sample_energy - coefficient_energy)
        if not all(
            math.isfinite(value)
            for value in (sample_energy, coefficient_energy, defect)
        ):
            raise FloatingPointError(
                "finite periodic Parseval arithmetic became nonfinite"
            )
        return FinitePeriodicParsevalDiagnostics(
            sample_energy,
            coefficient_energy,
            defect,
        )


@final
class NumpyCenteredUniformMeshHermiticityAnalyzer:
    """Measure Hermiticity for one constructor-bound complete transform."""

    __slots__ = ("_transform_result",)

    def __init__(
        self,
        *,
        transform_result: NumpyCenteredUniformMeshMatrixFourierTransformResult,
    ) -> None:
        if (
            type(transform_result)
            is not NumpyCenteredUniformMeshMatrixFourierTransformResult
        ):
            raise TypeError(
                "transform_result must be "
                "NumpyCenteredUniformMeshMatrixFourierTransformResult"
            )
        self._transform_result = transform_result

    def result(self) -> FinitePeriodicHermiticityDiagnostics:
        """Compare every mode with its centered cyclic negative.

        Returns
        -------
        FinitePeriodicHermiticityDiagnostics
            Maximum twisted cyclic and non-Nyquist ordinary Frobenius defects.

        Raises
        ------
        FloatingPointError
            If phase or norm arithmetic becomes nonfinite.
        """
        mesh = self._transform_result.mesh
        size = mesh.size
        lower = -(size // 2)
        upper = (size - 1) // 2
        axis_modes = np.arange(lower, upper + 1, dtype=np.int64)
        target_modes = -axis_modes
        wraps = np.zeros(size, dtype=np.int64)
        above = target_modes > upper
        below = target_modes < lower
        target_modes[above] -= size
        wraps[above] = -1
        target_modes[below] += size
        wraps[below] = 1
        target_indices = target_modes - lower
        grid = self._transform_result.series.matrix_coefficients.reshape(
            (
                *(size,) * mesh.dimension,
                self._transform_result.series.matrix_size,
                self._transform_result.series.matrix_size,
            )
        )
        partner = grid[np.ix_(*(target_indices,) * mesh.dimension)]
        phase_arguments = np.zeros((size,) * mesh.dimension, dtype=np.float64)
        for component in range(mesh.dimension):
            shape = [1] * mesh.dimension
            shape[component] = size
            phase_arguments = phase_arguments + (
                mesh.coordinate_origin[component] * wraps.reshape(tuple(shape))
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            wrap_phases = np.exp(
                np.complex128(-2.0j * np.pi * size)
                * phase_arguments.astype(np.complex128)
            )
            expected = wrap_phases[..., None, None] * np.swapaxes(
                grid.conjugate(), -1, -2
            )
            defects = np.linalg.norm(partner - expected, axis=(-2, -1))
            twisted = float(np.max(defects))
            ordinary_mask = np.ones((size,) * mesh.dimension, dtype=np.bool_)
            if size % 2 == 0:
                for component in range(mesh.dimension):
                    shape = [1] * mesh.dimension
                    shape[component] = size
                    ordinary_mask &= axis_modes.reshape(tuple(shape)) != lower
            ordinary = float(np.max(defects[ordinary_mask]))
        if not math.isfinite(twisted) or not math.isfinite(ordinary):
            raise FloatingPointError(
                "finite periodic Hermiticity arithmetic became nonfinite"
            )
        return FinitePeriodicHermiticityDiagnostics(twisted, ordinary)


__all__ = [
    "NumpyCenteredUniformMeshHermiticityAnalyzer",
    "NumpyFinitePeriodicParsevalAnalyzer",
]
