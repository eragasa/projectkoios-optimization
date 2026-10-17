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
    """Measure the complete normalized transform's matrix Parseval identity."""

    __slots__ = ()

    def analyze(
        self,
        definition: NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
        result: NumpyCenteredUniformMeshMatrixFourierTransformResult,
        /,
    ) -> FinitePeriodicParsevalDiagnostics:
        """Compare sample-mean and complete-coefficient squared Frobenius energies.

        Parameters
        ----------
        definition
            Complete source matrix samples.
        result
            Complete transform result for the same mesh and matrix size.

        Returns
        -------
        FinitePeriodicParsevalDiagnostics
            Threshold-free represented energies and absolute defect.

        Raises
        ------
        TypeError
            If either argument has the wrong exact nominal type.
        ValueError
            If mesh or matrix dimensions do not correlate.
        FloatingPointError
            If norm arithmetic becomes nonfinite.
        """
        if (
            type(definition)
            is not NumpyCenteredUniformMeshMatrixFourierTransformDefinition
        ):
            raise TypeError(
                "definition must be "
                "NumpyCenteredUniformMeshMatrixFourierTransformDefinition"
            )
        if type(result) is not NumpyCenteredUniformMeshMatrixFourierTransformResult:
            raise TypeError(
                "result must be NumpyCenteredUniformMeshMatrixFourierTransformResult"
            )
        if result.mesh != definition.mesh:
            raise ValueError("result mesh must equal definition mesh")
        if result.series.matrix_size != definition.matrix_size:
            raise ValueError("result matrix size must equal definition matrix size")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sample_energy = float(
                np.sum(np.abs(definition.matrix_samples) ** 2)
                / definition.mesh.point_count
            )
            coefficient_energy = float(
                np.sum(np.abs(result.series.matrix_coefficients) ** 2)
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
    """Measure origin-aware cyclic Hermiticity of complete matrix coefficients."""

    __slots__ = ()

    def analyze(
        self,
        result: NumpyCenteredUniformMeshMatrixFourierTransformResult,
        /,
    ) -> FinitePeriodicHermiticityDiagnostics:
        """Compare every mode with its centered cyclic negative.

        Parameters
        ----------
        result
            Complete transform result whose mesh origin determines every wrap phase.

        Returns
        -------
        FinitePeriodicHermiticityDiagnostics
            Maximum twisted cyclic and non-Nyquist ordinary Frobenius defects.

        Raises
        ------
        TypeError
            If ``result`` has the wrong exact nominal type.
        FloatingPointError
            If phase or norm arithmetic becomes nonfinite.
        """
        if type(result) is not NumpyCenteredUniformMeshMatrixFourierTransformResult:
            raise TypeError(
                "result must be NumpyCenteredUniformMeshMatrixFourierTransformResult"
            )
        mesh = result.mesh
        size = mesh.size
        lower = -(size // 2)
        upper = size // 2 - 1
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
        grid = result.series.matrix_coefficients.reshape(
            (
                *(size,) * mesh.dimension,
                result.series.matrix_size,
                result.series.matrix_size,
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
