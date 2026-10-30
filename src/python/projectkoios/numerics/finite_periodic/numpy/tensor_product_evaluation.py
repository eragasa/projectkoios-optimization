"""Separable NumPy tensor-grid evaluation of finite matrix Fourier series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast, final

import numpy as np

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.numerics.numpy.validation import (
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)
from projectkoios.numerics.validation import exact_positive_integer

from .matrix_evaluation import NumpyFiniteFourierMatrixEvaluationResult
from .matrix_series import NumpyFiniteFourierMatrixSeries


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyTensorProductFourierMatrixEvaluationDefinition:
    """Define separable tensor-grid evaluation of a finite matrix series.

    Parameters
    ----------
    source_mesh
        Centered construction mesh defining the available integer-mode box.
    series
        Complete or selected modes lying inside the construction-mesh inventory.
    coordinate_axes
        Canonical immutable float64 coordinate axes, one nonempty one-dimensional array
        per mesh dimension.
    maximum_working_elements
        Positive upper bound on every coefficient grid, phase matrix, and contraction
        intermediate. The retained output array is excluded from this bound.

    Notes
    -----
    This definition is intended for complete or large tensor-grid evaluations. Small
    arbitrary selected inventories should use
    :class:`NumpyFiniteFourierMatrixEvaluationDefinition`. Use :meth:`from_arrays` for
    defensive copies or :meth:`from_immutable_arrays` for an explicit zero-copy
    boundary.
    """

    source_mesh: CenteredUniformMesh
    series: NumpyFiniteFourierMatrixSeries
    coordinate_axes: tuple[Float64Array, ...] = field(repr=False)
    maximum_working_elements: int = 1_000_000

    @classmethod
    def from_arrays(
        cls,
        source_mesh: CenteredUniformMesh,
        series: NumpyFiniteFourierMatrixSeries,
        coordinate_axes: tuple[Float64Array, ...],
        maximum_working_elements: int = 1_000_000,
    ) -> Self:
        """Construct with immutable defensive copies of exact coordinate axes.

        Parameters
        ----------
        source_mesh
            Centered construction mesh.
        series
            Canonical finite matrix Fourier series.
        coordinate_axes
            Exact finite float64 coordinate-axis arrays.
        maximum_working_elements
            Positive working-array element bound.

        Returns
        -------
        NumpyTensorProductFourierMatrixEvaluationDefinition
            Canonical independently owned tensor-product evaluation definition.

        Raises
        ------
        TypeError
            If ``coordinate_axes`` is not an exact tuple or an axis is not float64.
        ValueError
            If an axis contains a nonfinite value.
        """
        if type(coordinate_axes) is not tuple:
            raise TypeError("coordinate_axes must be an exact tuple")
        return cls(
            source_mesh,
            series,
            tuple(
                NumpyImmutableArrayConstructor(
                    axis,
                    f"coordinate_axes[{index}]",
                    np.dtype(np.float64),
                ).execute()
                for index, axis in enumerate(coordinate_axes)
            ),
            maximum_working_elements,
        )

    @classmethod
    def from_immutable_arrays(
        cls,
        source_mesh: CenteredUniformMesh,
        series: NumpyFiniteFourierMatrixSeries,
        coordinate_axes: tuple[Float64Array, ...],
        maximum_working_elements: int = 1_000_000,
    ) -> Self:
        """Construct without copying canonical immutable coordinate axes.

        Parameters
        ----------
        source_mesh
            Centered construction mesh.
        series
            Canonical finite matrix Fourier series.
        coordinate_axes
            Canonical immutable finite float64 coordinate-axis arrays.
        maximum_working_elements
            Positive working-array element bound.

        Returns
        -------
        NumpyTensorProductFourierMatrixEvaluationDefinition
            Definition retaining all coordinate axes by identity.
        """
        return cls(
            source_mesh,
            series,
            coordinate_axes,
            maximum_working_elements,
        )

    def __post_init__(self) -> None:
        """Validate mesh correlation, coordinate axes, inventory, and memory bound.

        Raises
        ------
        TypeError
            If a field has the wrong exact nominal or array representation.
        ValueError
            If dimensions, axes, mode bounds, or working-memory bounds are invalid.
        """
        if (type(self.source_mesh), type(self.series)) != (
            CenteredUniformMesh,
            NumpyFiniteFourierMatrixSeries,
        ):
            raise TypeError(
                "source_mesh and series must have their exact public record types"
            )
        if self.series.dimension != self.source_mesh.dimension:
            raise ValueError("series dimension must equal source_mesh dimension")
        if type(self.coordinate_axes) is not tuple:
            raise TypeError("coordinate_axes must be an exact tuple")
        if len(self.coordinate_axes) != self.source_mesh.dimension:
            raise ValueError("one coordinate axis is required per mesh dimension")
        validated_axes = tuple(
            NumpyImmutableArrayValidator(
                axis,
                f"coordinate_axes[{index}]",
                np.dtype(np.float64),
            ).execute()
            for index, axis in enumerate(self.coordinate_axes)
        )
        if not all(axis.ndim == 1 and axis.size > 0 for axis in validated_axes):
            raise ValueError("coordinate axes must be nonempty one-dimensional arrays")
        exact_positive_integer(
            self.maximum_working_elements,
            "maximum_working_elements",
        )
        lower = -(self.source_mesh.size // 2)
        upper = (self.source_mesh.size - 1) // 2
        modes = self.series.integer_modes
        if bool(((modes < lower) | (modes > upper)).any()):
            raise ValueError("series modes must lie inside the source mesh inventory")
        required_minimum = max(
            self.coefficient_grid_elements,
            self.maximum_nonleading_phase_elements,
            self.elements_per_leading_axis_point,
        )
        if required_minimum > self.maximum_working_elements:
            raise ValueError("one required working array exceeds the working bound")

    @property
    def point_shape(self) -> tuple[int, ...]:
        """Return the tensor-product evaluation shape.

        Returns
        -------
        tuple of int
            Positive coordinate-axis lengths in declared order.
        """
        return tuple(int(axis.size) for axis in self.coordinate_axes)

    @property
    def coefficient_grid_elements(self) -> int:
        """Return the dense construction-grid coefficient element count.

        Returns
        -------
        int
            ``source_mesh.point_count * matrix_size**2``.
        """
        return self.source_mesh.point_count * self.series.matrix_size**2

    @property
    def maximum_nonleading_phase_elements(self) -> int:
        """Return the largest complete nonleading axis phase-matrix size.

        Returns
        -------
        int
            Largest axis phase size after the first axis, or zero in one dimension.
        """
        if self.source_mesh.dimension == 1:
            return 0
        return max(
            int(axis.size) * self.source_mesh.size for axis in self.coordinate_axes[1:]
        )

    @property
    def elements_per_leading_axis_point(self) -> int:
        """Return the largest contraction size attributable to one leading point.

        Returns
        -------
        int
            Maximum intermediate or output-slice element count for one first-axis
            coordinate.
        """
        size = self.source_mesh.size
        matrix_elements = self.series.matrix_size**2
        if self.source_mesh.dimension == 1:
            return max(size, matrix_elements)
        second = int(self.coordinate_axes[1].size)
        if self.source_mesh.dimension == 2:
            return max(size * matrix_elements, second * matrix_elements)
        third = int(self.coordinate_axes[2].size)
        return max(
            size * size * matrix_elements,
            second * size * matrix_elements,
            second * third * matrix_elements,
        )

    @property
    def leading_axis_batch_size(self) -> int:
        """Return the positive batch size satisfying the working-element bound.

        Returns
        -------
        int
            Number of leading-axis coordinates processed per contraction batch.
        """
        permitted = (
            self.maximum_working_elements // self.elements_per_leading_axis_point
        )
        return min(int(self.coordinate_axes[0].size), permitted)

    @property
    def maximum_temporary_elements(self) -> int:
        """Return the derived largest non-output working-array element count.

        Returns
        -------
        int
            Maximum coefficient-grid, phase-matrix, or contraction-batch size.
        """
        return max(
            self.coefficient_grid_elements,
            self.maximum_nonleading_phase_elements,
            self.leading_axis_batch_size * self.elements_per_leading_axis_point,
        )


@final
class NumpyTensorProductFourierMatrixSeriesEvaluator:
    """Evaluate matrix series on tensor grids by separable contractions."""

    __slots__ = ()

    def evaluate(
        self,
        definition: NumpyTensorProductFourierMatrixEvaluationDefinition,
        /,
    ) -> NumpyFiniteFourierMatrixEvaluationResult:
        """Evaluate a complete or selected series on a Cartesian-product point grid.

        Parameters
        ----------
        definition
            Canonical construction mesh, series, coordinate axes, and working bound.

        Returns
        -------
        NumpyFiniteFourierMatrixEvaluationResult
            Immutable tensor-grid matrix values and NumPy backend evidence.

        Raises
        ------
        TypeError
            If ``definition`` has the wrong exact nominal type.
        FloatingPointError
            If phase construction or a contraction becomes nonfinite.
        """
        if type(definition) is not NumpyTensorProductFourierMatrixEvaluationDefinition:
            raise TypeError(
                "definition must be NumpyTensorProductFourierMatrixEvaluationDefinition"
            )
        mesh = definition.source_mesh
        series = definition.series
        size = mesh.size
        matrix_size = series.matrix_size
        coefficient_grid = np.zeros(
            (*(size,) * mesh.dimension, matrix_size, matrix_size),
            dtype=np.complex128,
        )
        indices = tuple(
            series.integer_modes[:, component] + size // 2
            for component in range(mesh.dimension)
        )
        np.add.at(coefficient_grid, indices, series.matrix_coefficients)
        integer_axis = np.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=np.float64,
        )
        nonleading_phases: list[Complex128Array] = []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for coordinates in definition.coordinate_axes[1:]:
                phases = np.exp(
                    np.complex128(2.0j * np.pi)
                    * coordinates[:, None].astype(np.complex128)
                    * integer_axis[None, :].astype(np.complex128)
                )
                nonleading_phases.append(cast(Complex128Array, phases))
            output = np.empty(
                (*definition.point_shape, matrix_size, matrix_size),
                dtype=np.complex128,
            )
            leading_coordinates = definition.coordinate_axes[0]
            batch_size = definition.leading_axis_batch_size
            for start in range(0, int(leading_coordinates.size), batch_size):
                stop = min(start + batch_size, int(leading_coordinates.size))
                leading_phases = np.exp(
                    np.complex128(2.0j * np.pi)
                    * leading_coordinates[start:stop, None].astype(np.complex128)
                    * integer_axis[None, :].astype(np.complex128)
                )
                if mesh.dimension == 1:
                    output[start:stop] = np.einsum(
                        "ar,rij->aij",
                        leading_phases,
                        coefficient_grid,
                        optimize=True,
                    )
                elif mesh.dimension == 2:
                    first = np.einsum(
                        "ar,rsij->asij",
                        leading_phases,
                        coefficient_grid,
                        optimize=True,
                    )
                    output[start:stop] = np.einsum(
                        "bs,asij->abij",
                        nonleading_phases[0],
                        first,
                        optimize=True,
                    )
                else:
                    first = np.einsum(
                        "ar,rstij->astij",
                        leading_phases,
                        coefficient_grid,
                        optimize=True,
                    )
                    second = np.einsum(
                        "bs,astij->abtij",
                        nonleading_phases[0],
                        first,
                        optimize=True,
                    )
                    output[start:stop] = np.einsum(
                        "ct,abtij->abcij",
                        nonleading_phases[1],
                        second,
                        optimize=True,
                    )
        if not bool(np.isfinite(output).all()):
            raise FloatingPointError(
                "tensor-product finite Fourier evaluation became nonfinite"
            )
        return NumpyFiniteFourierMatrixEvaluationResult.from_array(
            dimension=series.dimension,
            point_shape=definition.point_shape,
            term_count=series.term_count,
            matrix_values=cast(Complex128Array, output),
            backend_identifier="numpy.einsum.tensor_product",
            backend_version=np.__version__,
        )


__all__ = [
    "NumpyTensorProductFourierMatrixEvaluationDefinition",
    "NumpyTensorProductFourierMatrixSeriesEvaluator",
]
