"""Own Procrustes-specific vectorized numerical evidence and bounds.

Notes
-----
The operations in this module are pure array reductions shared by result validation and
solver assembly. They do not call a decomposition backend or cross the public façade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from projectkoios.numerics.numpy.typing import BoolArray, Float64Array
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesTolerance,
)


def batch_size(batch_shape: tuple[int, ...], /) -> int:
    """Return the exact number of leading-index items.

    Parameters
    ----------
    batch_shape
        Leading batch dimensions.

    Returns
    -------
    int
        Product of the dimensions, including zero.
    """
    return math.prod(batch_shape)


def batch_index(flat_index: int, batch_shape: tuple[int, ...], /) -> tuple[int, ...]:
    """Convert a flat C-order item index to its leading batch index.

    Parameters
    ----------
    flat_index
        Zero-based flat item index.
    batch_shape
        Nonempty leading shape.

    Returns
    -------
    tuple[int, ...]
        Deterministic leading index.
    """
    return tuple(
        int(component) for component in np.unravel_index(flat_index, batch_shape)
    )


def first_true_index(
    mask: BoolArray, batch_shape: tuple[int, ...], /
) -> tuple[int, ...]:
    """Return the first true leading index in C order.

    Parameters
    ----------
    mask
        Boolean item mask with the exact leading shape.
    batch_shape
        Nonempty leading shape.

    Returns
    -------
    tuple[int, ...]
        First failing leading index.
    """
    flat = int(np.flatnonzero(mask.reshape(-1))[0])
    return batch_index(flat, batch_shape)


def validation_bound(
    tolerance: ComplexUnitaryProcrustesTolerance,
    scale: Float64Array | float,
    count: int,
    /,
) -> Float64Array:
    """Return vectorized dimension-scaled validation bounds.

    Parameters
    ----------
    tolerance
        Existing owner software tolerances.
    scale
        Nonnegative scalar or per-item scale.
    count
        Positive operation-count bound.

    Returns
    -------
    numpy.ndarray
        Finite float64 validation bounds.
    """
    result = count * (
        tolerance.validation_absolute + tolerance.validation_relative * scale
    )
    array = np.asarray(result, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError("validation bound became nonfinite")
    return array


@dataclass(frozen=True, slots=True)
class ResidualReconstruction:
    """Hold roundoff-aware optimal squared-residual evidence.

    Parameters
    ----------
    squared_residuals
        Reconstructed nonnegative optimal values in squared-residual units.
    roundoff_cutoffs
        Applied cancellation cutoffs in the same squared units.

    Notes
    -----
    Instances are produced by :meth:`from_procrustes_evidence`; the record is
    encapsulated behind the batched Procrustes façade.
    """

    squared_residuals: Float64Array
    roundoff_cutoffs: Float64Array

    @classmethod
    def from_procrustes_evidence(
        cls,
        source_power: Float64Array,
        target_power: Float64Array,
        singular_values: Float64Array,
        /,
    ) -> ResidualReconstruction:
        """Reconstruct optimal squared residuals and their roundoff cutoffs.

        Parameters
        ----------
        source_power
            Per-item squared source Frobenius norms.
        target_power
            Per-item squared target Frobenius norms.
        singular_values
            Complete per-item cross singular spectra.

        Returns
        -------
        ResidualReconstruction
            Roundoff-aware residual values and squared-unit cutoffs.

        Raises
        ------
        FloatingPointError
            If arithmetic becomes nonfinite or a residual is materially negative.
        """
        twice_singular = 2.0 * singular_values
        if not bool(np.isfinite(twice_singular).all()):
            raise FloatingPointError("cross nuclear-norm contribution became nonfinite")
        scale = np.maximum(source_power, target_power)
        scale = np.maximum(scale, np.max(twice_singular, axis=-1))
        nonzero = scale > 0.0
        safe_scale = np.where(nonzero, scale, 1.0)
        normalized = source_power / safe_scale + target_power / safe_scale
        normalized = normalized - np.sum(
            twice_singular / safe_scale[..., None],
            axis=-1,
        )
        roundoff_coefficient = (
            ComplexUnitaryProcrustesSolver.residual_roundoff_coefficient(
                singular_values.shape[-1]
            )
        )
        normalized_absolute_sum = source_power / safe_scale + target_power / safe_scale
        normalized_absolute_sum = normalized_absolute_sum + np.sum(
            twice_singular / safe_scale[..., None],
            axis=-1,
        )
        roundoff_bound = roundoff_coefficient * normalized_absolute_sum
        cutoff = roundoff_bound * scale
        if not bool(np.isfinite(normalized).all()) or not bool(
            np.isfinite(cutoff).all()
        ):
            raise FloatingPointError("residual reconstruction became nonfinite")
        materially_negative = normalized < -roundoff_bound
        if bool(materially_negative.any()):
            index = first_true_index(
                np.asarray(materially_negative, dtype=np.bool_),
                source_power.shape,
            )
            raise FloatingPointError(
                "reconstructed optimal squared residual is materially negative "
                f"at batch index {index}"
            )
        normalized = np.where(np.abs(normalized) <= roundoff_bound, 0.0, normalized)
        squared = normalized * scale
        squared = np.where(nonzero, squared, 0.0)
        if not bool(np.isfinite(squared).all()):
            raise FloatingPointError("optimal squared residual became nonfinite")
        return cls(
            squared_residuals=np.asarray(squared, dtype=np.float64),
            roundoff_cutoffs=np.asarray(cutoff, dtype=np.float64),
        )
