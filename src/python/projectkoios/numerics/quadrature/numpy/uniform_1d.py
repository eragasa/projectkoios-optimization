"""NumPy uniform closed-interval trapezoidal quadrature.

This module constructs immutable binary64 weights for the complete endpoint-inclusive
rule and for its homogeneous-Dirichlet interior-node restriction. The only mass matrix
provided is the diagonal lumped matrix whose diagonal equals the applicable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, final

import numpy as np

from projectkoios.numerics.numpy.typing import Float64Array, Int64Array
from projectkoios.numerics.numpy.validation import (
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)
from projectkoios.numerics.validation import exact_finite_float

_MASS_MATRIX_CONVENTION = "diagonal-lumped-diag-weights"


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyDiagonalQuadratureMassMatrix:
    """Store one positive diagonal quadrature mass matrix without dense expansion.

    Parameters
    ----------
    diagonal
        Canonical immutable float64 array containing the positive diagonal entries.
    convention
        Fixed identity ``"diagonal-lumped-diag-weights"``. Consistent finite-element
        mass matrices and generalized eigensolver behavior are outside this capability.

    Notes
    -----
    Use :meth:`from_array` to obtain a defensive immutable copy. The raw constructor is
    the canonical zero-copy boundary used when an owning result already retains
    immutable byte-backed weights.
    """

    diagonal: Float64Array = field(repr=False)
    convention: str = field(default=_MASS_MATRIX_CONVENTION, init=False)

    @classmethod
    def from_array(cls, diagonal: Float64Array, /) -> Self:
        """Construct from an exact float64 vector using a defensive copy."""
        return cls(
            NumpyImmutableArrayConstructor(
                diagonal,
                "diagonal",
                np.dtype(np.float64),
            ).execute()
        )

    def __post_init__(self) -> None:
        """Validate canonical one-dimensional positive diagonal storage."""
        diagonal = NumpyImmutableArrayValidator(
            self.diagonal,
            "diagonal",
            np.dtype(np.float64),
        ).execute()
        if diagonal.ndim != 1 or diagonal.size == 0:
            raise ValueError("diagonal must be a nonempty one-dimensional array")
        if bool((diagonal <= 0.0).any()):
            raise ValueError("diagonal entries must be positive")

    @property
    def dimension(self) -> int:
        """Return the positive mass-matrix dimension."""
        return int(self.diagonal.size)

    def to_dense(self) -> Float64Array:
        """Return a defensive immutable dense representation of ``diag(diagonal)``."""
        dense = np.diag(self.diagonal)
        return NumpyImmutableArrayConstructor(
            dense,
            "dense_mass_matrix",
            np.dtype(np.float64),
        ).execute()


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyHomogeneousDirichletInteriorQuadrature1D:
    """Record the explicit interior-node restriction of one full rule.

    Parameters
    ----------
    full_rule
        Complete endpoint-inclusive trapezoidal result with at least three nodes.
    full_grid_indices
        Canonical immutable int64 indices ``[1, ..., N - 2]`` into the full grid.
    interior_weights
        Canonical immutable float64 weights at those nodes. Every entry equals the full
        grid spacing because the endpoint values are declared homogeneous Dirichlet.

    Notes
    -----
    The restriction is equivalent to the full trapezoidal bilinear form only when the
    represented factors vanish at both endpoints. It is not a generic rule for dropping
    nonzero endpoint data.
    """

    full_rule: NumpyUniformClosedIntervalTrapezoidalRuleResult
    full_grid_indices: Int64Array = field(repr=False)
    interior_weights: Float64Array = field(repr=False)
    restriction_convention: str = field(
        default="homogeneous-dirichlet-interior-nodes",
        init=False,
    )

    @classmethod
    def from_arrays(
        cls,
        full_rule: NumpyUniformClosedIntervalTrapezoidalRuleResult,
        full_grid_indices: Int64Array,
        interior_weights: Float64Array,
        /,
    ) -> Self:
        """Construct from exact arrays using independent defensive copies."""
        return cls(
            full_rule,
            NumpyImmutableArrayConstructor(
                full_grid_indices,
                "full_grid_indices",
                np.dtype(np.int64),
            ).execute(),
            NumpyImmutableArrayConstructor(
                interior_weights,
                "interior_weights",
                np.dtype(np.float64),
            ).execute(),
        )

    def __post_init__(self) -> None:
        """Validate full-grid correlation, canonical indices, and equal weights."""
        if type(self.full_rule) is not NumpyUniformClosedIntervalTrapezoidalRuleResult:
            raise TypeError(
                "full_rule must be NumpyUniformClosedIntervalTrapezoidalRuleResult"
            )
        definition = self.full_rule.definition
        if definition.point_count < 3:
            raise ValueError(
                "homogeneous-Dirichlet interior restriction requires at least three "
                "full-grid points"
            )
        indices = NumpyImmutableArrayValidator(
            self.full_grid_indices,
            "full_grid_indices",
            np.dtype(np.int64),
        ).execute()
        weights = NumpyImmutableArrayValidator(
            self.interior_weights,
            "interior_weights",
            np.dtype(np.float64),
        ).execute()
        expected_shape = (definition.interior_point_count,)
        if indices.shape != expected_shape:
            raise ValueError("full_grid_indices must contain every interior node")
        if weights.shape != expected_shape:
            raise ValueError(
                "interior_weights must contain one weight per interior node"
            )
        expected_indices = np.arange(1, definition.point_count - 1, dtype=np.int64)
        if not bool(np.array_equal(indices, expected_indices)):
            raise ValueError("full_grid_indices must equal [1, ..., N - 2]")
        if not bool(np.array_equal(weights, self.full_rule.full_weights[indices])):
            raise ValueError("interior_weights must be the indexed full-grid weights")
        if not bool((weights == self.full_rule.spacing).all()):
            raise ValueError("every interior weight must equal the full-grid spacing")

    @property
    def definition(self) -> UniformClosedIntervalTrapezoidalRuleDefinition:
        """Return the correlated full-rule definition."""
        return self.full_rule.definition

    @property
    def interval(self) -> tuple[float, float]:
        """Return the correlated closed interval."""
        return self.definition.interval

    @property
    def point_count(self) -> int:
        """Return the complete endpoint-inclusive point count."""
        return self.definition.point_count

    @property
    def spacing(self) -> float:
        """Return the retained full-grid spacing."""
        return self.full_rule.spacing

    @property
    def rule_identifier(self) -> str:
        """Return the versioned rule identity."""
        return self.definition.rule_identifier

    @property
    def rule_version(self) -> int:
        """Return the positive rule contract version."""
        return self.definition.rule_version

    @property
    def mass_matrix(self) -> NumpyDiagonalQuadratureMassMatrix:
        """Return the diagonal lumped mass matrix retaining the interior weights."""
        return NumpyDiagonalQuadratureMassMatrix(self.interior_weights)


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyUniformClosedIntervalTrapezoidalRuleResult:
    """Record one complete endpoint-inclusive uniform trapezoidal rule.

    Parameters
    ----------
    definition
        Unit-free interval and endpoint-inclusive point count with fixed convention
        metadata and rule identity.
    spacing
        Finite positive represented spacing ``(b - a) / (N - 1)``.
    full_weights
        Canonical immutable float64 array with endpoint weights ``spacing / 2`` and
        interior weights ``spacing``.

    Notes
    -----
    Use :meth:`from_array` for a defensive array copy. The result provides an explicit
    homogeneous-Dirichlet restriction rather than silently discarding endpoints.
    """

    definition: UniformClosedIntervalTrapezoidalRuleDefinition
    spacing: float
    full_weights: Float64Array = field(repr=False)

    @classmethod
    def from_array(
        cls,
        definition: UniformClosedIntervalTrapezoidalRuleDefinition,
        spacing: float,
        full_weights: Float64Array,
        /,
    ) -> Self:
        """Construct from an exact float64 vector using a defensive copy."""
        return cls(
            definition,
            spacing,
            NumpyImmutableArrayConstructor(
                full_weights,
                "full_weights",
                np.dtype(np.float64),
            ).execute(),
        )

    def __post_init__(self) -> None:
        """Validate canonical weights and reconstruct the complete rule relation."""
        if type(self.definition) is not UniformClosedIntervalTrapezoidalRuleDefinition:
            raise TypeError(
                "definition must be UniformClosedIntervalTrapezoidalRuleDefinition"
            )
        spacing = exact_finite_float(self.spacing, "spacing")
        if spacing != self.definition.spacing:
            raise ValueError("spacing must equal the definition spacing")
        weights = NumpyImmutableArrayValidator(
            self.full_weights,
            "full_weights",
            np.dtype(np.float64),
        ).execute()
        if weights.shape != (self.definition.point_count,):
            raise ValueError("full_weights must contain one weight per full-grid node")
        expected = np.full(
            self.definition.point_count,
            spacing,
            dtype=np.float64,
        )
        expected[0] = 0.5 * spacing
        expected[-1] = 0.5 * spacing
        if not bool(np.array_equal(weights, expected)):
            raise ValueError(
                "full_weights must equal the composite trapezoidal weights"
            )

    @property
    def interval(self) -> tuple[float, float]:
        """Return the ordered closed interval."""
        return self.definition.interval

    @property
    def point_count(self) -> int:
        """Return the endpoint-inclusive node count."""
        return self.definition.point_count

    @property
    def rule_identifier(self) -> str:
        """Return the versioned rule identity."""
        return self.definition.rule_identifier

    @property
    def rule_version(self) -> int:
        """Return the positive rule contract version."""
        return self.definition.rule_version

    @property
    def full_mass_matrix(self) -> NumpyDiagonalQuadratureMassMatrix:
        """Return the complete diagonal lumped mass matrix without copying weights."""
        return NumpyDiagonalQuadratureMassMatrix(self.full_weights)

    def restrict_homogeneous_dirichlet(
        self,
    ) -> NumpyHomogeneousDirichletInteriorQuadrature1D:
        """Return weights at full-grid indices ``1`` through ``N - 2``.

        Raises
        ------
        ValueError
            If the complete grid has fewer than three points and therefore has no
            homogeneous-Dirichlet interior node.
        """
        if self.point_count < 3:
            raise ValueError(
                "homogeneous-Dirichlet interior restriction requires at least three "
                "full-grid points"
            )
        indices = np.arange(1, self.point_count - 1, dtype=np.int64)
        weights = np.full(indices.size, self.spacing, dtype=np.float64)
        return NumpyHomogeneousDirichletInteriorQuadrature1D.from_arrays(
            self,
            indices,
            weights,
        )


@final
class NumpyUniformClosedIntervalTrapezoidalRule:
    """Construct complete and homogeneous-Dirichlet uniform quadrature weights."""

    __slots__ = ()

    def calculate(
        self,
        definition: UniformClosedIntervalTrapezoidalRuleDefinition,
        /,
    ) -> NumpyUniformClosedIntervalTrapezoidalRuleResult:
        """Calculate the fixed composite trapezoidal weights.

        Parameters
        ----------
        definition
            Exact interval and endpoint-inclusive point count.

        Returns
        -------
        NumpyUniformClosedIntervalTrapezoidalRuleResult
            Immutable complete weights with an explicit interior-restriction operation.
        """
        if type(definition) is not UniformClosedIntervalTrapezoidalRuleDefinition:
            raise TypeError(
                "definition must be UniformClosedIntervalTrapezoidalRuleDefinition"
            )
        spacing = definition.spacing
        weights = np.full(definition.point_count, spacing, dtype=np.float64)
        weights[0] = 0.5 * spacing
        weights[-1] = 0.5 * spacing
        return NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(
            definition,
            spacing,
            weights,
        )


__all__ = [
    "NumpyDiagonalQuadratureMassMatrix",
    "NumpyHomogeneousDirichletInteriorQuadrature1D",
    "NumpyUniformClosedIntervalTrapezoidalRule",
    "NumpyUniformClosedIntervalTrapezoidalRuleResult",
]
