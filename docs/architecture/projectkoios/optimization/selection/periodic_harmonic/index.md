# Periodic single-harmonic maximization

## Status

This page specifies the current owner contract for deterministic maximization of one
real periodic harmonic over one canonical fundamental interval.

## Ownership

`projectkoios.optimization.selection.harmonic` owns selection of the canonical maximizer
of

$$
f(\phi)=o+c\cos(m\phi)+s\sin(m\phi),
$$

where $o$, $c$, and $s$ are finite real coefficients and $m$ is a positive integer.
The operation belongs in Optimization because it selects one argument from a continuous
periodic family. It does not own the meaning or construction of the coefficients.

The capability is unit-free and contains no matrix-field, Hamiltonian, frame, Fourier,
threshold, or scientific-disposition semantics.

## Public API

```python
class PeriodicHarmonicMaximumStatus(StrEnum):
    UNIQUE = "unique"
    NUMERICALLY_CONSTANT = "numerically-constant"


@dataclass(frozen=True, slots=True)
class PeriodicHarmonic:
    offset: float
    cosine_coefficient: float
    sine_coefficient: float
    harmonic_order: int
    fundamental_interval_start: float


@dataclass(frozen=True, slots=True)
class PeriodicHarmonicMaximum:
    harmonic: PeriodicHarmonic
    maximum_value: float
    amplitude: float
    canonical_maximizing_argument: float | None
    fundamental_period: float
    applied_degeneracy_cutoff: float
    status: PeriodicHarmonicMaximumStatus


class PeriodicHarmonicMaximizer:
    def __init__(
        self,
        *,
        absolute_degeneracy_tolerance: float = 1.0e-12,
        relative_degeneracy_tolerance: float = 1.0e-10,
    ) -> None: ...

    def maximize(
        self,
        harmonic: PeriodicHarmonic,
        /,
    ) -> PeriodicHarmonicMaximum: ...
```

## Canonical interval

The harmonic defines the half-open interval

$$
[\alpha,\alpha+2\pi/m),
$$

where $\alpha$ is `fundamental_interval_start`. A half-open interval removes the
duplicate periodic endpoint. A unique maximum modulo the fundamental period therefore
has exactly one returned representative.

If the harmonic amplitude is tolerance-qualified as numerically constant, the result
returns the mathematical maximum value but sets `canonical_maximizing_argument` to
`None`. It does not fabricate a unique angle.

## Consumer boundary

Consumers retain coefficient derivation, normalization, units, parameter meaning,
comparison with finite inventories, evidence policy, and interpretation. In particular,
a consumer may form coefficients from weighted matrix-field correlations, but those
matrix records and weights do not enter this API.

## Navigation

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Selection ownership](../index.md)
