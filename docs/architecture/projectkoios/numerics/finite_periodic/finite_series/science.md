# Finite Fourier-series interpretation boundary

## Status

This page records the current interpretation boundary of the implemented finite
Fourier-series evaluator. Consumer-owned physical meaning remains outside Numerics.

## Mathematical object

The capability represents a finite Fourier series, equivalently a finite trigonometric
polynomial written in complex-exponential form. For ordered real coordinates $x_i$,
ordered integer modes $m_j$, and ordered complex coefficients $c_j$, it evaluates

$$
f(x_i)=\sum_j c_j e^{+2\pi i x_i\cdot m_j}.
$$

“Finite Fourier series,” “integer mode,” “complex coefficient,” “positive exponent,” and
“support of the coefficient sequence” are the mathematical terms used by this
architecture. “Bloch sum” is reserved for a physics-semantic adapter that establishes
that interpretation. “Discrete Fourier transform” is not used for this operation because
no uniform dual grids or inverse-transform normalization are required.

## Unit-free coordinates

Coordinates are dimensionless real numbers and modes are exact integer tuples. The
contract does not construct reduced coordinates from a physical wavevector or lattice
basis. A caller that starts with a physical reciprocal vector must perform and validate
that conversion through its physical owner before calling this capability.

The complex coefficients are also unit-free at this boundary. A physics owner may treat
their numeric values as magnitudes in one declared unit and reattach that unit to the
result, but Numerics neither accepts nor returns physical quantity objects.

## Consumer interpretations

A PhysKit scalar-hopping adapter may interpret

- $x_i$ as reduced reciprocal coordinates;
- $m_j$ as direct-lattice displacement indices;
- $c_j$ as hopping-coefficient magnitudes; and
- $f(x_i)$ as scalar Bloch-Hamiltonian values.

That adapter remains responsible for model identity, basis identity, energy reference,
energy unit, lattice compatibility, Hermiticity diagnostics, and any relation between
opposite displacement terms.

A KSD experiment may consume the Numerics capability directly when it already owns the
unit and model mapping. KSD remains responsible for sample roles, feature construction,
weights, normalizations, candidate coordinates, thresholds, admissible sets, evidence,
and scientific interpretation.

Optimization may consume the evaluated values when constructing an objective, but the
Fourier-series evaluation itself selects no candidate and therefore remains Numerics.

## Symmetry and Hermiticity

The evaluator does not infer, impose, or diagnose symmetry. In particular, it does not:

- add missing negative modes;
- conjugate coefficients;
- coalesce duplicate modes;
- impose real-valued output;
- average symmetry-related coefficients; or
- decide whether a model is Hermitian or invariant under an operation.

These omissions are necessary for representation honesty: the direct sum must preserve
the exact coefficient inventory supplied by the caller.

## Interpretation limits

Successful evaluation establishes only the represented unit-free finite sum. It does not
establish:

- physical correctness of a lattice or hopping model;
- completeness of a finite mode inventory;
- convergence of a truncated Fourier series;
- material validity or scientific validation;
- adequacy of a candidate model;
- probabilistic uncertainty; or
- correctness of consumer-owned units, gauges, thresholds, or certificates.

## Terminology references

The mathematical terminology follows the standard treatment of finite Fourier series and
trigonometric polynomials, for example:

- A. Zygmund, *Trigonometric Series*, third edition, Cambridge University Press,
  2002; and
- E. M. Stein and R. Shakarchi, *Fourier Analysis: An Introduction*, Princeton
  University Press, 2003.

These references support the mathematical vocabulary; they do not define ProjectKoios
software contracts, floating-point behavior, or scientific interpretation.
