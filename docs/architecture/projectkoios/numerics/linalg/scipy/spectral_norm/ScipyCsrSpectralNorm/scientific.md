# `ScipyCsrSpectralNorm` scientific boundary

`ScipyCsrSpectralNorm` performs unit-free binary64 software evaluation. Direct small
shapes produce a rounded direct result. Larger shapes produce a deterministic iterative
approximation that has passed owner structural and residual checks.

The class does not establish:

- physical units or dimensional consistency;
- operator identity, basis meaning, or matrix provenance;
- a scientific convergence or acceptance threshold;
- uncertainty in the returned approximation;
- fitness for an extreme-range correlated residual diagnostic;
- equivalence to an independently computed dense result; or
- authorization to replace a consumer's complete-SVD path.

A unit-aware consumer restores the matrix-entry unit to the scalar. A consumer selecting
an iterative sparse route must expose that choice and the exact controls rather than
silently replacing complete dense semantics.
