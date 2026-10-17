# ComplexUnitaryProcrustesSolver

`ComplexUnitaryProcrustesSolver` is the concrete behaviorful shared-policy root for
scalar and batched right-Procrustes solver families.

It owns exact tolerance validation, additive rank-cutoff policy, the binary64 residual
roundoff coefficient, and executed-versus-absent backend evidence validation. It does
not declare a generic `solve` return. Scalar and batched nominal bases own their concrete
definition and result contracts.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
