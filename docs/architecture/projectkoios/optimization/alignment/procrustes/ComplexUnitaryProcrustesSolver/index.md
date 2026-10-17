# ComplexUnitaryProcrustesSolver

`ComplexUnitaryProcrustesSolver[DefinitionT, ResultT]` is the generic behaviorful nominal
root for scalar and batched right-Procrustes solver families.

It owns exact tolerance validation, additive rank-cutoff policy, the binary64 residual
roundoff coefficient, and executed-versus-absent backend evidence validation. Its typed
`solve` method remains abstract because family definitions and results differ.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
