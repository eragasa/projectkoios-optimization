# Numerical validation

`projectkoios.numerics.validation` owns dependency-free validation and normalization of built-in numerical representations.

Shared immutable representations such as `FiniteComplexMatrix` are owned by
[`projectkoios.numerics.typing`](../typing/index.md); this module validates their runtime
invariants. It does not assign optimization, physical, or scientific meaning.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
