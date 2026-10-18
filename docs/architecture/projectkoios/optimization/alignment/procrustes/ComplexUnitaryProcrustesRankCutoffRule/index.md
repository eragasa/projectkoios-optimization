# ComplexUnitaryProcrustesRankCutoffRule

`ComplexUnitaryProcrustesRankCutoffRule` is the dependency-free closed selection for
combining absolute and scaled-relative numerical-rank controls.

## Values

The enum changes only software numerical-rank evidence and preserves strict comparison
against the applied cutoff.

- `ADDITIVE` applies `rank_absolute + rank_relative * largest_singular_value`.
- `MAXIMUM` applies `max(rank_absolute, rank_relative * largest_singular_value)`.
- `ADDITIVE` remains the default in `ComplexUnitaryProcrustesTolerance`.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
- [Procrustes capability](../index.md)
