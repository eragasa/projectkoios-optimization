# ResultArrays

`ResultArrays` is an encapsulated frozen construction record used by the public result
while normalizing and validating its ten arrays.

## Ownership

- It groups completions, spectra, ranks, rank status, cutoffs, powers, and residual
  evidence under named fields.
- It owns no validation, copying, mutation, or public behavior.
- Its lifetime is confined to result construction.
- It is defined in `contract.py` and is not exported by `base.py` or the package façade.

The arrays may be intermediate values; only
`NumpyBatchedComplexUnitaryProcrustesResult` establishes immutable public backing after
all relations pass.

## Verification

Dedicated unit tests verify exact field identity, frozen field binding, slots, and
absence from the supported façade. Result tests separately verify every relation for
which this record transports evidence.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
