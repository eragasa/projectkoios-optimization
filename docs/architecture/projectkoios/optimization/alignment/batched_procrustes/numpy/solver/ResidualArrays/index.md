# ResidualArrays

`ResidualArrays` is an encapsulated frozen solver record for vectorized power and
optimal-residual evidence.

## Ownership

- It groups source and target squared Frobenius norms.
- It groups reconstructed optimal squared residuals, their square roots, and applied
  squared-unit roundoff cutoffs.
- It performs no reconstruction or validation itself.
- It is consumed only while assembling the validated public result and is not exported
  by the façade.

## Verification

Dedicated unit tests verify exact field identity, frozen field binding, slots, and
absence from the supported façade. Residual reconstruction, solver integration, and
public result tests verify the numerical relations represented by its fields.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
