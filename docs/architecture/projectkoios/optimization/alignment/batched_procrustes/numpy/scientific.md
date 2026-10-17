# NumPy batched Procrustes consumer needs

A consumer chooses the NumPy backend when its prepared mathematical inputs are exact
complex128 arrays and vectorized dense SVD is appropriate.

- Consumers own the meaning of leading indices and matrix columns.
- Consumers choose scientific thresholds before constructing the owner tolerance.
- Consumers may persist rank, residual, and roundoff evidence, but scientific conclusions
  require downstream policy and validation.
- Any deficient item prevents interpreting the associated diagnostic completion as a
  unique transformation.
- KSD closure and plaquette policy remains in KSD; PhysKit frame and unit meaning remains
  in PhysKit.

The NumPy modules receive no lattice, unit, topology, disposition, persistence, or human
acceptance objects.
