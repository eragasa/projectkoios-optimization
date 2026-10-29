# NumPy implementation

NumPy imports remain inside this namespace and concrete NumPy backend packages.
Dependency-free numerical contracts do not import NumPy.

`projectkoios.numerics.numpy.validation` owns reusable construction and validation of
canonical immutable arrays with one required dtype. Capability-local backend operations
own representation checks that depend on an operation-specific dtype union, matrix
shape, nonempty axes, finiteness, layout acceptance, and numerical algorithm.

Dense linear-algebra backend modules live under
`projectkoios.numerics.linalg.numpy`. In particular,
`projectkoios.numerics.linalg.numpy.spectral_norm` accepts direct float64 or complex128
ndarrays, validates its complete matrix contract, and passes the caller array directly
to `numpy.linalg.svdvals`.
