# Implementation

## Public facade

`quadratic/quadratic.py` defines `PositiveDefiniteQuadraticLoss`. The class validates the
exact built-in Boolean backend flag, lazily imports one concrete backend, and delegates
its read-only properties and evaluation methods. It is not imported by
`projectkoios.optimization`.

The facade accepts:

- a nonempty tuple center;
- a dense tuple matrix, or a SciPy sparse matrix when `sparse=True`;
- a finite nonnegative `minimum_squared_loss`; and
- the keyword-only `sparse` backend flag.

Shared private tuple validation lives in `quadratic/_validation.py` and reuses the target's
`_validation.finite_float` contract. Booleans and non-real values fail by kind; values
that cannot become finite binary64 fail by value.

## Dense implementation

`quadratic/numpy/quadratic.py` defines
`NumpyPositiveDefiniteQuadraticLoss`. The public facade owns dense input normalization;
the backend accepts and retains exact canonical tuple storage, requires exact symmetry,
and calls `numpy.linalg.cholesky` for dense positive-definiteness acceptance. Evaluation
materializes transient NumPy arrays without installing private caches or replacing frozen
fields.

## Sparse implementation

`quadratic/scipy/quadratic.py` defines
`ScipySparsePositiveDefiniteQuadraticLoss`. Tuple and sparse sources are copied into
canonical CSR storage. Duplicate entries are summed, explicit zeros removed, and indices
sorted before exact represented-symmetry comparison. The public `matrix` property
returns a defensive copy.

For dimensions above one, the implementation requests the smallest algebraic
eigenvalue from `scipy.sparse.linalg.eigsh`. Nonconvergence and nonpositive results fail
construction. The one-dimensional case checks its sole diagonal directly.

## Ownership boundary

Optimization owns these reusable unit-free objects and their backend behavior. PhysKit
owns unit-bearing wrappers and dimensional conversion. Scientific packages own
normalization, parameter meaning, thresholds, and acceptance policy, and must adapt to
this API rather than redefining the generic loss.
