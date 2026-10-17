# Centered uniform transform implementation

## Source organization

```text
projectkoios/numerics/finite_periodic/
├── meshes.py                 # CenteredUniformMesh
├── transforms.py             # dependency-free diagnostic records
├── batched_finite_series.py  # backend-neutral positive-exponent contract
└── numpy/
    ├── batched_finite_series.py  # shared bounded channel evaluator
    ├── matrix_series.py          # immutable mode/coefficient record
    ├── matrix_evaluation.py      # bounded arbitrary-mode evaluation
    ├── tensor_product_evaluation.py
    │                              # bounded separable grid evaluation
    ├── transforms.py             # complete centered forward transform
    └── transform_diagnostics.py  # Parseval and Hermiticity analyzers
```

The package initializers remain import-empty. Importing `meshes` or the backend-neutral
`transforms` module does not import NumPy. NumPy arrays appear only in modules below
`finite_periodic.numpy`.

## Canonical arrays

Array-bearing records use exact NumPy dtypes:

- reduced coordinates: `float64`;
- integer modes: `int64`, bounded in magnitude by `2**53` for exact float64 phase
  conversion; and
- samples, coefficients, and evaluated matrices: `complex128`.

Array aliases come from `projectkoios.numerics.numpy.typing`; immutable storage and
finiteness checks reuse `projectkoios.numerics.numpy.validation`; exact scalar checks
reuse `projectkoios.numerics.validation`.

Raw frozen constructors accept only C-contiguous, read-only arrays whose ultimate
backing owner is immutable bytes. `from_array(...)` or `from_arrays(...)` performs an
explicit defensive copy. `from_immutable_array(...)` or
`from_immutable_arrays(...)` is the named zero-copy boundary. Constructors never repair
or replace frozen fields after construction.

## Forward transform

`NumpyCenteredUniformMeshMatrixFourierTransformer.transform(...)`:

1. validates the exact nominal definition;
2. calls `numpy.fft.fftn(..., norm="forward")` on the first `d` tensor axes;
3. centers each transformed axis with `numpy.fft.fftshift`;
4. enumerates integer modes lexicographically over
   `[-N/2, ..., N/2 - 1]^d`;
5. applies the negative-exponent phase for the exact first mesh coordinate; and
6. returns immutable flattened `M × n × n` coefficients with backend evidence.

The transform supports `d = 1, 2, 3`, one common positive even `N`, arbitrary finite
per-axis offsets, and any positive square matrix size.

## Bounded inverse evaluation

`NumpyFiniteFourierMatrixSeriesEvaluator` treats the `n²` matrix entries as coefficient
channels. It reuses the internal channel operation that implements the existing
`BatchedFiniteFourierSeriesEvaluator` convention. Points are flattened in C order, then
processed in batches satisfying

```text
point_batch_count * mode_batch_count <= maximum_phase_elements
```

for every temporary point-by-mode phase matrix. The output is reshaped to the supplied
coordinate tensor followed by the two matrix axes. Selected-mode series may be empty;
in that case evaluation returns exact zero matrices without constructing phases.

The existing tuple-valued batched evaluator now uses the same bounded operation with a
fixed one-million-element phase bound. Small problems preserve one shared phase matrix
and one dense reduction.

## Separable tensor-grid evaluation

`NumpyTensorProductFourierMatrixSeriesEvaluator` embeds the declared selected inventory
into the centered construction grid. Duplicate modes are accumulated in declared order.
It then contracts one coordinate axis at a time with `numpy.einsum`, batching the first
evaluation axis. The definition exposes the coefficient-grid, nonleading-phase,
per-leading-point, batch-size, and maximum-temporary element counts. Every retained
working array obeys `maximum_working_elements`; the final output is intentionally
excluded because it is the requested result rather than temporary storage.

This path supports complete inventories and selected Cartesian-product inventories. It
avoids the generic `O(P*M)` point-mode phase matrix for large complete evaluation grids,
including the retained `N=32` construction and `65**3` evaluation case.

## Sparse boundary

No separate sparse FFT backend is implemented. A complete uniform transform must return
every coefficient for reconstruction and Parseval evidence, while the small matrix
blocks are ordinarily dense. Sparse matrix storage would therefore add conversion cost
without changing the dense complete result.

`NumpyFiniteFourierMatrixSeries` already represents a selected sparse mode inventory as
explicit integer rows plus corresponding dense blocks; its evaluator never allocates a
full coefficient grid and bounds both point and mode batches. Nonuniform transforms,
approximate top-k discovery, and sparse storage within large matrix blocks require
separate demonstrated contracts and dependency review.

## Diagnostics

`NumpyFinitePeriodicParsevalAnalyzer` accepts a complete definition/result pair and
reports the sample-mean and coefficient-sum squared Frobenius energies.

`NumpyCenteredUniformMeshHermiticityAnalyzer` reshapes a complete result to its tensor
inventory. It pairs every mode with its centered cyclic negative, derives signed wrap
counts per axis, and applies the exact phase implied by the retained coordinate origin.
The ordinary diagnostic excludes every lower Nyquist boundary hyperplane.

Neither analyzer applies a tolerance or returns an acceptance Boolean.

## Verification

- Unit tests cover exact mesh and diagnostic-record invariants.
- Contract tests cover exact dtypes, immutable ownership, tensor shape, square matrices,
  coordinate width, and the temporary phase bound.
- Integration tests recover an independently constructed single mode and verify complete
  round trips in one, two, and three dimensions.
- Numerical tests compare reduced-coordinate phases with oblique-2D and FCC-primitive
  physical dual-lattice phases.
- Hermitian random matrix fields verify cyclic Nyquist phases and ordinary pairing.
- Instrumented evaluation verifies every temporary phase matrix respects the declared
  element bound.
- An isolated-wheel test verifies dependency-free imports followed by NumPy execution.
