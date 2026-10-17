# Shared-grid batched finite Fourier implementation

The implementation separates dependency-free records and template behavior from the
capability-local NumPy backend.

- Core contracts import no array backend.
- Frozen records validate canonical immutable fields without replacing them after
  construction.
- NumPy is imported only from the concrete backend module.
- Package initializers remain import-empty.
- No method enum or one-option factory is introduced.

## Source organization

The two source modules mirror the contract/backend ownership split.

- `finite_periodic/batched_finite_series.py` owns canonical records, exact empty cases,
  definition/result correlation, required backend evidence, and the nominal evaluator.
- `finite_periodic/numpy/batched_finite_series.py` owns binary64/complex128 dense evaluation.
- `finite_periodic/finite_series.py` remains the unchanged scalar ordered capability.
- `finite_periodic/numpy/finite_series.py` remains the unchanged scalar NumPy backend.

## Empty behavior

The template resolves mathematically exact empty cases before optional backend code is
invoked.

- `P = 0` returns zero point rows while retaining `channel_count = S`.
- `S = 0` returns `P` empty rows.
- `M = 0` returns a `P × S` matrix of exact complex zeros.
- Backend identifier and version are absent for every template-resolved case.

## NumPy execution

The nontrivial backend shares phase work across channels and bounds large evaluations.

- Coordinates become a `P × d` binary64 array.
- Modes become an `M × d` binary64 array.
- Point and mode batches contain at most one million phase elements per temporary.
- Inner products accumulate one coordinate component at a time into `B × M_b`.
- One complex128 exponential and one `numpy.matmul` are used per point/mode batch pair.
- Small problems retain a single shared phase matrix and reduction.
- Public output is converted to immutable built-in complex tuples with shape `P × S`.

## Verification organization

Tests mirror source ownership and distinguish contract, integration, and bounded
numerical claims.

- Core unit tests cover canonical shapes, scalar types, mode bounds, empty cases, and
  evidence through public records.
- Contract tests exercise public template behavior and preserve undeclared exceptions.
- NumPy integration tests verify shared phase construction and dense multiplication for
  representative small problems; the matrix-series integration tests verify the shared
  bounded backend across multiple batches.
- Numerical tests compare representative columns with the scalar backend under an explicit scale-aware bound.
- The isolated-wheel test verifies dependency-free import before NumPy backend import.
