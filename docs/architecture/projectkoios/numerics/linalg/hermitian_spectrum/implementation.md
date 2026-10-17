# Batched Hermitian-spectrum implementation

The implementation keeps dependency-free contracts separate from the optional NumPy
backend and mirrors that boundary in its tests.

- Package initializers remain import-empty.
- The contract module imports no optional numerical package.
- The NumPy and SciPy modules own only concrete array conversion and eigensolver
  execution.

## Source

Three capability-local source files divide shared mathematical behavior from backend
behavior.

- `src/python/projectkoios/numerics/linalg/hermitian_spectrum.py` owns canonical storage
  validation, immutable records, inferred nonempty and explicit empty shapes, Hermiticity
  diagnostics, representative construction, aggregate correlation, and the nominal
  solver template.
- `src/python/projectkoios/numerics/linalg/numpy/hermitian_spectrum.py` owns complex128
  array conversion and one nonempty batched `numpy.linalg.eigvalsh` call.
- `src/python/projectkoios/numerics/linalg/scipy/hermitian_spectrum.py` owns complex128
  array conversion and one nonempty batched `scipy.linalg.eigvalsh` call.
- Consumers import either concrete backend directly because backend choice follows from
  the imported implementation rather than a global factory.
- Definition, tolerance, per-matrix diagnostic, aggregate diagnostic, and result records
  have sole strict canonical constructors. Exact built-in storage is validated explicitly without
  reflective attribute access, coercion, or postconstruction field replacement.

## Tests

Source-mirrored tests separate dependency-free contracts, concrete integration, bounded
numerical verification, and isolated-wheel behavior.

- `tests/projectkoios/numerics/linalg/hermitian_spectrum/` covers definitions, results,
  the shared solver contract, and wheel imports.
- `tests/projectkoios/numerics/linalg/numpy/hermitian_spectrum/` covers NumPy integration
  and bounded numerical verification.
- `tests/projectkoios/numerics/linalg/scipy/hermitian_spectrum/` covers SciPy integration
  and bounded numerical verification.
- Contract cases cover canonical storage, explicit dimensions, empty construction,
  empty short-circuiting, cutoff equality, whole-batch failure,
  representative evidence, absent empty aggregates, and malformed backend output.
- Backend cases cover NumPy and SciPy one-call batching, nonconvergence translation, and
  nonfinite output.
- Numerical cases cover high-precision reference spectra, batch order, uniform shifts,
  and tolerance-qualified representative construction.
- Wheel cases cover dependency-free contract import and installed NumPy execution.

## Documentation conventions

Production documentation uses concise declarative language for computational scientists,
while inline comments expose the rationale behind non-obvious numerical choices.

- Public and private production callables use NumPy-style docstrings.
- Docstrings state parameters, returns, raised failure classes, and mathematical notes.
- Inline comments explain strict scalar storage, checked difference arithmetic,
  multiplicity-preserving order, empty aggregates, cutoff
  reconstruction, per-matrix scaling, cutoff equality, exact represented Hermiticity,
  empty short-circuiting, NumPy stack dispatch, and failure translation.
- Comments explain rationale rather than narrating self-evident syntax.
