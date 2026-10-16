# Sparse partial-spectrum implementation

The implementation separates dependency-free CSR records and shared qualification from
SciPy sparse conversion and ARPACK execution.

- Package initializers remain import-empty.
- The core sparse module imports no optional numerical package.
- The SciPy module owns sparse-array conversion, seeded start vectors, ARPACK, and
  residual calculation.

## Source

Two source files implement the nominal capability and its first concrete backend.

- `src/python/projectkoios/numerics/linalg/sparse_hermitian_spectrum.py` owns canonical
  CSR validation, selection records, represented-Hermitian construction, result records,
  and the nominal solver template.
- `src/python/projectkoios/numerics/linalg/scipy/sparse_hermitian_spectrum.py` owns the
  SciPy adapter and `scipy.sparse.linalg.eigsh` backend.
- `src/python/projectkoios/numerics/linalg/scipy/__init__.py` remains import-empty.

## Tests

Mirrored tests distinguish dependency-free record behavior from SciPy integration and
bounded numerical verification.

- Core unit tests cover canonical CSR invariants and explicit iterative controls.
- Core contract tests cover sparse Hermiticity cutoff equality, whole-problem rejection,
  explicit representative construction, and residual correlation.
- SciPy integration tests cover adapter canonicalization, four selection rules, complex
  matrices, count constraints, metadata, and residuals.
- Numerical tests cover seeded sparse cases against dense references and repeatability for
  a fixed seed.
- Namespace tests verify that importing the SciPy package does not eagerly import NumPy
  or SciPy.

## Documentation conventions

Production documentation uses declarative NumPy-style docstrings and focused comments for
computational scientists.

- Docstrings state sparse representation, controls, results, and failure classes.
- Inline comments explain sparse-complexity preservation, represented-input evidence,
  deterministic start-vector construction, internal eigenvector use, and backend failure
  translation.
- Schematics use Mermaid exclusively.
