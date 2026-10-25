# Matrix-norm contract implementation

## Source

```text
src/python/projectkoios/numerics/linalg/
└── matrix_norm.py
```

The module imports only the standard library. `MatrixNormABC` supplies the final result
validation template, while concrete subclasses must implement the read-only
`matrix_norm` property and `_evaluate`. No factory or backend registration exists in this
foundation.

Optional matrix representation aliases are owned separately:

- `projectkoios.numerics.numpy.typing.DenseMatrix` covers exact float64 or complex128
  NumPy matrix annotations; and
- `projectkoios.numerics.scipy.typing.CsrMatrix` covers SciPy `csr_matrix` and
  `csr_array` representations.

Those aliases do not perform runtime validation and importing them explicitly imports
their optional backend. `NumpyFrobeniusNorm` and `ScipyCsrFrobeniusNorm` are the current
concrete implementations of the nominal contract; additional semantics and backends
require separate review.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
└── test__MatrixNormABC__unit.py
```

Unit tests cover enum form, abstractness, successful dispatch, nonfiniteness, negativity,
input handoff, and exception propagation. Repository build verification checks
distribution inclusion and dependency-free import behavior.
