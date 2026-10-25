# Matrix-norm contract numerics

The nominal contract separates mathematical identity, matrix representation, and
implementation arithmetic.

- `matrix_norm` identifies the selected semantic member of `MatrixNormEnum`.
- `MatrixT` identifies the concrete input representation accepted by a subclass.
- `_evaluate(matrix)` performs representation checks and numerical evaluation in the
  subclass.
- The final public call accepts only an exact built-in `float` result that is finite and
  nonnegative.

A wrong result type raises `TypeError`. A nonfinite or negative implementation result
raises `FloatingPointError`. The boundary does not coerce NumPy scalar values, infer
units, apply normalization, or decide scientific acceptance.

The enum uses opaque `auto()` values. Member identity and member names carry the Python
semantic contract; enum values are not stable external labels. Presence of a semantic
enum member does not establish an implemented numerical method.
