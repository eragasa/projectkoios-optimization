# Finite Fourier-series numerical contract

## Status

This page specifies the current numerical contract implemented by the dependency-free
records, nominal evaluator, and ordered NumPy backend.

## Definition

For positive coordinate dimension $d$, ordered points
$x_i\in\mathbb{R}^d$, ordered modes $m_j\in\mathbb{Z}^d$, and ordered
coefficients $c_j\in\mathbb{C}$, the result is

$$
y_i=\sum_{j=0}^{T-1}c_j\exp\!\left(+2\pi i
\sum_{k=0}^{d-1}x_{ik}m_{jk}\right).
$$

The operation performs no normalization. Point order determines result order. Coordinate
components, modes, and coefficients retain their supplied order. Duplicate modes remain
distinct terms and are not coalesced.

## Dependency-free input contract

`FiniteFourierSeriesEvaluationDefinition` is an immutable frozen record containing:

- `dimension: int`;
- `coordinates: tuple[tuple[float, ...], ...]`;
- `integer_modes: tuple[tuple[int, ...], ...]`; and
- `coefficients: tuple[complex, ...]`.

The record requires exact built-in scalar and tuple types so the public contract imports
no numerical backend.

Validation requirements are:

1. `dimension` is an exact built-in integer greater than zero; booleans are rejected.
2. Every coordinate row is an exact tuple of length `dimension` containing exact
   built-in finite floats.
3. Every mode row is an exact tuple of length `dimension` containing exact built-in
   integers; booleans are rejected.
4. Each mode component lies in the signed 64-bit range and has magnitude no greater
   than $2^{53}$, so conversion to binary64 preserves the integer value exactly.
5. `coefficients` is an exact tuple of exact built-in complex values with finite real
   and imaginary components.
6. The number of coefficients equals the number of mode rows.
7. Empty coordinate and mode collections are permitted because `dimension` is retained
   explicitly.

The $2^{53}$ limit is an integer-to-binary64 representation guarantee, not a general
accuracy guarantee for arbitrarily oscillatory phases. The initial contract makes no
accuracy claim outside its independently verified bounded fixtures.

## Result contract

`FiniteFourierSeriesEvaluationResult` is an immutable frozen record containing:

- `dimension: int`;
- `point_count: int`;
- `term_count: int`; and
- `values: tuple[complex, ...]`.

It requires positive `dimension`, nonnegative exact built-in counts, and exactly
`point_count` finite built-in complex values. The evaluator template method enforces
structural correlation of dimension, counts, output length, and finiteness with the input
definition. Numerical correspondence to coordinates, modes, and coefficients is an
implementation obligation tested by the conformance and numerical suites; the nominal
boundary cannot prove it from result metadata alone.

Boundary cases are exact:

- zero points produce an empty value tuple;
- zero terms produce one exact `0.0 + 0.0j` value per point; and
- zero coordinate dimension is invalid even when both collections are empty.

## Initial binary64/complex128 backend

`NumpyFiniteFourierSeriesEvaluator` uses float64 coordinates, exactly converted bounded
integer modes, complex128 coefficients, and complex128 outputs.

Its required evaluation order is:

1. process points in supplied order;
2. for each point, process terms in supplied order;
3. form each coordinate-mode inner product by accumulating coordinate components in
   supplied coordinate order using binary64 arithmetic;
4. form the positive phase $2\pi(x_i\cdot m_j)$ in binary64;
5. evaluate the complex exponential in complex128; and
6. accumulate coefficient-times-phase terms in supplied term order using a scalar
   complex128 accumulator.

The backend must not sort modes, combine duplicates, delegate the reduction to an
unspecified parallel BLAS reduction, or silently change summation order. The algorithm
and ordering are deterministic within a pinned Python/NumPy/platform environment. No
bitwise identity is promised across NumPy versions, libm implementations, hardware, or
external backends.

The implementation must execute with NumPy floating-point invalid and overflow conditions
configured to raise. Any nonfinite intermediate or output is translated to
`FloatingPointError`. It must not return `NaN`, infinity, a partial result, or a silently
wrapped phase.

## Error semantics

The dependency-free definition and result records raise:

- `TypeError` for wrong semantic container or exact scalar types; and
- `ValueError` for invalid dimension, row width, count correlation, nonfinite values,
  or out-of-range integer modes.

The evaluator raises:

- `TypeError` when the definition or backend result has the wrong nominal type;
- `ValueError` when a backend result violates shape or structural-correlation
  invariants; and
- `FloatingPointError` for nonfinite backend arithmetic or output.

The initial NumPy backend translates no other exception type. All programming exceptions,
`MemoryError`, `KeyboardInterrupt`, `SystemExit`, and other undeclared backend exceptions
propagate unchanged. A later backend that requires another stable failure contract must
receive separate architecture review; “safe to translate” is not an open exception
category.

## Numerical verification requirements

Independent and adversarial tests must cover:

1. constant, single-mode, and multiple-mode analytic values;
2. positive exponent orientation against values where the negative convention differs;
3. negative modes and complex coefficients;
4. conjugate-paired terms producing independently known real values;
5. duplicate modes retained as separate ordered summands;
6. cancellation-sensitive sums and declared term-order behavior;
7. one-, two-, and three-dimensional bounded fixtures;
8. periodicity under integer coordinate translations on bounded modes;
9. empty points, empty terms, and rejection of zero dimension;
10. exact rejection of non-built-in scalars, nonfinite values, mismatched shapes, and
    integer modes outside the representation limit;
11. backend nonfinite arithmetic and unchanged propagation of undeclared backend failures;
12. malicious nominal subclasses returning wrong types, counts, or nonfinite values;
13. agreement with an independently written high-precision scalar
    complex-exponential reference on bounded sanitized cases; and
14. consumer-shaped regression fixtures without importing PhysKit or KSD.

The high-precision reference must evaluate with at least 80 decimal digits before one
final conversion to binary64 complex components. Its bounded fixture family is:

- dimension from one through three;
- at most 32 terms and 64 points;
- coordinate magnitudes no greater than one;
- mode-component magnitudes no greater than 16; and
- coefficient magnitudes no greater than four.

For binary64 machine epsilon $\epsilon$, each backend value $y_i$ and high-precision
reference value $r_i$ must satisfy the absolute criterion

$$
|y_i-r_i|\leq 512\epsilon\max\!\left(1,\sum_j|c_j|\right).
$$

Analytic exact-angle fixtures use the same absolute criterion with factor 128. Periodicity
fixtures use factor 1024 because both sides are independently evaluated. Cancellation
fixtures use the absolute coefficient-sum scale above rather than an unstable relative
error near zero. These constants are bounded implementation-acceptance criteria, not a
forward-error theorem or a public accuracy guarantee for every accepted input.

These tests establish software behavior and bounded numerical agreement only. They do
not establish convergence of a Fourier expansion or scientific validity of a consumer
model.
