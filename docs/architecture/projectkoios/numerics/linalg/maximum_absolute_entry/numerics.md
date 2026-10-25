# Maximum-absolute-entry numerics

The dense NumPy implementation validates an exact nonempty two-dimensional ndarray with
exact float64 or complex128 storage and finite entries. It evaluates entry magnitudes and
takes their maximum without mutating or normalizing the caller array.

The public result is converted to an exact built-in float. A finite complex entry can
have a nonrepresentable binary64 magnitude; nonfinite magnitude evaluation therefore
raises `FloatingPointError` rather than returning infinity. Zero matrices return exact
`0.0`.

This is deterministic binary64 software verification. It is not a threshold decision,
scientific acceptance claim, or unit policy.
