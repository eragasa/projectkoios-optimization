# `ComplexAffineResidualQuadraticConstructor`

## Purpose

`projectkoios.optimization.affine.complex_quadratic.ComplexAffineResidualQuadraticConstructor`
is the nominal SDK boundary that converts an ordered dense complex observation system
with **real parameter coordinates** into a unit-free completed-square quadratic.

For complex coefficient rows $A_i\in\mathbb{C}^p$, complex targets
$b_i\in\mathbb{C}$, positive optional objective multipliers $w_i$, and
$x\in\mathbb{R}^p$, the represented objective is

$$
L(x)=\sum_i w_i\left|A_i x-b_i\right|^2.
$$

Weights are objective multipliers, not pre-square-rooted values. `None` means unit
weights. Observation order and parameter-coordinate order are preserved.

## Ownership boundary

The capability owns:

- validation of finite rectangular complex coefficient rows and targets;
- validation of one finite strictly positive weight per observation;
- dense real-parameter numerical rank acceptance;
- direct weighted least-squares solution without normal-equation solving;
- the exactly symmetric real normal matrix
  $Q=\operatorname{Re}(A^\dagger W A)$;
- direct residual evaluation of the minimum at the fitted center; and
- construction and relational validation of `PositiveDefiniteQuadraticLoss`.

Callers own tensor flattening, units, scientific normalization, observation and
coordinate meaning, labels, thresholds, admissibility, certificates, and disposition.
The API has no complex parameter-coordinate mode and assigns no tensor semantics.

## Contract and backend

`ComplexAffineResidualQuadraticTolerance` has one strict canonical constructor. It
accepts exact finite nonnegative built-in floats and validates without postconstruction
field replacement.

`ComplexAffineResidualQuadraticConstructor.construct()` is a final public template
method. It normalizes dependency-free tuple inputs, calls one nominal backend subclass,
and validates the returned dense quadratic against independently reconstructed normal,
stationarity, and direct-residual relationships.

`NumpyComplexAffineResidualQuadraticConstructor` is the initial implementation. It
stacks real and imaginary observation equations internally, applies square-root row
weights, and calls `numpy.linalg.lstsq(..., rcond=None)`. It requires NumPy-reported
full real-coordinate column rank and never regularizes a rejected system. It then forms
an exactly mirrored real normal matrix and evaluates the minimum from the original
complex residual at the fitted center; it does not recover the minimum by subtracting
large completed-square constants and does not clamp a negative cancellation artifact.

`ComplexAffineResidualQuadraticConstructorFactory` lazily constructs the closed
`NUMPY_DENSE` built-in. External implementations subclass the nominal constructor and
are injected directly.

## Verification boundary

Tests cover complex observations with real coordinates, positive objective weights,
rank deficiency, cancellation-sensitive nonzero minima, exact symmetry, backend
failure translation, malicious backend results, factory laziness, and independent
stacked-real references. These are software and bounded numerical verification, not
scientific validation or conditioning guarantees.
