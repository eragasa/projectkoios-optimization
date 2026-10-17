# Capability-local solver template

A solver capability exposes a nominal abstract class as its stable SDK integration point.
Concrete owner and third-party implementations opt in explicitly through inheritance.
Structural protocols are not used.

## Public vocabulary

For a mathematical problem named `<Problem>`:

- `<Problem>Result`: immutable backend-neutral result;
- `<Problem>Solver`: capability-local abstract class;
- `<Problem>Method`: closed enum for built-in implementations;
- `<Problem>SolverFactory`: lazy built-in constructor; and
- `<Backend><Problem>Solver`: concrete implementation.

Example:

```text
LinearLeastSquaresResult
LinearLeastSquaresSolver
LinearLeastSquaresMethod
LinearLeastSquaresSolverFactory
NumpyLinearLeastSquaresSolver
```

## SDK integration

Higher-level APIs accept a `LinearLeastSquaresSolver` instance. A consumer or integration
package can provide another implementation only by explicitly subclassing the nominal
abstract class and satisfying its conformance tests.

The owner factory constructs supported built-in implementations. Third-party packages
provide their own construction or inject a subclass instance directly; they do not
mutate the owner factory through an unbounded string registry.

## Fitting specialization

A fitting solver result additionally retains selected parameters, fitted values,
residuals, objective value, and applicable rank or conditioning diagnostics. Fitting is
an optimization problem and remains under `projectkoios.optimization`.

## Pages

- [Schematics](schematics.md)
- [Implementation template](implementation.md)
- [Architecture templates](../index.md)
