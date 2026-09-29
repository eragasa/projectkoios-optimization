# Contracts

`OptimizationEngine` is a behaviorful structural protocol: `ask` proposes a
candidate, `tell` consumes finite objective feedback exactly once, and
`checkpoint` returns versioned serializable state. It is not a universal marker
or execution superclass.

`ForwardEvaluator` receives a candidate and returns raw observations. It does
not update an optimizer. `ResultsHandler` is bound to the exact proposing engine
instance, then receives each candidate and raw evaluation; it retains the raw
pair, derives declared feedback, and calls `tell` on that bound object.
`WeightedResultsHandler` is the dependency-free in-memory implementation and
owns no persistence.

`ProbabilityDistribution` is a behaviorful abstract composition of a cumulative
distribution function and inverse cumulative distribution function. A
`ProbabilityDistributionSampler` consumes a distribution explicitly; the
inverse-transform implementation owns the uniform-fraction source rather than
hiding global random state. Optional NumPy and SciPy implementations remain in
the backend-oriented adapter namespaces.

Candidates carry run identity, stable candidate identity, proposal sequence,
and finite coordinates. Text identifiers and measurement/objective names are
nonempty, trimmed, and at most 256 characters; weighted-loss measurement names
use the identical measurement-name contract. Optimizer families additionally
use the portable lowercase family grammar and a 128-character maximum.

Real-valued public fields reject booleans and non-real values, normalize to a
built-in binary64 `float`, and reject conversion overflow or nonfinite results.
Parameter bounds also require the representable width `upper - lower` to remain
finite. Optimizers enforce parameter-space dimension and bounds. Feedback
carries finite nonnegative scalar loss and ordered finite nonnegative
objectives.
