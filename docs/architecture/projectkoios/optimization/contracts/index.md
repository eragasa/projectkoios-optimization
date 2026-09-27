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

Candidates carry run identity, stable candidate identity, proposal sequence,
and finite coordinates. Optimizers enforce parameter-space dimension and
bounds. Feedback carries finite nonnegative scalar loss and ordered finite
nonnegative objectives.
