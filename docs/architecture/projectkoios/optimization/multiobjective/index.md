# Multi-objective optimization

`MultiObjectiveOptimizer` asks for candidate decision vectors and accepts ordered finite
objective vectors. Scientific QoIs, names, units, direction conversion, and evaluator
execution remain outside this numerical boundary.

Its archive pairs each nondominated candidate with its objective vector:

- `pareto_set` is the decision-vector approximation;
- `pareto_front` is the corresponding objective-vector approximation.

`pareto/ParetoOptimizer` implements the base using explicit sampling stages, and
`MultiObjectiveOptimizerFactory.create_pareto(...)` constructs it without string
dispatch. Each stage may use a different parameter sampler and receives the preceding
Pareto set and front together with the complete ordered decision/objective observation
history. The additional history permits adaptive samplers to use dominated observations
without moving evaluation or workflow ownership into Optimization.

The canonical capability record is [`pareto/index.md`](pareto/index.md). It keeps the
protected selector and optimizer, PyPosPack/PyMatMC2 historical mechanism authority,
KSD's recovery-only status, exact donor identities, corrections, performance guards,
and owner-authored shell extensions together under the Pareto namespace.
