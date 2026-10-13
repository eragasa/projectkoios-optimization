# Multi-objective optimization

`MultiObjectiveOptimizer` asks for candidate decision vectors and accepts ordered finite
objective vectors. Scientific QoIs, names, units, direction conversion, and evaluator
execution remain outside this numerical boundary.

Its archive pairs each nondominated candidate with its objective vector:

- `pareto_set` is the decision-vector approximation;
- `pareto_front` is the corresponding objective-vector approximation.

`pareto/ParetoOptimizer` implements the base using explicit sampling stages, and
`MultiObjectiveOptimizerFactory.create_pareto(...)` constructs it without string
dispatch. Each stage may use a different parameter sampler and receives the preceding Pareto set
and front together with the complete ordered decision/objective observation history.
The additional history permits adaptive samplers to use dominated observations without
moving evaluation or workflow ownership into Optimization.

`pareto/shells.py` recursively peels zero-based nondominated shells and pulls each shell
back to exact decision vectors. It groups equal objective vectors before invoking the
historical kernel so duplicates receive the same shell rank. This is an owner-authored
research extension, not reconstructed PyPosPack behavior. Its design record is
[`pareto-shell-contrastive-sampling.md`](../../../../research/pareto-shell-contrastive-sampling.md).

The preserved PyPosPack selection kernel lives separately in `pareto/algorithm.py`.
It must not be modified without explicit human authorization and both correctness and
speed evidence. `test__ParetoAlgorithm__numerical.py` compares it directly with the
minimally corrected donor implementation and with full brute force on 20,000 correlated
ten-objective candidates.
