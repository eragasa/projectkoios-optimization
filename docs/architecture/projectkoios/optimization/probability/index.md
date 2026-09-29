# Samplers, probability slots, and backend adapters

## Sampler foundation

`sampling/base.py` owns the dependency-free `Sampler[SampleT, SeedT]` contract.
Its final public `sample` operation validates a positive bounded batch count,
requires an exact immutable output tuple, and delegates value normalization to
the concrete sampler. This is a behaviorful optimization sampling contract, not
a universal pipeline runtime.

A future optimization engine may wrap a sampler as its proposal strategy. The
engine remains responsible for candidate identity, outstanding proposals,
feedback through `tell`, state, and checkpoints. Workflow owns connection of
candidate output to simulation execution and routing of QoI-derived feedback
back to the engine.

## Probability sampler pipeline

`ProbabilityDistributionSampler` is a `Sampler[float, SeedT]` with two explicit
composition slots:

1. `source_slot`: a `UnitUniformSampler` producing values in `[0, 1)`;
2. `transform_slot`: an `InverseProbabilityDistributionFunction` mapping those
   values to the requested distribution.

The fixed local flow is:

```text
sample request -> unit-uniform source slot -> inverse-CDF transform slot
               -> finite immutable sample batch
```

`SplitMix64UnitUniformSampler` is the dependency-free deterministic source-slot
implementation. `BoundedUniformOptimizer` shares only its low-level
`unit_fraction` primitive, preserving the established
`splitmix64-uniform-box:1` proposal sequence without changing the optimizer's
version-one contract.

## Probability functions and distributions

`probability/base.py` owns scalar validated CDF, inverse-CDF, and composed
probability-distribution contracts. `probability/distributions.py` owns the
finite uniform parameters, CDF, inverse CDF, and composed uniform distribution.
Uniform endpoints must be finite, strictly ordered, and have a finite
representable binary64 width.

## Backend adapter organization

All adapters for one optional dependency live together:

- `adapters/numpy/probability.py` owns NumPy CDF and inverse-CDF adapters,
  `NumpyRandomStateUnitUniformSampler`, and the direct frozen-compatible uniform
  sampler. The RandomState implementation is isolated and does not mutate
  NumPy global random state.
- `adapters/scipy/probability.py` owns SciPy CDF and inverse-CDF transform slots
  over the narrow public `cdf` and `ppf` surface of a frozen `scipy.stats`
  object. It does not depend on private SciPy classes.

NumPy and SciPy remain optional extras. Adapter packages are never imported by
the package root, so core optimization has no external runtime dependency. A
SciPy inverse-CDF slot can be paired with either the SplitMix64 or NumPy source
slot.

All public scalar results must be finite. An unbounded SciPy quantile returning
infinity at probability zero or one therefore fails closed. These contracts
establish software and bounded numerical behavior only; they do not constitute
statistical or scientific validation of a selected distribution.
