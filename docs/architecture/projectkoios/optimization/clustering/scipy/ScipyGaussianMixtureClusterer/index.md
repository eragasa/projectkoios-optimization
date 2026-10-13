# `ScipyGaussianMixtureClusterer`

## Contract

`ScipyGaussianMixtureClusterer.cluster(observations, request)` accepts an ordered,
nonempty rectangular matrix of finite numeric tuples. It returns either
`GaussianMixtureClusteringCompleted` or a closed numerical failure. Every successful
assignment carries the zero-based index of the exact input row; clustering never
reorders or compresses the observations.

Empty input returns `NO_OBSERVATIONS`. Malformed, nonrectangular, Boolean, textual, NaN,
or infinite input fails at the public boundary. A single observation produces an
explicit one-component standardized-space result with zero covariance and no backend
fit.

## Preserved numerical behavior

For two or more observations, the implementation preserves the KSD recovery behavior:

1. calculate population means and standard deviations feature by feature;
2. replace exactly zero feature scales with `1.0`;
3. fit component counts from one through
   `min(maximum_components, floor(observation_count / 2))`;
4. use scikit-learn full covariance, the requested random seed, `n_init=5`,
   `max_iter=500`, and `reg_covar=1e-8`;
5. retain both AIC and BIC for every fitted model;
6. minimize `(BIC, component_count)`;
7. sort components lexicographically by standardized mean and then original component
   index; and
8. return canonical assignments, maximum posterior probabilities, component weights,
   standardized means, covariances, convergence state, iteration count, and backend
   version.

## Boundaries

The capability does not own feature names, scientific semantics, campaign identity,
KDE sampling, archive reduction, plotting, persistence, or model acceptance. The
existing
[`ScipyGmmConditionedKdeSampler`](../../../sampling/scipy/ScipyGmmConditionedKdeSampler/index.md)
remains separate and unchanged.

Source:
[`clustering/scipy/gmm.py`](../../../../../../../src/python/projectkoios/optimization/clustering/scipy/gmm.py).
Provenance: [source mapping](../../../../../../provenance/source-mapping.md).
