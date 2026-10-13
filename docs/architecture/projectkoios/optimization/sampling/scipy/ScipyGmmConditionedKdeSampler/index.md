# ScipyGmmConditionedKdeSampler

`projectkoios.optimization.sampling.scipy.gmm_kde.ScipyGmmConditionedKdeSampler`
reconstructs the numerical center of Eugene J. Ragasa's PyPosPack GMM/cluster-conditioned
KDE work without its filesystem, MPI, plotting, calculator, or material-campaign code.

For each adaptive stage it:

1. uses the preceding exact Pareto set/front pullback as the seed population;
2. constructs numeric decision, objective, or combined decision/objective features;
3. fits deterministic full-covariance scikit-learn Gaussian mixtures with
   `random_state=0`;
4. evaluates component counts by AIC or BIC and rejects partitions whose clusters do not
   contain more observations than decision coordinates;
5. assigns every seed observation to its predicted cluster;
6. fits a separately standardized local Gaussian KDE covariance to each cluster's exact
   decision vectors; and
7. allocates the proposal batch equally across clusters by default, matching the
   historical cluster engine's `n_samples_per_cluster` behavior.

The default feature space is the combined, unscaled decision/objective matrix. This maps
the historical `names='all'` analysis to numerical records without retaining scientific
column names. Decision-only and objective-only feature spaces remain explicit options.
BIC, full covariance, at most ten components, equal cluster allocation, and Silverman's
bandwidth are explicit algorithm parameters rather than scientific thresholds.

## Covariance behavior

`CovarianceStabilizationMethod.HISTORICAL_ZERO_EIGENVALUE` preserves the donor repair:
ordinary eigendecomposition, replacement of eigenvalues exactly equal to zero with
`1e-15`, and covariance reconstruction. It is retained for historical comparison and
still fails when floating-point rank loss appears as small negative rather than exactly
zero eigenvalues.

`CovarianceStabilizationMethod.RELATIVE_EIGENVALUE_FLOOR` is a separately identified
modernization. It symmetrizes the covariance, uses `eigh`, and floors the spectrum at a
configured fraction of its largest absolute eigenvalue. It is not represented as
historical PyPosPack behavior.

The extraction also makes sorted cluster order and exact seeded local sampling explicit.
GMM clusters remain KDE seed populations; they are not final archive compression and do
not replace exact decision-vector pullback.

## Dependency and ownership boundary

The capability is available through the optional `gmm` dependency extra and through
`ProbabilitySamplerFactory.create_gmm_conditioned_kde(...)`. Import-empty package
initializers keep NumPy, SciPy, and scikit-learn lazy. Scientific feature names, units,
normalization choices, evaluator execution, feasibility policy, and workflow persistence
remain consumer-owned.

Exact commits, blobs, hashes, licensing, and adaptations are recorded in
[`docs/provenance/source-mapping.md`](../../../../../../provenance/source-mapping.md) and
`TRANSFER.toml`.
