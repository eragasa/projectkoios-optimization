# Gaussian-mixture clustering

`optimization/clustering/` owns dependency-free Gaussian-mixture request, score,
assignment, component, completion, and failure records. An ordered finite numeric matrix
crosses the capability boundary; feature names, units, scientific interpretation,
selection thresholds, and campaign identity remain consumer-owned.

The optional implementation in
[`clustering/scipy/gmm.py`](../../../../../src/python/projectkoios/optimization/clustering/scipy/gmm.py)
standardizes each feature, evaluates a bounded inventory of full-covariance models,
records AIC and BIC, selects by BIC, canonically orders components by standardized mean,
and pulls every assignment back to its exact input row. Package initializers remain
import-empty, so importing the core package does not import NumPy or scikit-learn.

This standalone clustering capability is distinct from
`ScipyGmmConditionedKdeSampler`. It returns clustering evidence and does not sample,
compress an archive, interpret clusters, or change the protected GMM-conditioned KDE
pipeline.

See
[`ScipyGaussianMixtureClusterer`](scipy/ScipyGaussianMixtureClusterer/index.md)
and the [source mapping](../../../../provenance/source-mapping.md).
