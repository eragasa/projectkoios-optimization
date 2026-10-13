# Sampling

`sampling/base.py` owns the dependency-free `Sampler[SampleT, SeedT]` contract for exact
immutable batches.

`sampling/uniform.py` owns `UniformRandomSampler`, which maps deterministic SplitMix64
fractions into an ordered finite `ParameterSpace`. It produces decision vectors only: it
does not accept objective feedback, rank candidates, or track a scalar best value.

`sampling/scipy/kde.py` owns the optional SciPy Gaussian KDE sampler extracted from
PyPosPack. It preserves the original `Silverman1986_h`, `Chiu1999_h`, data orientation,
`gaussian_kde(X, h)` construction, and one-vector-at-a-time resampling order. The
filesystem KDE-data file is replaced by the preceding stage's explicit Pareto set.
`KdeBandwidthMethod` and `KdeBandwidthFactory` replace the donor's inconsistent string
dispatch while resolving directly to the preserved functions.

`sampling/scipy/gmm_kde.py` owns the optional PyPosPack-lineage GMM-conditioned KDE
sampler. It fits deterministic full-covariance Gaussian mixtures to numeric Pareto
seed features, retains exact cluster-member pullback to decision vectors, and fits a
local KDE per cluster. BIC/AIC selection, equal historical cluster allocation, exact-zero
historical covariance repair, and the separate modern relative eigenspectrum floor are
explicit. GMM clusters are KDE seed populations rather than compressed final archives.

`sampling/scipy/pareto_shells.py` owns the optional owner-authored Pareto-shell
contrastive KDE sampler. It recursively obtains shell-zero and dominated-shell decision
vectors from the complete evaluated history, draws a bounded candidate pool from the
shell-zero KDE, and ranks that pool by a positive-to-negative log-density ratio. It is a
research extension rather than recovered PyPosPack behavior; the historical global KDE
implementation remains unchanged in `sampling/scipy/kde.py`.

`ProbabilitySamplerFactory` in `probability/factory.py` provides explicit construction of
uniform parameter sampling, inverse-CDF distribution samplers, the preserved optional
SciPy KDE sampler, GMM-conditioned KDE, and the separate contrastive shell sampler.
Samplers may be injected
into Pareto sampling stages; they are not owned by evaluation engines.
