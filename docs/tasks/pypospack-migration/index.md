# PyPosPack-to-Optimization migration

PyPosPack is the source authority for all transferred historical Pareto behavior and its
adaptive KDE sampling machinery.

## Status

- State: implemented and provenance-bound.
- Source commit: `be453fa7191e55a0426f66e8b5b5b0b103c8b29d`.
- Canonical architecture:
  [Pareto optimization](../../architecture/projectkoios/optimization/multiobjective/pareto/index.md).
- Machine-readable mapping: [`TRANSFER.toml`](../../../TRANSFER.toml).
- Detailed mapping: [source mapping](../../provenance/source-mapping.md).

## Transferred inventory

- Performance-sensitive Pareto selector.
- Iterative Pareto sampling-stage policy.
- Global Gaussian KDE sampler.
- `Silverman1986_h` bandwidth function.
- `Chiu1999_h` bandwidth function.
- GMM-conditioned local KDE sampler.
- Decision, objective, and combined GMM feature spaces.
- AIC/BIC model selection.
- Exact observation-row pullback.
- Equal and mixture component allocation.
- Historical exact-zero covariance repair.
- Explicit modern relative-eigenspectrum covariance stabilization.

## Pareto selector provenance

- Path: `pypospack/pareto.py`.
- Blob: `217ea0a75e60c79529f566aace35c1e14d4c627b`.
- SHA-256: `dd0a541d16b34ccc803967e37471ea0f26f9068725a5c69c83454400bb0b0198`.
- Preserved behavior: chunk pruning, comparison order, early breaks, and historical
  weak-dominance duplicate handling.
- Documented correction: retain the final incomplete chunk.

## Iterative sampler provenance

- Path: `pypospack/pyposmat/engines/mc_sampler_iterate.py`.
- Blob: `a39c7ba9956d7b6fac6399d3fdf915bb3418bed8`.
- SHA-256: `bdaaca875f52e82fa17e440dd30a3e7dd5db8df56db2698f4f263bf723838c6e`.
- Preserved behavior: ordered sampler stages and preceding-archive feedback.
- Optimization boundary: forward evaluation remains external.

## Global KDE provenance

- Bandwidth path: `pypospack/kde.py`.
- Bandwidth blob: `36e4a3ac656140501781419b2f6557456ff40c95`.
- Bandwidth SHA-256:
  `011bd3766e02a349b51c9a229029023f2b98ffa740e5f47ff4c0ecf047fcee9f`.
- Sampler path: `pypospack/pyposmat/engines/mc_sampler.py`.
- Sampler blob: `a4c5d595e08df98aaca9886a694a913a339804eb`.
- Sampler SHA-256:
  `3aa52f84f51875d2cae8080cf02ae6b6e1d194229641e25bedd7c53da7b94016`.
- Preserved behavior: donor orientation, `gaussian_kde(X, h)`, and one-vector-at-a-time
  resampling.

## GMM-conditioned KDE provenance

- GMM analysis path:
  `dev/pyposmat/analysis/GaussianMixtureModel/gmm_analysis.py`.
- GMM analysis blob: `9d1284398aaff77b38f11bf34859f79e02d40b42`.
- GMM analysis SHA-256:
  `c4a6eb17b0bc68d3fea443c3f1afc37c10d2cbd6f2ebd93cb47018eb87c2fd40`.
- Pullback path:
  `dev/pyposmat/analysis/GaussianMixtureModel/dev__GmmClusterAnalysis.py`.
- Pullback blob: `78f7ec8e08d138963fe4cd10b722d400d2ad76dc`.
- Pullback SHA-256:
  `7928fa82677b1d66453beaef02588fe3b96857d8933a75f5431a032f47dd286b`.
- Cluster sampler path: `pypospack/pyposmat/engines/cluster_engine.py`.
- Cluster sampler blob: `2015d7fe38534b61fff735576e701f17a41f196b`.
- Cluster sampler SHA-256:
  `b58d5b939e654968cda3ae28a569721aba713abc1323b25d9e85a9e1b826120b`.
- Covariance source commit: `024f3280b463984b39f6c76541cda4e64a77885e`.
- Covariance path: `pypospack/statistics.py`.
- Covariance blob: `655aaa0faad6189a90dcd8778f3e266f73e7def3`.
- Covariance SHA-256:
  `40828df796d8f200ec2d401d20f2c9fd9c3b1a27356b9287a867c5ad8e9f278a`.

## Optimization-authored extensions

- Recursive Pareto-shell decomposition.
- Shell-contrastive KDE sampling.
- Duplicate grouping before protected selector invocation.
- These extensions are not PyPosPack translations.

## Protection rules

- Keep the selector and optimizer under `multiobjective/pareto`.
- Preserve exact source, commit, path, blob, SHA-256, license, adaptation, and tests.
- Require donor-relative correctness for selector changes.
- Require timing evidence for performance-sensitive selector changes.
- Require explicit human authorization for moves, splits, renames, and refactors.
