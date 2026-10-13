# Static source, import, test, and license mapping

This mapping was established before implementation. Recovery material is
evidence, not a canonical release. The iterative multi-objective optimization block
originates in Eugene J. Ragasa's PhD work; surviving behavior and provenance are
scattered across
PyPosPack and multiple research repositories. Each retained behavior
therefore requires an exact source mapping rather than attribution to only the nearest
recovery wrapper.

## Accepted recovery sources

Both read-only recovery commits `ff50d06e74ad094ec84e53149d1717d66e2154fa`
and `151200940cb8c3d81f0acb4759d9879a6926a6d8` in
`/Users/eugene/repos/ksdft2effmass` carry the identical maintained application
engine tree `9a58c5fa4ce0240c55377bf2347df492d101ff6a`.

| Recovery path | Blob | SHA-256 | Destination behavior |
|---|---|---|---|
| `python/src/ksdft2effmass/application/engines/base.py` | `33ca76648401deacf8938cca7e1ff910137f2c95` | `56023cc3232cf8d5d149a8000f714be3601f11d2fae5df8195115c3fe0d26f49` | behaviorful engine ports and explicit concrete factory methods, without the generic execute base or passive identity records |
| `python/src/ksdft2effmass/application/engines/potential_fitting.py` | `d5fac55227e5c41fddd7855ddf135c00a0456c25` | `c457beb0507f575a3f03935037545d9983da63fbd63546dcdfe380eb89cabb1e` | bounds, SplitMix64 sampling, weighted loss, coordinate-pattern policy, and result records |

The source imports from `ksdft2effmass.provenance`, `pathlib`, and application
engine modules. The core extracted implementation imports only the Python
standard library and `projectkoios.optimization`; application provenance,
filesystem paths, potential/QoI models, evaluator implementations, and report
writers are not imported. NumPy and SciPy are available only through explicit
capability-local backend modules. Package initializers import no capabilities or
optional backends.

## PyPosPack iterative Pareto and performance sources

The interchangeable stage policy comes from public PyPosPack commit
`be453fa7191e55a0426f66e8b5b5b0b103c8b29d`, path
`pypospack/pyposmat/engines/mc_sampler_iterate.py`, blob
`a39c7ba9956d7b6fac6399d3fdf915bb3418bed8`, SHA-256
`bdaaca875f52e82fa17e440dd30a3e7dd5db8df56db2698f4f263bf723838c6e`.
The owner extraction retains the ability to use a different sampler in each iteration,
including adaptive samplers that consume the preceding Pareto archive. Filesystem, YAML,
MPI, logging, calculator execution, and string-selected sampling modes are not retained.

The optional Gaussian KDE sampler also comes from that public snapshot. The preserved
bandwidth functions are in `pypospack/kde.py`, blob
`36e4a3ac656140501781419b2f6557456ff40c95`, SHA-256
`011bd3766e02a349b51c9a229029023f2b98ffa740e5f47ff4c0ecf047fcee9f`.
The sampler loop is in `pypospack/pyposmat/engines/mc_sampler.py`, blob
`a4c5d595e08df98aaca9886a694a913a339804eb`, SHA-256
`3aa52f84f51875d2cae8080cf02ae6b6e1d194229641e25bedd7c53da7b94016`.
The bandwidth implementation was introduced by Eugene J. Ragasa in commit
`07193e06886b9f072558d84ffc73bf809ee3a3e4` on 2018-03-12. The extraction preserves
`Silverman1986_h`, `Chiu1999_h`, decision-vector orientation, SciPy
`gaussian_kde(X, h)` construction, and one-vector-at-a-time resampling order. An enum and
`KdeBandwidthFactory` replace only the donor's inconsistent `silverman1984`,
`silverman1985`, and undefined bare `Silverman1986` dispatch spellings. Explicit seeded
`RandomState` replaces mutation of NumPy's process-global generator. PyPosMat files,
MPI, logging, calculators, constrained potential parameters, and campaign loops are not
retained.

The GMM-conditioned KDE extraction combines four explicit pieces of PyPosPack lineage.
Full-covariance `GaussianMixture`, fixed `random_state=0`, AIC/BIC model comparison, and
parameter/QoI/all feature selection survive in commit
`be453fa7191e55a0426f66e8b5b5b0b103c8b29d`, path
`dev/pyposmat/analysis/GaussianMixtureModel/gmm_analysis.py`, blob
`9d1284398aaff77b38f11bf34859f79e02d40b42`, SHA-256
`c4a6eb17b0bc68d3fea443c3f1afc37c10d2cbd6f2ebd93cb47018eb87c2fd40`.
The exact simulation-identifier pullback and per-cluster representative analysis survive
in `dev__GmmClusterAnalysis.py`, blob
`78f7ec8e08d138963fe4cd10b722d400d2ad76dc`, SHA-256
`7928fa82677b1d66453beaef02588fe3b96857d8933a75f5431a032f47dd286b`.

Cluster-conditioned KDE dataflow survives in
`pypospack/pyposmat/engines/cluster_engine.py`, blob
`2015d7fe38534b61fff735576e701f17a41f196b`, SHA-256
`b58d5b939e654968cda3ae28a569721aba713abc1323b25d9e85a9e1b826120b`:
validate cluster size, iterate clusters, fit KDE to each cluster's free parameters, and
request an equal sample count per cluster. The extraction replaces data files and
scientific columns with exact Pareto decision/objective records, sorts cluster order for
determinism, and rejects candidate component partitions too small for local KDE. The
donor `isValidPartition()` returned `True` from inside its first-cluster loop; the
extraction applies the stated size requirement to every predicted cluster. It does not
retain t-SNE, DBSCAN, filesystem, MPI, calculators, plots, or campaign orchestration.

The historical covariance workaround is preserved from commit
`024f3280b463984b39f6c76541cda4e64a77885e`, path `pypospack/statistics.py`, blob
`655aaa0faad6189a90dcd8778f3e266f73e7def3`, SHA-256
`40828df796d8f200ec2d401d20f2c9fd9c3b1a27356b9287a867c5ad8e9f278a`.
It replaces eigenvalues exactly equal to zero with `1e-15` and reconstructs the
covariance. The symmetric `eigh` relative eigenspectrum floor is clearly separated as a
modern numerical alternative. GMM-labeled cluster members remain KDE seed populations;
they are not represented as archive compression.

The owner-authored Pareto-shell decomposition and contrastive KDE sampler are not donor
translations. Eugene Joseph M. Ragasa recorded the recursive shell pullback and
positive-versus-dominated-shell sampling idea on 2026-10-13. Repository and Git-history
searches found no implementation in PyPosPack, KSD, or the frozen Frankenstein design
evidence. The dated local design record is
`docs/research/pareto-shell-contrastive-sampling.md`. These additions call the preserved
selector and bandwidth capability but remain explicitly outside claims of historical
reconstruction or literature novelty. They also remain separate from the reconstructed
GMM-conditioned KDE capability.

The performance-critical Pareto selector comes from the same public PyPosPack commit,
`be453fa7191e55a0426f66e8b5b5b0b103c8b29d`, path `pypospack/pareto.py`, blob
`217ea0a75e60c79529f566aace35c1e14d4c627b`, SHA-256
`dd0a541d16b34ccc803967e37471ea0f26f9068725a5c69c83454400bb0b0198`.
KSD is not the origin of this engine behavior. The identical blob merely survives as
recovery evidence under KSD commit `ff50d06e...` at
`python/src/ksdft2effmass/pypospack/pareto.py`.

Eugene J. Ragasa developed the original implementation. Its source header credits
Dmitriy Morozov of Lawrence Berkeley National Laboratory with December 2016 Pareto
performance improvements. The upstream commit is BSD-2-Clause; the exact retained
license is `licenses/PyPosPack-BSD-2-Clause.txt` and the attribution is repeated in
`NOTICE`.

Both upstream files are BSD-2-Clause and part of the PhD-work lineage; KSD copies are
later recovery evidence, not source origin.

The adaptive lazy chunk-pruning loop and comparison order are preserved. The sole
algorithmic correction replaces `zip(*[iter(indices)] * size)`, which silently discarded
a final short batch, with Python 3.14 `itertools.batched`. A 501-point regression and a
20,000-point, ten-objective numerical comparison against the full brute-force result
protect that correction. Further movement or refactoring of
`optimization/multiobjective/pareto/` requires explicit human authorization.

## Test mapping

The recovery software-test tree is
`c0c8557e424611ac8cfcec0ab01d958a3802fc1b`; the numerical-test tree is
`fb98950834f18b4a92f74b872e418ea99a131929`.

Accepted behavioral evidence:

- `test__EngineIdentity.py`, `test__EngineDescriptor.py` → lineage evidence for the
  concrete engine factory boundary; passive identity/descriptor records are not retained.
- `test__PotentialFittingEngine.py` → replaced by explicit ask/tell, injected
  forward-evaluator, and separate results-handler tests.
- `test__PotentialFittingAnalyticReportAssembler.py`,
  `test__PotentialSamplingAnalyticReportAssembler.py`, and
  `test__PotentialFittingIterationSeriesReportAssembler.py` → dependency-free
  deterministic ranking and Pareto records only.
- `test__BoundedUniformParameterSampler.py` → exact SplitMix64 reference
  sequence.
- `test__BoundedCoordinatePatternLossMinimizer.py` → ask/tell bounded quadratic
  numerical case and resumability equivalence.

`test__BaseEngine.py` is not ported because a universal execute base conflicts
with the accepted behaviorful optimizer contract. The exact PyPosPack Pareto selector, global Gaussian KDE sampler, and GMM-conditioned local KDE sampler are
the retained PyPosPack-lineage behaviors. Recursive Pareto-shell pullback and contrastive
KDE sampling are separately identified owner-authored research additions. GMM campaign
analysis and plotting, PyMatMC2, unrelated PyPosPack/PyPosMat filesystem
and calculator machinery, potential models, campaign execution, provenance
correlators, plot and report artifacts, dispatch, retry, and workflow tests are excluded.

## Architecture source lineage

Read-only Frankenstein commit
`3eb562f2d6167ec20d6f2c892c517509a7abf283` supplies historical architecture
source-lineage evidence consulted during extraction:

- `docs/architecture/decisions/0001-separate-optimizer-from-potential-problem.md`,
  blob `addf57563e61fc8b6200b14fc7f9e193206d2309`;
- `docs/architecture/decisions/0006-solve-potential-fitting-as-forward-evaluated-inverse-problem.md`,
  blob `374446768b2c2c0de1b2f3d8a465161e1a2bb41e`;
- multi-objective documentation tree
  `ec1c32a106a5fa8331966955b12fa27e17cde9c7`;
- inverse-problem forward-evaluation documentation tree
  `ff193d1dea39018e4ade26a50a3275fa581c1b93`.

These frozen artifacts record source lineage; they are not current architecture
authority. The living ownership split is documented by this repository: an
optimizer proposes and updates; a functional evaluator returns raw observations;
a results handler retains observations, derives declared feedback, and calls
`tell` on the exact proposing engine.

The same frozen commit supplies compatibility evidence, not canonical source,
for probability behavior:

- `src/projectkoios/frankensteins/adapters/bindings/numpy/probability.py`, blob
  `92ae8ab087fa0e1fc85df49b6431910b634d9fac`, SHA-256
  `8184ec4cf9e59085532d8f421d2146cda500eea21b25dcefb836ff1a93cc1d42`;
- `src/projectkoios/frankensteins/mathematics/probability.py`, blob
  `edb4c7c6e02fe8e74a9af71d046be0866be3caae`, SHA-256
  `44d9ceb2a4028c01c713e3e68216d4bbcaa93230a347372b2b559af66c3817e8`.

The maintained implementation preserves isolated NumPy `RandomState` uniform
sequence behavior in the optional NumPy adapter. It does not copy the
behaviorless nominal parameter base, validation framework, statistical
validator, or NumPy dependency into the core namespace. SciPy CDF/PPF support
is a new narrow structural adapter rather than recovered behavior.

## Semantic adaptations

- Potential-specific records became dimension-checked generic parameter and
  metric records.
- Empty parameter spaces and fixed coordinates are supported; reversed bounds
  remain invalid.
- Probability distributions expose behaviorful scalar CDF and inverse-CDF
  contracts. `ProbabilityDistributionSampler` is a generic `Sampler` subtype
  with separate inverse-CDF transform and SplitMix64 or optional NumPy
  RandomState source slots.
- Sampling became a resumable ask/tell engine. Multiple outstanding proposals
  may be told out of order; unknown and duplicate tells fail closed.
- Coordinate minimization became an explicit one-outstanding-candidate ask/tell
  state machine. Repeated `ask` returns the same pending candidate.
- Each resumable numerical algorithm exposes its own immutable typed state record.
  JSON, checksums, persistence, and lifecycle wrapping remain outside Optimization.
- Evaluation, report writing, workflow, and provenance correlation were removed.
- Ranking ties use `(loss, candidate_id)`; search accepts only strict
  improvement and therefore retains earlier proposal order on ties.
- Zero and negative weights are rejected. Every numeric public record rejects
  NaN and infinity.

These adaptations are maintained design decisions, not claims of byte
identity, historical reconstruction conformance, global convergence, or
scientific validation.

## Affine and diagonal-metric owner export

KSD owner-export candidate commit
`879996761b67f8b78dc30dead3f03d28ff7cbea6` (root tree
`80f994634d823aa5bca426d7451564f60d113c4e`) supplies the reviewed source evidence
under export tree `5e90d8596731dec6cffc29533830c0332d465eed`:

| Candidate path | Blob | SHA-256 | Owner adaptation |
|---|---|---|---|
| `exports/optimization/affine.py` | `9739498c4e071d6d45c3f6f8abc5f9339d302112` | `1c2473e40aa94a466c4b1b92605a2f8b83ba583853d43970dc8e172ef08e10ad` | split dependency-free affine records/evaluation from the optional constructor; replace normal-equation solving with direct weighted-design least squares and numerical rank rejection |
| `exports/optimization/metric.py` | `191fb449f1f797fe3b288f1aceffa3cbd8725fc0` | `576e84d81abc245a5f37b05e79a6c7ee4cfbccb240e73a5e6fc3bc139e227efc` | retain immutable unit-free positive diagonal primal/dual geometry under the owner namespace |
| `exports/optimization/tests/integration/test_affine_quadratic.py` | `bc678c5a52742eff71f84df127eaad4ec6e6d34c` | `71ae002966de48441237fb55899b3ff63bc34f8956b9803174a2d9c8ad44dbbf` | map across source-mirrored integration, numerical, and property modules; add exact dependent-column regression |
| `exports/optimization/tests/unit/test_metric.py` | `05338c737185b9015c75a570ef9d9e4d164e7d41` | `f062b24a16b5b3bf970c7ed04da8be0d7b54d7940f8abcdd256e0a4251f14603` | map across source-mirrored unit and numerical metric modules |

The candidate's generic unit-free ownership was accepted in principle, but its original
constructor was not copied unchanged. Direct normal-equation solving could accept a
mathematically dependent design after binary64 Gram rounding. The owner implementation
first applies the selected NumPy or SciPy scale-aware numerical rank decision directly
to the weighted design and fails without regularization when rank is deficient. Final
represented SPD acceptance remains delegated to the matching quadratic backend.

All tests now live under `tests/projectkoios/optimization/`, mirroring the maintained
source namespace. Filenames and strict pytest markers preserve unit, integration,
numerical, and property classification without top-level taxonomy directories.

Feasibility/nonemptiness certificates, common-witness search, support optimization, set
distance, interval arithmetic, directed rounding, physical units, scientific thresholds,
and certificate interpretation were not transferred.

## License

The donor root `LICENSE` blob is
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` (SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`)
and contains Apache License 2.0. The destination `LICENSE` retains that exact
text. No third-party source is vendored beyond the identified BSD-2-Clause PyPosPack
adaptations. NumPy, SciPy, and scikit-learn are declared optional capability
implementations and are not dependencies of the core package. Pymoo 0.6.2 and
Matplotlib are optional benchmark-only dependencies used for the external NSGA-II
reference and recorded figures; no behavior from either package is included in the
distribution source.
