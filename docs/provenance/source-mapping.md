# Static source, import, test, and license mapping

This mapping was established before implementation. Recovery material is
evidence, not a canonical release.

## Accepted recovery sources

Both read-only recovery commits `ff50d06e74ad094ec84e53149d1717d66e2154fa`
and `151200940cb8c3d81f0acb4759d9879a6926a6d8` in
`/Users/eugene/repos/ksdft2effmass` carry the identical maintained application
engine tree `9a58c5fa4ce0240c55377bf2347df492d101ff6a`.

| Recovery path | Blob | SHA-256 | Destination behavior |
|---|---|---|---|
| `python/src/ksdft2effmass/application/engines/base.py` | `33ca76648401deacf8938cca7e1ff910137f2c95` | `56023cc3232cf8d5d149a8000f714be3601f11d2fae5df8195115c3fe0d26f49` | immutable optimizer identity/descriptor, without the generic execute base |
| `python/src/ksdft2effmass/application/engines/potential_fitting.py` | `d5fac55227e5c41fddd7855ddf135c00a0456c25` | `c457beb0507f575a3f03935037545d9983da63fbd63546dcdfe380eb89cabb1e` | bounds, SplitMix64 sampling, weighted loss, coordinate-pattern policy, pure Pareto/ranking records |

The source imports from `ksdft2effmass.provenance`, `pathlib`, and application
engine modules. The core extracted implementation imports only the Python
standard library and `projectkoios.optimization`; application provenance,
filesystem paths, potential/QoI models, evaluator implementations, and report
writers are not imported. NumPy and SciPy are available only through explicit
optional backend adapter modules and are not imported by the package root.

## Test mapping

The recovery software-test tree is
`c0c8557e424611ac8cfcec0ab01d958a3802fc1b`; the numerical-test tree is
`fb98950834f18b4a92f74b872e418ea99a131929`.

Accepted behavioral evidence:

- `test__EngineIdentity.py`, `test__EngineDescriptor.py` → immutable portable
  optimizer identity and descriptor tests.
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
with the accepted behaviorful optimizer contract. GMM, PyPosPack/PyMatMC2,
PyPosMat, potential models, campaign execution, provenance correlators, plot and
filesystem report artifacts, dispatch, retry, and workflow tests are excluded.

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
- Checkpoints are canonical JSON with schema, optimizer identity, and SHA-256
  validation; recovery source had no equivalent checkpoint contract.
- Evaluation, report writing, workflow, and provenance correlation were removed.
- Ranking ties use `(loss, candidate_id)`; search accepts only strict
  improvement and therefore retains earlier proposal order on ties.
- Zero and negative weights are rejected. Every numeric public record rejects
  NaN and infinity.

These adaptations are maintained design decisions, not claims of byte
identity, historical reconstruction conformance, global convergence, or
scientific validation.

## License

The donor root `LICENSE` blob is
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` (SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`)
and contains Apache License 2.0. The destination `LICENSE` retains that exact
text. No third-party source is vendored. NumPy and SciPy are declared optional
runtime adapters and are not dependencies of the core package.
