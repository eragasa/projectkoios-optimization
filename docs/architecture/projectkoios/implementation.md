# Implementation organization

## Distribution

The `projectkoios-optimization` distribution packages both namespace portions:

```text
src/python/projectkoios/
├── numerics/
└── optimization/
```

The repository remains dependency-free by default. NumPy, SciPy, and scikit-learn remain
optional capability-local backends.

## Import policy

- Every package initializer is import-empty.
- Public imports identify the defining module.
- Backend-neutral records do not import optional dependencies.
- Backend implementations live under `numpy`, `scipy`, or `sklearn` subpackages.
- One class has one canonical defining module.
- Replaceable solvers use capability-local nominal abstract classes as clean SDK
  integration points, never structural protocols.
- Capability factories accept closed method enums and construct supported concrete
  subclasses.
- Higher-level SDK APIs accept the nominal abstract class for explicit implementation
  injection.
- External packages subclass the capability abstract class and pass its conformance
  suite; they do not mutate an unbounded owner registry.
- Compatibility modules import canonical objects without copying behavior.

## Naming policy

- Stateless operations use descriptive functions.
- Stateful or configured operations use classes.
- Backend classes use explicit prefixes such as `Numpy`, `Scipy`, or `Sklearn`.
- Behavioral suffixes include `Solver`, `Fitter`, `Selector`, `Constructor`,
  `Calculator`, `Transporter`, `Matcher`, `Clusterer`, and `Sampler`.
- Result names include `Result`, `Fit`, `Minimum`, `Support`, `Assignment`, and
  `Clustering`.
- Generic `Manager`, `Helper`, `Utils`, and behaviorless `Base` types are excluded.

## Documentation mirroring

Source and test ownership mirror architecture paths:

```text
src/python/projectkoios/<namespace>/<capability>/<module>.py
tests/projectkoios/<namespace>/<capability>/test__<PublicType>__<kind>.py
docs/architecture/projectkoios/<namespace>/<capability>/index.md
```

Each implemented capability records:

- public contract;
- mathematical definition;
- backend behavior;
- validation and failures;
- provenance and adaptations; and
- software and numerical verification scope.

## Migration state

Current implementation is concentrated under `projectkoios.optimization`. Namespace
realignment occurs in bounded compatibility-preserving slices. Protected Pareto modules
remain in place. New source is not added until its architecture, provenance, and test
mapping are reviewed.
