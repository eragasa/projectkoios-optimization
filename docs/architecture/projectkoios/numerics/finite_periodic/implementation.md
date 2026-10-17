# Finite-periodic implementation organization

## Current and target modules

```text
projectkoios/numerics/finite_periodic/
├── __init__.py
├── meshes.py                 # centered uniform mesh record
├── finite_series.py
├── batched_finite_series.py
├── transforms.py             # dependency-free diagnostic records
├── sewing.py
├── frames.py
├── wilson.py
├── real_space.py
├── symmetry.py
├── density.py
├── numpy/
│   ├── __init__.py
│   ├── finite_series.py
│   ├── batched_finite_series.py
│   ├── transforms.py         # centered matrix transforms and bounded evaluation
│   ├── sewing.py
│   ├── frames.py
│   ├── wilson.py
│   ├── real_space.py
│   ├── symmetry.py
│   └── density.py
└── scipy/
    ├── __init__.py
    └── phase_spectra.py
```

`scipy/phase_spectra.py` contains decomposition support only. Optimized phase matching
belongs under `projectkoios.optimization.assignment`.

## Public records

Current records:

- `FiniteFourierSeriesEvaluationDefinition`
- `FiniteFourierSeriesEvaluationResult`
- `BatchedFiniteFourierSeriesEvaluationDefinition`
- `BatchedFiniteFourierSeriesEvaluationResult`
- `CenteredUniformMesh`
- `FinitePeriodicParsevalDiagnostics`
- `FinitePeriodicHermiticityDiagnostics`
- the array-owning records in `finite_periodic.numpy.matrix_series`,
  `finite_periodic.numpy.matrix_evaluation`,
  `finite_periodic.numpy.tensor_product_evaluation`, and
  `finite_periodic.numpy.transforms`

Planned records:

- `PeriodicMeshNeighbor`
- `ReciprocalMeshTranslation`
- backend-neutral transform arrays beyond the current NumPy capability
- `FinitePeriodicTruncationResult`
- `ReciprocalSewingMap`
- `OrthonormalFramePath`
- `ProjectorPath`
- `ProjectedOperatorPath`
- `WilsonLoopResult`
- `FinitePeriodicRealSpaceMomentResult`
- `PeriodicDensityResult`

## Validation

- Mesh order and dimensions are explicit.
- Transform direction and normalization are explicit.
- Sewing direction and finite-basis loss are explicit.
- Frame rank and ambient dimension are explicit.
- Closure edges are supplied explicitly.
- Phase branch conventions are explicit.
- Nonfinite and dimension-mismatched inputs fail closed.

## Verification

- Verify exact forward/inverse finite transforms.
- Verify Parseval identities and omitted-block norms.
- Verify one- and two-dimensional sewing maps by index lookup.
- Verify projector invariance under unitary frame changes.
- Verify Wilson unitarity defects and phase spectra.
- Verify gauge covariance without attaching physical meaning.
- Verify direct real-space reconstruction normalization, center, and spread formulas
  for supplied states.
- Do not implement Wannier gauge optimization or Wannier localization; those remain
  Wannier90-owned.
- Retain owner-native periodic-1D and periodic-2D regression fixtures.

## Status

Scalar and batched finite-series evaluation plus centered uniform one-, two-, and
three-dimensional matrix transforms are implemented. Package initializers remain
import-empty and the Numerics distribution includes a `py.typed` marker. Remaining
modules and records shown above are planned. Optimization alignment and assignment
contracts are designed separately so this package remains free of objective-selection
behavior.
