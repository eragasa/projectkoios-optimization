# Centered uniform finite-mesh matrix Fourier transforms

This capability owns unit-free, metric-free Fourier mechanics for complete one-, two-,
and three-dimensional centered uniform tensor meshes. Coordinates are reduced reciprocal
coordinates \(q\), modes are integer lattice coordinates \(n\), and every phase is
\(\exp(\pm 2\pi i q\cdot n)\).

## Public contract

Dependency-free records:

- `CenteredUniformMesh` describes the common positive axis size and exact per-axis
  offsets for even or odd parity;
- `FinitePeriodicParsevalDiagnostics` records the two Parseval energies and absolute
  defect; and
- `FinitePeriodicHermiticityDiagnostics` records twisted cyclic and ordinary
  non-Nyquist coefficient defects.

NumPy records and operations:

- `NumpyCenteredUniformMeshMatrixFourierTransformDefinition` stores complete matrix
  samples;
- `NumpyCenteredUniformMeshMatrixFourierTransformer` performs the normalized forward
  transform;
- `NumpyCenteredUniformMeshMatrixFourierTransformResult` correlates the complete
  centered inventory and backend evidence;
- `NumpyFiniteFourierMatrixSeries` stores a complete or selected ordered matrix series;
- `NumpyFiniteFourierMatrixEvaluationDefinition` and
  `NumpyFiniteFourierMatrixEvaluationResult` retain bounded evaluation inputs and
  outputs;
- `NumpyFiniteFourierMatrixSeriesEvaluator` evaluates selected modes without an
  unbounded point-by-mode phase matrix;
- `NumpyTensorProductFourierMatrixEvaluationDefinition` and
  `NumpyTensorProductFourierMatrixSeriesEvaluator` evaluate complete or selected
  construction-box inventories on Cartesian-product grids by bounded separable
  contractions;
- `NumpyFinitePeriodicParsevalAnalyzer` computes complete-transform Parseval evidence;
  and
- `NumpyCenteredUniformMeshHermiticityAnalyzer` computes origin-aware cyclic
  Hermiticity evidence.

## Ownership boundary

Numerics owns:

- parity-aware centered mode enumeration;
- explicit mesh-offset origin phases;
- forward normalization by \(1/N^d\);
- positive-exponent finite-series evaluation;
- caller-visible phase-matrix and separable working-array element bounds;
- complete-transform Parseval arithmetic; and
- cyclic wrap-phase and ordinary coefficient Hermiticity defects.

Consumers own:

- physical direct and reciprocal lattice vectors;
- units and validation of lattice duality;
- metrics such as \(G=A^\mathsf{T}A\);
- shell definitions and complete-shell selection;
- energy-reference normalization and error channels;
- tolerance selection and scientific dispositions; and
- retained-state overlap metrics or nonorthogonal orbital bases.

## Geometry independence

The transform never maps reduced coordinates into Cartesian space and never applies a
Euclidean metric. It owns only \(q\), \(n\), and the phase
\(\exp(\pm2\pi i q\cdot n)\). Under an angular reciprocal convention,
\(B=2\pi A^{-\mathsf{T}}\), \(k=Bq\), and \(R=An\), so

\[
 \exp(i k\cdot R)=\exp(2\pi i q\cdot n).
\]

A cycles-per-length convention writes the equivalent physical phase differently.
Consumers declare and validate that convention; Numerics does not define physical
\(k\). The same reduced contract therefore applies to orthogonal cells, oblique
two-dimensional cells, and FCC primitive coordinates. Numerical tests verify the
oblique and FCC identities using the angular reciprocal convention.

## Evidence boundary

Passing reconstruction, Parseval, or Hermiticity checks verifies represented numerical
relationships only. It does not validate a physical model, lattice choice, gauge,
truncation, or scientific claim.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Schematics](schematics.md)
- [Scientific boundary](scientific.md)
- [Finite-periodic namespace](../index.md)
