# Finite-periodic numerics

`projectkoios.numerics.finite_periodic` owns direct finite-periodic transformations and
geometric calculations that do not choose an optimum through their public contract.

## Status

Ordered scalar and shared-grid batched finite-series evaluation are implemented.
Centered uniform one-, two-, and three-dimensional matrix transforms, bounded selected-
mode evaluation, Parseval evidence, and cyclic Hermiticity evidence are implemented in
`meshes.py`, `transforms.py`, and their NumPy backend. Other inventory modules remain
prospective and require capability-specific architecture review before implementation.

## Modules

| Module | Owned behavior |
|---|---|
| `meshes.py` | centered uniform mesh geometry; other finite indices and wrapped-neighbor records remain prospective |
| `finite_series.py` / `batched_finite_series.py` | ordered finite Fourier-series evaluation at supplied real coordinates |
| `transforms.py` | exact centered finite-mesh matrix transforms, bounded inverse evaluation, Parseval diagnostics, and cyclic Hermiticity diagnostics; generic truncation remains prospective |
| `sewing.py` | explicit nonunitary partial-shift maps between finite reciprocal bases |
| `frames.py` | frame paths, projectors, and projected-operator paths |
| `wilson.py` | ordered link products, unitarity defects, principal phases, unwrapping, and winding |
| `real_space.py` | direct finite Born–von Karman reconstruction, normalization, centers, and spreads for supplied states |
| `symmetry.py` | finite-group actions and periodic-coordinate transforms |
| `density.py` | deterministic periodic density reconstruction |

## Optimization boundary

The following operations do not belong in this numerical package:

- Procrustes alignment;
- selection of gauge transformations by an alignment objective;
- optimized frame transport;
- phase synchronization;
- linear-sum phase matching; and
- minimax point-set assignment.

Those operations belong in `projectkoios.optimization` and consume finite-periodic
records from this package.

## Interpretation boundary

Consumers own the physical meaning of reciprocal coordinates, sewing, gauges, Wilson
phases, centers, polarization, topology, densities, and equivalence thresholds. This
package does not optimize Wannier gauges or implement Wannier localization; Wannier
localization remains Wannier90-owned.

## Pages

- [Finite Fourier-series evaluation](finite_series/index.md)
- [Shared-grid batched finite Fourier-series evaluation](batched_finite_series/index.md)
- [Centered uniform finite-mesh matrix transforms](transforms/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Numerics namespace](../index.md)
