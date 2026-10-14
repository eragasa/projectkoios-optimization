# SciPy typing

## Status

This page specifies the current shared typing boundary for concrete optional SciPy
capabilities.

## Ownership

`projectkoios.numerics.scipy.typing` owns narrow static interfaces for third-party SciPy
objects consumed by Project Koios where upstream annotations are absent or incomplete.
The module currently exports:

- `Float64SparseMatrix`, covering the canonical sparse operations used by quadratic
  capabilities;
- `GaussianKde`, covering density evaluation, integration, log-density, and resampling;
  and
- `OptimizeResult`, covering the result fields inspected after minimization.

These interfaces describe third-party representations. They are not structural
substitutes for Project Koios solver contracts, do not select a backend, and do not add a
universal numerical hierarchy.

## Dependency boundary

Importing `projectkoios.numerics.scipy` alone imports no optional backend. Importing its
`typing` module explicitly imports the shared NumPy typing definitions but does not import
SciPy at runtime. Concrete SciPy modules still perform their own explicit optional SciPy
imports.

Consumers cast a value only after constructing it through SciPy or checking its sparse
representation. This confines unavoidable trust in an incompletely typed third-party API
to one visible boundary instead of spreading `Any` through numerical code.

## Verification

Mapped integration tests exercise the exact sparse, KDE, and optimization-result
operations declared by the interfaces against installed SciPy objects. Separate unit
tests verify that the package initializer is import-empty. Strict mypy includes both the
NumPy and SciPy typing test trees.
