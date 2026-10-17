# Remove broad typing, generic returns, postconstruction mutation, and magic tests

## Status

- State: in progress.
- Scope: production Python plus mapped verification tests.
- Classification: contract clarity and maintainability debt, not a demonstrated
  numerical defect.

## Enforced outcomes

Production code must reach all of the following:

1. no return annotation contains a class or function type parameter;
2. no `Any` annotation;
3. no `object` parameter or return annotation;
4. no direct `__setattr__` call; and
5. frozen records contain canonical storage from initial construction.

Tests must use capability-local `fixtures/` packages, named immutable case records, and
independently derived expectations. Moving an unexplained literal into a constant does
not resolve a magic test. Mathematical identities and explicit boundary cases remain
valid when their role is named by the case or parameter identifier.

## Baseline and progress

The initial AST inventory reported:

| Smell | Initial count | Current count |
|---|---:|---:|
| generic return annotations | 5 | 0 |
| `Any` annotations | 0 | 0 |
| `object` parameter/return annotations | 91 | 71 |
| direct `__setattr__` calls | 129 | 129 |

The generic-return removal replaced the universal generic scalar sampler with a concrete
finite-float sampler contract and moved scalar and batched Procrustes execution returns
to their family-specific solver bases. Current box and periodic-harmonic value factories
now use `SupportsFloat` rather than `object` inputs.

The first fixture migration established capability-local packages for scalar/batched
Procrustes, box-quadratic minima, and periodic harmonics. Their named cases derive
expected evidence from declared mathematical relations. Independent review reported
`NO_BLOCKING_FINDINGS` after the harmonic 3-4-5 oracle was made independent of the
production canonicalization helper.

A literal inventory found 141 test files containing numeric literals and 2,813
non-identity numeric occurrences. Those counts are discovery aids, not an automatic
finding count: each case requires review to distinguish a named boundary or analytical
identity from unexplained setup data.

## Sequence

1. Replace broad annotations with capability-specific protocols, exact records, or
   concrete unions.
2. Replace postconstruction normalization with explicit value factories and canonical
   raw constructors.
3. Move repeated and analytical test setup into capability-local `fixtures/` packages.
4. Reduce the related McCabe findings while preserving numerical operation order and
   failure classifications.
5. Add a zero-tolerance AST gate after the production inventory reaches zero.

The provenance-protected GMM validator and records remain excluded until their owner
explicitly authorizes that named source boundary.

## Definition of done

- [ ] All four production AST inventories are zero.
- [ ] Every mapped test uses named boundary cases or independently derived fixtures.
- [ ] No test obtains its expected value from the production operation under test.
- [ ] Ruff, formatting, strict mypy, full tests, compileall, lock verification,
      reproducible artifacts, and isolated wheel smoke pass.
- [ ] Independent review reports no blocking contract, numerical, ownership, or test-
      oracle finding.
