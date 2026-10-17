# User manual

`user_manual.pdf` is the checked user-facing manual. Its LaTeX source is
`user_manual.tex`, with chapters under `chapters/`.

## Instructions for AI documentation authors

Treat this directory as a methods-and-usage manual for scientist-developers, not as an
architecture index or API name list. Write for a reader who must understand, audit, and
correctly apply the numerical method, not merely discover callable names. Assume the
reader has the installed distribution and this manual, but should not have to consult
source code, tests, architecture pages, or downstream repositories to understand or use
a documented capability correctly.

### Required content

For every documented capability, include all information needed to perform the operation:

1. Begin with scientific/developer context: explain in plain language what problem the
   method solves, what kinds of prepared data it compares, what transformation it is
   allowed to choose, why a user would choose it, and what nearby problems it does not
   solve. Define specialist terminology before presenting formulas.
2. State the mathematical problem, dimensions, notation, constraints, assumptions,
   orientation, and returned quantity.
3. Explain the method: derive or justify the reduction from the stated objective to the
   implemented procedure, explain why the selected solution is optimal, and identify
   when it is unique. Give the exact procedure in execution order, including
   decomposition, selection, cutoff, reconstruction, validation, and edge-case formulas
   where they are public behavior.
4. State installation extras and the supported public import path.
5. State exact accepted representations, types, dtypes, shapes, finiteness rules, and
   empty-input behavior. Say explicitly when implicit coercion is rejected.
6. Explain every supported construction path, including ownership and copying:
   defensive-copy factories, zero-copy factories, and canonical immutable-storage
   requirements when applicable.
7. Provide a complete, executable usage example that constructs inputs, invokes the
   public API, handles status, applies the result in the correct orientation, and reads
   the evidence needed for interpretation.
8. Describe result fields, their shapes or units, their mathematical meaning, and the
   relations users may rely on.
9. Describe rank-deficient, nonunique, empty, and failure behavior. Distinguish a
   successful diagnostic result from an exception.
10. List public exception classes by inducing condition. Exact exception message text is
   not documentation unless another project instruction explicitly requires it.
11. Explain numerical resolution, conditioning limitations, and validation evidence.
    State why scaling, cutoffs, or fail-closed checks are used and what information they
    cannot establish. A reported zero, full-rank status, or small residual must not be
    presented as a scientific conclusion.
12. Cite primary papers, standard monographs, or authoritative software references for
    the mathematical and numerical techniques. Prefer DOI-linked primary sources for
    named methods and established numerical-analysis texts for stability claims. A
    citation supplements but never replaces the manual's own explanation of the method.

Do not force users to reconstruct essential procedures from scattered architecture
documents. Cross-references may provide background, but the manual itself must contain
the complete operational and mathematical instructions.

### Sources of truth

Before editing a chapter:

- read the current public production contract and concrete implementation completely;
- read the corresponding `index.md`, `implementation.md`, `numerics.md`,
  `schematics.md`, and `scientific.md` architecture pages;
- inspect public exports and tests for exact names, dtypes, shapes, defaults, and edge
  cases; and
- resolve any disagreement in favor of current qualified production behavior, then
  update stale architecture documentation separately rather than copying a conflict into
  the manual.

Never invent an API, backend, guarantee, tolerance, scientific meaning, or compatibility
promise. Document an unavailable backend explicitly as unavailable rather than implying
future or inferred support.

### Mathematical and scientific boundaries

Use consistent notation and define every symbol before use. Preserve operation order
when it affects public numerical behavior. For Procrustes capabilities, for example,
retain the documented right action, `source` adjoint times `target` cross matrix,
`left @ right_adjoint` selection, additive rank cutoff, strict-greater-than comparison,
and squared-residual roundoff cutoff.

Keep consumer-owned meaning outside Optimization. Units, operators, lattice topology,
scientific thresholds, dispositions, validation claims, and human acceptance belong to
the appropriate consumer. Clearly distinguish:

- software validation of inputs and invariants;
- bounded numerical verification of an implementation; and
- scientific validation or acceptance.

### Examples and style

- Use only supported public façades in ordinary examples.
- Make examples executable and deterministic.
- Use exact required Python and NumPy types; do not rely on undocumented coercion.
- Show fail-closed access when uniqueness is required.
- Prefer semantic variable names over unexplained literals.
- Explain advanced construction separately from the normal path.
- Use `\path{...}` for long code identifiers in prose and `lstlisting` for code.
- Escape LaTeX special characters outside verbatim/listing environments.
- Keep chapters user-oriented; source-tree layout, private helper records, and test-file
  inventories belong in architecture documentation unless they change how users operate
  the public capability.

## Build and verification

Build reproducibly from this directory:

```bash
SOURCE_DATE_EPOCH=1 TZ=UTC pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
SOURCE_DATE_EPOCH=1 TZ=UTC pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
```

Run both passes so the table of contents and references are stable. Review the log for
undefined references, missing glyphs, and overfull boxes; do not treat a zero exit status
alone as sufficient. Inspect the rendered PDF for clipped formulas, listings, page
breaks, and navigable contents.

When source changes are accepted, rebuild and stage `user_manual.pdf` with the exact
LaTeX sources that produced it. Do not hand-edit generated auxiliary files, and do not
commit them.

The manual documents current library contracts. Consumer scientific policy remains in
consumer repositories and is described here only as an external responsibility.
