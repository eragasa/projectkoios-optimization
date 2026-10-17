# Alignment consumer needs

Consumers use alignment to compare mathematical representations whose scientific meaning
remains outside Optimization.

- PhysKit may attach basis, operator, lattice, or hopping meaning before and after calling
  an alignment API.
- KSD may choose scientific rank thresholds, decide whether deficient evidence makes an
  interpretation inapplicable, and persist evidence required by a scientific protocol.
- Consumers must preserve orientation, index topology, units, and provenance themselves.
- Consumers must not infer scientific validity from software validation, a small residual,
  or a full-rank numerical result alone.

Optimization receives arrays and numerical controls only. Consumer frame semantics,
closure meaning, dispositions, and human acceptance never enter this library boundary.
