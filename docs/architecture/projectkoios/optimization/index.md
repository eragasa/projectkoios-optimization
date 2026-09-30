# `projectkoios.optimization`

## Ownership index

| Area | Current implementation | Owned behavior |
|---|---|---|
| Identity | [`identity.py`](../../../../src/python/projectkoios/optimization/identity.py) | immutable versioned optimizer identity and algorithm/checkpoint descriptor |
| Records | [`records.py`](../../../../src/python/projectkoios/optimization/records.py) | bounds, spaces, candidates, raw evaluations, feedback, updates |
| Ports | [`ports.py`](../../../../src/python/projectkoios/optimization/ports.py) | optimizer ask/tell, forward evaluator, results handler |
| Checkpoints | [`checkpoint.py`](../../../../src/python/projectkoios/optimization/checkpoint.py) | canonical JSON, schema and digest validation |
| Uniform algorithm | [`uniform.py`](../../../../src/python/projectkoios/optimization/uniform.py) | bounded SplitMix64 proposals, out-of-order tell |
| Sampling | [`sampling/`](../../../../src/python/projectkoios/optimization/sampling/) | generic behaviorful sampler contract and exact immutable sample batches |
| Probability | [`probability/`](../../../../src/python/projectkoios/optimization/probability/) | CDF/inverse-CDF products, uniform distributions, and slot-composed probability sampling |
| NumPy adapters | [`adapters/numpy/`](../../../../src/python/projectkoios/optimization/adapters/numpy/) | optional NumPy probability functions and isolated RandomState source slot |
| SciPy adapters | [`adapters/scipy/`](../../../../src/python/projectkoios/optimization/adapters/scipy/) | optional structural frozen-distribution CDF/PPF slots |
| Pattern algorithm | [`pattern.py`](../../../../src/python/projectkoios/optimization/pattern.py) | bounded coordinate local search, resumable ask/tell |
| Loss | [`loss.py`](../../../../src/python/projectkoios/optimization/loss.py) | positive-weight finite squared residual aggregation |
| Reports | [`reports.py`](../../../../src/python/projectkoios/optimization/reports.py) | deterministic scalar ranking and Pareto membership |

The core namespace is dependency-free and has no calculator, filesystem,
subprocess, network, dispatch, retry, persistence, or workflow authority. NumPy
and SciPy bindings are explicit optional backend adapter modules and are not
imported by the package root. Atomistic and potential models, campaigns, GMM,
PyPosPack, PyPosMat, and PyMatMC2 are outside this repository.

See [source mapping](../../../provenance/source-mapping.md) for exact recovery
and architecture source-lineage evidence.
