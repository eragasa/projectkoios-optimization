# Project Koios Optimization

`projectkoios-optimization` owns dependency-free optimization contracts and
deterministic bounded algorithms under `projectkoios.optimization`.

The package provides explicit ask/tell engines, versioned checkpoints,
SplitMix64 uniform-box sampling, a generic sampler contract, slot-composed
probability sampling with CDF/inverse-CDF behavior, weighted squared loss, bounded coordinate
pattern search, and pure ranking/Pareto reports. Forward evaluation and result
handling remain separate injected ports. The package does not execute
calculators, filesystems, subprocesses, networks, workflows, dispatch, retries,
or persistence.

The package root and core probability contracts are dependency-free. Explicit
optional adapters are available through the `numpy` and `scipy` extras;
they are imported only from `projectkoios.optimization.adapters.numpy` and
`projectkoios.optimization.adapters.scipy`.

See [`TRANSFER.toml`](TRANSFER.toml) and
[`docs/provenance/source-mapping.md`](docs/provenance/source-mapping.md) for the
recovery-source and design-evidence mapping. Verification is software and
bounded numerical verification only; it is not scientific validation.

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/python/projectkoios/optimization
python -m build --sdist --wheel
```
