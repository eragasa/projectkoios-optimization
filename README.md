# Project Koios Optimization

`projectkoios-optimization` owns dependency-free optimization contracts and
deterministic bounded algorithms under `projectkoios.optimization`.

The package provides explicit ask/tell engines, versioned checkpoints,
SplitMix64 uniform-box sampling, weighted squared loss, bounded coordinate
pattern search, and pure ranking/Pareto reports. Forward evaluation and result
handling remain separate injected ports. The package does not execute
calculators, filesystems, subprocesses, networks, workflows, dispatch, retries,
or persistence.

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
