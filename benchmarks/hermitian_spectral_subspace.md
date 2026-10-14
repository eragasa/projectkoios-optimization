# Hermitian spectral-subspace numerical calibration

The mapped numerical test
`test__NumpyHermitianSpectralSubspaceSolver__numerical.py` evaluates 256 deterministic
complex Hermitian matrices with dimensions one through eight. Each binary64 input is
converted exactly through `float.as_integer_ratio()` and independently decomposed with
`mpmath.eigh` at 100 and 160 decimal digits.

The test verifies high-precision eigenpair residual, orthonormality, reconstruction,
precision convergence, and selected-projector convergence before comparing the NumPy
implementation with the 160-digit reference. Internal frames are never compared across a
degenerate block; only the invariant projector is compared.

Regenerate retained environment and maximum-error evidence with:

```bash
PROJECTKOIOS_HERMITIAN_CALIBRATION_OUTPUT=\
benchmarks/results/hermitian_spectral_subspace_calibration.json \
PYTHONPATH=src/python .venv/bin/python -m pytest -q \
tests/projectkoios/numerics/linalg/numpy/hermitian_spectral_subspace/\
test__NumpyHermitianSpectralSubspaceSolver__numerical.py::\
test_256_seeded_cases_agree_with_converged_mpmath_oracle
```

The JSON records SHA-256 identities for the shared implementation, NumPy backend, and
numerical-test files; deterministic seeds; Python, mpmath, and NumPy versions;
`numpy.show_config()` linked-library information; the maximum generated input-entry
magnitude; and observed maximum absolute errors and acceptance-bound ratios. These file
hashes bind the evidence to the exact local implementation and calibration procedure
even when the worktree has no new commit. The current evidence is one platform calibration, not a
cross-platform stability claim. Additional CI platforms append independent qualification
evidence rather than overwriting the meaning of this record.

This is bounded software and numerical verification only. It is not validation of a
physical Hamiltonian, retained rank, material, or scientific gap criterion.
