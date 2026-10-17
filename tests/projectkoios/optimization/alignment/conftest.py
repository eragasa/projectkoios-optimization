from __future__ import annotations

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes.numerics import (
    ResidualReconstruction,
)
from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesDecomposition,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesTolerance,
)
from tests.projectkoios.optimization.alignment.fixtures import (
    BatchedProcrustesEvidenceFixture,
    UniqueScalarResultCase,
    unique_scalar_result_case,
)


@pytest.fixture
def unique_scalar_result() -> UniqueScalarResultCase:
    """Return a named analytically derived scalar result fixture."""
    return unique_scalar_result_case()


@pytest.fixture
def scalar_identity_decomposition() -> ComplexUnitaryProcrustesDecomposition:
    """Return a named complete scalar decomposition fixture."""
    dimension = 2
    identity = tuple(
        tuple(complex(int(row == column), 0.0) for column in range(dimension))
        for row in range(dimension)
    )
    singular_values = tuple(float(dimension - index) for index in range(dimension))
    return ComplexUnitaryProcrustesDecomposition(
        singular_values=singular_values,
        left_singular_vectors=identity,
        right_singular_vectors_adjoint=identity,
    )


@pytest.fixture
def procrustes_rank_tolerance() -> ComplexUnitaryProcrustesTolerance:
    """Return the shared additive-rank-policy fixture."""
    return ComplexUnitaryProcrustesTolerance(
        rank_absolute=0.25,
        rank_relative=0.5,
    )


@pytest.fixture
def procrustes_evidence() -> BatchedProcrustesEvidenceFixture:
    """Return reusable evidence constructed by the owning numerical policies."""
    spectra = np.array([[2.0, 1.0]], dtype=np.float64)
    source_powers = np.array([3.0], dtype=np.float64)
    target_powers = np.array([3.0], dtype=np.float64)
    reconstruction = ResidualReconstruction.from_procrustes_evidence(
        source_powers,
        target_powers,
        spectra,
    )
    rank_cutoffs = np.array(
        [
            ComplexUnitaryProcrustesSolver.rank_cutoff(
                DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
                float(spectra[0, 0]),
            )
        ],
        dtype=np.float64,
    )
    return BatchedProcrustesEvidenceFixture(
        completions=np.eye(2, dtype=np.complex128)[None, ...],
        spectra=spectra,
        ranks=np.array([2], dtype=np.intp),
        full_rank=np.array([True], dtype=np.bool_),
        rank_cutoffs=rank_cutoffs,
        source_powers=source_powers,
        target_powers=target_powers,
        squared_residuals=reconstruction.squared_residuals,
        residual_norms=np.asarray(
            np.sqrt(reconstruction.squared_residuals),
            dtype=np.float64,
        ),
        residual_cutoffs=reconstruction.roundoff_cutoffs,
    )
