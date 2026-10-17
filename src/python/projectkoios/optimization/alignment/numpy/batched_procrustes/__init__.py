"""Expose the supported NumPy batched complex-unitary Procrustes façade.

Notes
-----
Only names approved by :mod:`.base` cross this package boundary. Contract records,
numerical evidence helpers, and backend construction records remain encapsulated in
their implementation modules.
"""

from .base import (
    BatchedComplexUnitaryProcrustesSolver,
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)

__all__ = [
    "BatchedComplexUnitaryProcrustesSolver",
    "NumpyBatchedComplexUnitaryProcrustesDefinition",
    "NumpyBatchedComplexUnitaryProcrustesResult",
    "NumpyBatchedComplexUnitaryProcrustesSolver",
]
