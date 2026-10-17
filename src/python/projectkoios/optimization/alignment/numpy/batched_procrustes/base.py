"""Define the supported NumPy batched Procrustes façade.

Notes
-----
This module owns no numerical implementation. It is the reviewed export boundary for the
nominal batch solver, exact definition and result records, and concrete NumPy solver.
"""

from .contract import (
    BatchedComplexUnitaryProcrustesSolver,
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesResult,
)
from .solver import NumpyBatchedComplexUnitaryProcrustesSolver

__all__ = [
    "BatchedComplexUnitaryProcrustesSolver",
    "NumpyBatchedComplexUnitaryProcrustesDefinition",
    "NumpyBatchedComplexUnitaryProcrustesResult",
    "NumpyBatchedComplexUnitaryProcrustesSolver",
]
