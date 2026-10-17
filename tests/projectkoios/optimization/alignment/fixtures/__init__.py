"""Named fixtures for scalar and batched Procrustes verification."""

from .batched_procrustes import BatchedProcrustesEvidenceFixture
from .scalar_procrustes import UniqueScalarResultCase, unique_scalar_result_case

__all__ = [
    "BatchedProcrustesEvidenceFixture",
    "UniqueScalarResultCase",
    "unique_scalar_result_case",
]
