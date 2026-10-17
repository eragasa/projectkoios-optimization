from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BatchedProcrustesEvidenceFixture:
    """Group reusable, internally consistent batch evidence arrays."""

    completions: np.ndarray
    spectra: np.ndarray
    ranks: np.ndarray
    full_rank: np.ndarray
    rank_cutoffs: np.ndarray
    source_powers: np.ndarray
    target_powers: np.ndarray
    squared_residuals: np.ndarray
    residual_norms: np.ndarray
    residual_cutoffs: np.ndarray
