# Solver numerics

The solver uses `source.conj().swapaxes(-1, -2) @ target`, selects `left @ right_adjoint`,
and validates factor unitarity, decomposition reconstruction, completion unitarity, and
direct-versus-reconstructed residuals. Rank uses the owner-selected additive or maximum
cutoff with strict greater-than comparison.
