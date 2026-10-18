# Scalar solver numerics

The scalar template forms one `source` adjoint times `target` cross matrix, validates a
complete SVD, applies the shared declared rank cutoff, reconstructs the optimum, and
returns a unitary transformation only at full rank. A deficient result withholds the
backend's arbitrary null-space completion.
