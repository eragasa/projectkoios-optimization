# ResidualReconstruction numerics

The reconstruction evaluates source power plus target power minus twice the complete
cross nuclear norm. Scaling precedes subtraction. The roundoff bound uses the normalized
absolute contribution sum and a dimension-aware reduction factor.

Cancellation-scale values become exact zero; materially negative or nonfinite values
raise `FloatingPointError`.
