# Rank-cutoff rule numerics

For largest cross singular value $\sigma_{\max}$, absolute control $a$, and relative
control $r$, the applied cutoff is selected explicitly.

- `ADDITIVE` gives $a+r\sigma_{\max}$.
- `MAXIMUM` gives $\max(a,r\sigma_{\max})$.
- Singular values count toward numerical rank only when they are strictly greater than
  the applied cutoff.
- Equality is therefore rank deficient under either rule.
