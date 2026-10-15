# Complex-unitary right Procrustes alignment

## Contract

For equally shaped nonempty finite complex matrices $A,B\in\mathbb{C}^{m\times n}$,
the capability solves

$$
\min_{Q\in\mathbb{C}^{n\times n},\;Q^\dagger Q=I}
\lVert A Q-B\rVert_F.
$$

`source_matrix` is $A$, `target_matrix` is $B$, and `transformation` is $Q$. Thus a
consumer aligns a raw/source frame toward an accepted/target frame by applying
`source @ transformation`.

## Rank and uniqueness

Let

$$
A^\dagger B=U\Sigma V^\dagger.
$$

The unique full right-unitary polar factor is $Q=UV^\dagger$ when the cross matrix is
full rank. The applied numerical-rank cutoff is

$$
\tau_{\mathrm{rank}}=	au_{\mathrm{abs}}
+\tau_{\mathrm{rel}}\sigma_{\max}.
$$

Singular values strictly greater than the cutoff count toward `numerical_rank`. At full
numerical rank the result status is `UNIQUE` and contains $Q$. Otherwise the status is
`RANK_DEFICIENT` and `transformation` is `None`; the implementation does not expose a
unitary assembled from arbitrary null-space singular vectors.

## Evidence

Every result retains:

- singular values of $A^\dagger B$;
- numerical rank and the applied cutoff;
- squared Frobenius norms of $A$ and $B$;
- the optimal squared residual, its square root, and the applied reconstruction-roundoff
  cutoff; and
- trusted API-level backend identity/version strings.

The optimum is independently reconstructable from

$$
r_*^2=\lVert A\rVert_F^2+\lVert B\rVert_F^2
-2\sum_i\sigma_i(A^\dagger B).
$$

The result does not establish a scientific rank threshold, frame protocol, path
continuity, or periodic closure.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Alignment namespace](../index.md)
