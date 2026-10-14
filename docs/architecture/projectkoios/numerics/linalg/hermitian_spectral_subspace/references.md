# Hermitian spectral-subspace references

## Inspection record

These sources were inspected on 2026-10-14 rather than copied from an unverified
secondary bibliography. DOI metadata was checked against Crossref; the Springer,
Johns Hopkins University Press, Elsevier, author, and Netlib/LAPACK pages were also read
where available. `scripts/check_reference_identifiers.py` validates the DOI/ISBN syntax
and checksums, while `scripts/inspect_reference_metadata.py` retrieves normalized DOI or
ISBN metadata for human inspection.

The references establish mathematical and numerical-analysis context. They do not
replace the explicit ProjectKoios software contract, define its tolerances, or establish
scientific validation.

## Academic references

- Rajendra Bhatia, *Matrix Analysis*, Graduate Texts in Mathematics 169, Springer,
  1997. [DOI 10.1007/978-1-4612-0653-8](https://doi.org/10.1007/978-1-4612-0653-8).
  The inspected publisher description identifies the text as a systematic treatment of
  finite-dimensional matrix and operator methods. It supports the spectral-theorem,
  unitary-invariance, projector, and matrix-norm vocabulary used here.
- Tosio Kato, *Perturbation Theory for Linear Operators*, second edition, Springer,
  1995. [DOI 10.1007/978-3-642-66282-9](https://doi.org/10.1007/978-3-642-66282-9).
  The inspected edition explicitly covers finite-dimensional perturbation and spectral
  theory. It supports treating an isolated spectral subspace and its projector as the
  invariant object rather than treating one eigenvector basis as canonical.
- Chandler Davis and W. M. Kahan, “The Rotation of Eigenvectors by a Perturbation. III,”
  *SIAM Journal on Numerical Analysis* 7(1), 1–46, 1970.
  [DOI 10.1137/0707001](https://doi.org/10.1137/0707001). The inspected deposited
  abstract asks how invariant subspaces change under Hermitian perturbations and states
  that the bounds depend on perturbation or residual size and on the separating spectral
  gap. This directly motivates boundary isolation and projector-based comparison.
- G. W. Stewart and Ji-guang Sun, *Matrix Perturbation Theory*, Academic Press, 1990.
  ISBN 978-0-12-670230-9. The inspected publisher page identifies dedicated treatments
  of norms and metrics, eigenvalue perturbation, invariant subspaces, and generalized
  eigenvalue problems. It supports the separation between eigenvalue ordering,
  subspace conditioning, and coordinate-dependent bases.
- Nicholas J. Higham, “Computing a Nearest Symmetric Positive Semidefinite Matrix,”
  *Linear Algebra and its Applications* 103, 103–118, 1988.
  [DOI 10.1016/0024-3795(88)90223-6](https://doi.org/10.1016/0024-3795(88)90223-6).
  The inspected record and publisher description use the symmetric part
  $(A+A^T)/2$ as the structured representative in the real problem. The present
  architecture uses the complex-Hermitian analogue $(A+A^\dagger)/2$ explicitly and
  reports the adjustment instead of allowing a backend to choose one triangle silently.
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, fourth edition, Johns
  Hopkins University Press, 2013.
  [DOI 10.56021/9781421407944](https://doi.org/10.56021/9781421407944). The inspected
  publisher table of contents includes matrix norms, subspace metrics, finite-precision
  computation, invariant-subspace computation, symmetric eigenvalue problems, and large
  sparse eigenvalue problems. Crossref deposits 2012 while the fourth-edition publisher
  citation is 2013; this page uses the edition's publication citation. The source
  supports the dense-decomposition and residual terminology while also making clear that
  sparse iterative methods are a separate algorithmic family.
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition,
  SIAM, 2002.
  [DOI 10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027). The inspected
  author and DOI records identify the finite-precision error-analysis reference used for
  scale-aware residual checks and for distinguishing bounded numerical verification from
  exact arithmetic claims.
- E. Anderson, Z. Bai, C. Bischof, L. S. Blackford, J. Demmel, J. Dongarra, J. Du Croz,
  A. Greenbaum, S. Hammarling, A. McKenney, and D. Sorensen, *LAPACK Users’ Guide*,
  third edition, SIAM, 1999.
  [DOI 10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604). The inspected
  Netlib edition lists complex Hermitian eigensystem drivers and explains that
  eigenvector error depends inversely on eigenvalue separation. It supports recording
  API-level backend provenance and treating convergence failure and subspace conditioning
  explicitly; it does not authenticate an implementation or identify a linked library.
