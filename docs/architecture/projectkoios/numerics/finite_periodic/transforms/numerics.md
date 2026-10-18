# Centered uniform transform numerics

## Mesh convention

For dimension \(d\in\{1,2,3\}\), common positive axis size \(N\), and retained offsets
\(s_a\), the axis coordinates are

\[
 q_{i,a}=-\frac12+\frac{i_a+s_a}{N},\qquad i_a=0,\ldots,N-1.
\]

The parity-aware centered integer axis is

\[
 \mathcal I_N=\{-\lfloor N/2\rfloor,\ldots,\lfloor(N-1)/2\rfloor\}.
\]

For even $N$ this is the half-open inventory
$\{-N/2,\ldots,N/2-1\}$ with one lower Nyquist endpoint. For odd $N$ it is the
symmetric inventory
$\{-(N-1)/2,\ldots,(N-1)/2\}$ with no Nyquist mode. The complete ordered inventory is
the tensor product \(\mathcal I_N^d\) in NumPy `ij` lexicographic order.

## Forward transform

For square complex matrix samples \(F(q_i)\),

\[
 C_n=\frac{1}{N^d}\sum_i F(q_i)\exp(-2\pi i\,q_i\cdot n).
\]

NumPy's forward-normalized FFT provides the phase for the index contribution \(i/N\).
The implementation then multiplies by

\[
 \exp(-2\pi i\,q_0\cdot n),
 \qquad q_{0,a}=-\frac12+\frac{s_a}{N},
\]

so nonzero offsets are never silently identified with an origin-zero mesh.

## Evaluation

A complete or selected series is evaluated by

\[
 \widetilde F(q_p)=\sum_{j=1}^{M} C_{n_j}
 \exp(+2\pi i\,q_p\cdot n_j).
\]

Coordinate components accumulate in declared component order. Matrix entries share each
phase calculation as independent coefficient channels. Point batches bound the
point-by-mode phase work by a declared maximum element count; no dense matrix spanning
all points and modes is required.

For a Cartesian-product evaluation grid, the separable path embeds coefficients in the
centered construction box and applies one positive-exponent phase matrix per coordinate
axis. Axis-wise contractions are algebraically identical to the finite sum above.
Duplicate inventory rows contribute additively. Batching the first coordinate axis
bounds each phase matrix and contraction intermediate while preserving the complete
requested output.

## Parseval identity

For the complete normalized transform,

\[
 \frac{1}{N^d}\sum_i\lVert F(q_i)\rVert_F^2
 =\sum_{n\in\mathcal I_N^d}\lVert C_n\rVert_F^2.
\]

The analyzer reports both represented sides and their absolute difference. It does not
normalize by a physical energy scale or choose a tolerance.

## Cyclic Hermiticity

Suppose each sampled matrix is Hermitian. For a represented mode \(n\), let its centered
cyclic negative be

\[
 \bar n=-n+wN,
\]

where each signed component of \(w\) records a wrap into \(\mathcal I_N\). The retained
mesh origin implies

\[
 C_{\bar n}=\exp(-2\pi i\,N q_0\cdot w)C_n^\dagger.
\]

This is the twisted cyclic relation measured over the complete inventory. For odd $N$,
ordinary inversion remains inside the symmetric inventory, $w=0$ for every mode, and
the ordinary diagnostic covers the complete inventory. For even $N$, away from the
lower Nyquist boundary hyperplanes, $w=0$ and the ordinary relation is
\(C_{-n}=C_n^\dagger\).

For the conventional even Monkhorst--Pack offset \(s_a=1/2\), each single-axis Nyquist
wrap contributes the familiar factor \(-1\).

## Failure classification

- `TypeError`: wrong nominal type, exact built-in type, array class, or dtype.
- `ValueError`: invalid dimension, nonpositive size, malformed shape, mutable storage,
  nonfinite public input, or inconsistent result correlation.
- `FloatingPointError`: nonfinite FFT, phase, reduction, Parseval, or Hermiticity
  arithmetic.

Exact exception text is not a scientific contract.
