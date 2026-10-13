# Positive diagonal metric geometry

## Purpose

`PositiveDiagonalMetric` is an immutable, dependency-free, unit-free representation of
one positive diagonal metric

$$
M=\operatorname{diag}(m_1,\ldots,m_n), \qquad m_i>0.
$$

It evaluates squared distance, distance, squared dual norm, and dual norm. Tuple order
is the coordinate convention; scientific names and channel alignment remain outside the
numerical object. The zero displacement and zero covector are valid.

## Binary64 behavior

Public inputs are exact tuples of finite non-Boolean real values with the metric's exact
dimension. Squared operations use square-root rescaling when direct squaring would
prematurely overflow or underflow. Norm operations use `math.hypot` on scaled
components to avoid premature squaring. Nonfinite differences, components, sums, and
results fail closed.

These policies preserve representable compensated scales; they do not provide interval
arithmetic, directed rounding, or an exact-real certificate.

## Ownership boundary

Physical callers must nondimensionalize coordinates and metric entries before crossing
this API. The metric assigns no units, physical meaning, separating-direction policy,
threshold, feasibility meaning, certificate interpretation, or scientific disposition.
