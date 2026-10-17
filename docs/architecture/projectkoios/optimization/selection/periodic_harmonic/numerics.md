# Periodic harmonic numerics

## Closed-form maximum

For

$$
f(\phi)=o+c\cos(m\phi)+s\sin(m\phi),
$$

define

$$
A=\operatorname{hypot}(c,s).
$$

The mathematical maximum is

$$
f_{\max}=o+A.
$$

When $A$ is not tolerance-qualified as numerically constant, define

$$
\theta=\operatorname{atan2}(s,c),
\qquad
\phi_0=\theta/m.
$$

Every maximizer is $\phi_0+2\pi k/m$. The implementation reduces $\phi_0$
deterministically into the declared half-open fundamental interval
$[\alpha,\alpha+2\pi/m)$.

## Degeneracy rule

The maximizer owns two finite nonnegative tolerance components, at least one of which is
positive. It calculates

$$
t_A=a+r\max(1,|o|,A),
$$

where $a$ and $r$ are the absolute and relative degeneracy tolerances. If $A\le t_A$,
the status is `numerically-constant` and no maximizing argument is returned. Otherwise
the status is `unique` and the canonical argument is returned.

The frozen result accepts and validates canonical exact binary64 fields without
postconstruction replacement or an alternate normalization constructor.

The status is a numerical resolution statement, not a claim that $c=s=0$ exactly.
`maximum_value` remains $o+A$ in either status, so the result does not silently replace
a small nonzero amplitude by zero.

## Ordering and boundary behavior

The interval start is included and its endpoint one period later is excluded. If
floating-point reduction produces the excluded endpoint, the implementation maps it to
the included start. This specifically makes a phase located on the periodic boundary
independent of the sign of a zero-valued sine coefficient.

## Failure behavior

- Boolean and non-real scalar inputs are rejected.
- NaN and infinity are rejected.
- The harmonic order must be a positive unsigned 32-bit integer.
- The fundamental period and interval endpoint must be distinguishable in binary64.
- Nonfinite amplitude, maximum-value, cutoff, phase, or canonicalization arithmetic
  raises `FloatingPointError`.

The calculation is closed form. It performs no iteration, sampling, local search, or
optional-backend call.
